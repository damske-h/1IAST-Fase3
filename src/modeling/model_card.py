"""
Gerador do *model card* — a ficha técnica reproduzível do modelo.

Um model card registra, num único arquivo versionável, **o que foi treinado, com
que dados, com que resultado e sob quais limites**. Serve a três propósitos:

1. **Proveniência** — se a camada Gold mudar, o card anterior diz exatamente
   qual base produziu quais números;
2. **Auditoria** — as métricas ficam registradas por execução, não apenas
   dentro da saída de um notebook;
3. **Disciplina** — nenhum número do relatório técnico é digitado à mão: todos
   saem daqui.

Uso:

    python -m src.modeling.model_card

Gera `reports/model_card.json`. Leva cerca de um minuto, porque reexecuta a
validação cruzada e a comparação de algoritmos.
"""

import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn
from sklearn.model_selection import GroupKFold

from ..evaluation import avaliar, validar_cruzado
from ..preprocessing import carregar_base_ml
from ..preprocessing import config as cfg
from . import features as F
from .alvo import expandir_binomial
from .features import construir_pipeline, modelos_candidatos, nomes_das_features

SEMENTE = 42
C_ESCOLHIDO = 1.0
DESTINO = cfg.RAIZ / "reports" / "model_card.json"


def _metricas_validacao_cruzada(X, y, peso, grupos):
    """Métricas fora do fold, com `GroupKFold` por município."""
    probabilidade = np.empty(len(y))
    for treino, validacao in GroupKFold(n_splits=5).split(X, y, grupos):
        modelo = construir_pipeline(C=C_ESCOLHIDO)
        modelo.fit(X.iloc[treino], y[treino], modelo__sample_weight=peso[treino])
        probabilidade[validacao] = modelo.predict_proba(X.iloc[validacao])[:, 1]
    return avaliar(y, probabilidade, peso)


def _metricas_split_temporal(base):
    """Treina em 2023 e avalia em 2024, com e sem o Rio Grande do Sul."""
    X_treino, y_treino, w_treino, _ = expandir_binomial(base[base["ano"] == 2023])
    modelo = construir_pipeline(C=C_ESCOLHIDO)
    modelo.fit(X_treino, y_treino, modelo__sample_weight=w_treino)

    def avaliar_recorte(recorte):
        Xr, yr, wr, _ = expandir_binomial(recorte)
        return avaliar(yr, modelo.predict_proba(Xr)[:, 1], wr)

    teste = base[base["ano"] == 2024]
    return {
        "todas_as_ufs": avaliar_recorte(teste),
        "sem_rio_grande_do_sul": avaliar_recorte(teste[teste["sigla_uf"] != "RS"]),
    }


def _comparacao_de_modelos(X, y, peso, grupos):
    """Todos os candidatos, mesma pipeline e mesma validação."""
    resultados = []
    for nome, estimador in modelos_candidatos(random_state=SEMENTE).items():
        inicio = time.perf_counter()
        folds = validar_cruzado(construir_pipeline(estimador=estimador),
                                X, y, peso, grupos, GroupKFold(n_splits=5))
        resultados.append({
            "modelo": nome,
            "auc": round(float(folds["validacao_auc"].mean()), 4),
            "auc_desvio": round(float(folds["validacao_auc"].std()), 4),
            "brier": round(float(folds["validacao_brier"].mean()), 4),
            "gap_treino_validacao": round(
                float(folds["treino_auc"].mean() - folds["validacao_auc"].mean()), 4),
            "segundos": round(time.perf_counter() - inicio, 1),
        })
    return sorted(resultados, key=lambda r: r["auc"], reverse=True)


def gerar(destino: Path = DESTINO) -> dict:
    """Monta o model card e grava o JSON."""
    base = carregar_base_ml().drop(columns=["_gold_processed_at"])
    X, y, peso, grupos = expandir_binomial(base)

    modelo_final = construir_pipeline(C=C_ESCOLHIDO)
    modelo_final.fit(X, y, modelo__sample_weight=peso)

    piso = avaliar(y, np.full(len(y), peso[y == 1].sum() / peso.sum()), peso)
    teto = avaliar(y, X["taxa_alfabetizacao"].to_numpy() / 100, peso)
    validacao = _metricas_validacao_cruzada(X, y, peso, grupos)

    fracao_do_intervalo = ((validacao["auc"] - piso["auc"])
                           / (teto["auc"] - piso["auc"]) * 100)

    arredondar = lambda d: {k: round(float(v), 4) for k, v in d.items()}

    card = {
        "modelo": "Regressão Logística (L2) sobre alvo binomial ponderado",
        "gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dados": {
            "origem": "camada Gold local — data/lake/gold/base_ml_alfabetizacao",
            "linhas": int(len(base)),
            "municipios": int(base["id_municipio"].nunique()),
            "unidades_da_federacao": int(base["sigla_uf"].nunique()),
            "anos": sorted(int(a) for a in base["ano"].unique()),
            "rede": "municipal (código 3)",
            "cobertura_declarada": ("25 das 27 UFs — o Distrito Federal não tem rede "
                                    "municipal e Roraima está ausente do arquivo do INEP"),
        },
        "alvo": {
            "definicao": "dado binário agrupado (binomial ponderado)",
            "construcao": ("cada linha município×ano vira duas observações: y=1 com peso "
                           "taxa/100 e y=0 com peso 1-taxa/100"),
            "observacoes_apos_expansao": int(len(y)),
            "soma_dos_pesos": round(float(peso.sum()), 2),
            "taxa_media_implicita": round(
                float(peso[y == 1].sum() / peso.sum() * 100), 2),
        },
        "features": {
            "colunas_de_entrada": len(F.colunas_do_modelo()),
            "features_apos_preprocessamento": len(nomes_das_features(modelo_final)),
            "blocos": {
                "formacao_docente_afd": F.AFD,
                "esforco_docente_ied": F.IED,
                "niveis_inse": F.INSE_NIVEIS,
                "tamanho_de_turma_atu": F.ATU,
                "socioeconomico_outras": F.INSE_OUTRAS,
                "ideb_defasado_2021": F.NUMERICAS_IDEB,
                "categoricas": F.CATEGORICAS,
            },
            "categorias_de_referencia_descartadas": F.REFERENCIAS,
            "excluidas_com_motivo": F.EXCLUIDAS,
        },
        "tratamento_de_vazamento": {
            "colunas_mapeadas": len(cfg.COLUNAS_VAZAMENTO),
            "aplicado_em": "src/preprocessing/gold.py — montar_base_ml()",
            "colunas": sorted(cfg.COLUNAS_VAZAMENTO),
        },
        "hiperparametros": {
            "C": C_ESCOLHIDO,
            "penalidade": "L2 (padrão do LogisticRegression)",
            "solver": "lbfgs",
            "max_iter": 2000,
            "random_state": SEMENTE,
        },
        "validacao": {
            "principal": "GroupKFold por id_municipio, 5 folds",
            "generalizacao_temporal": "treino em 2023, teste em 2024",
            "observacao": ("as métricas do split temporal são reportadas com e sem o RS: "
                           "o estado sofreu queda de 20,2 p.p. em 2024, atribuída a choque "
                           "exógeno que nenhuma variável da base descreve"),
        },
        "metricas": {
            "piso_prever_a_media": arredondar(piso),
            "modelo_validacao_cruzada": arredondar(validacao),
            "teto_oraculo": arredondar(teto),
            "fracao_do_intervalo_piso_teto_pct": round(float(fracao_do_intervalo), 1),
            "split_temporal": {k: arredondar(v)
                               for k, v in _metricas_split_temporal(base).items()},
        },
        "comparacao_de_algoritmos": _comparacao_de_modelos(X, y, peso, grupos),
        "limitacoes": [
            "falácia ecológica — estima o aluno típico do município, não uma criança",
            "peso por município, não por aluno — a base não traz o número de avaliados",
            "sem variáveis de política educacional — o maior efeito do modelo (UF) não é explicável",
            "INSE de 2023 replicado para 2024 como atributo estrutural",
            "AFD/ATU/IED usam o agregado Total, que inclui a rede privada",
            "apenas dois ciclos avaliativos — tendência e choque são indistinguíveis",
            "associações condicionais, não efeitos causais",
        ],
        "reprodutibilidade": {
            "semente": SEMENTE,
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "comandos": [
                "python -m src.preprocessing.run_pipeline",
                "python -m src.modeling.model_card",
            ],
        },
    }

    destino.parent.mkdir(parents=True, exist_ok=True)
    with open(destino, "w", encoding="utf-8", newline="\n") as arquivo:
        json.dump(card, arquivo, ensure_ascii=False, indent=2)
        arquivo.write("\n")

    return card


if __name__ == "__main__":
    resultado = gerar()
    metricas = resultado["metricas"]
    print(f"model card gravado em {DESTINO}")
    print(f"  AUC validação cruzada : {metricas['modelo_validacao_cruzada']['auc']}")
    print(f"  fração do intervalo   : {metricas['fracao_do_intervalo_piso_teto_pct']}%")
    print(f"  melhor algoritmo      : {resultado['comparacao_de_algoritmos'][0]['modelo']}")
