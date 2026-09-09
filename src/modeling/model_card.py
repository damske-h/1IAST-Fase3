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
comparação de algoritmos e a busca em grade.
"""

import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split

from ..evaluation import avaliar, limiar_por_recall
from ..preprocessing import carregar_base_ml
from ..preprocessing import config as cfg
from . import features as F
from .alvo import ANO_MODELAGEM, LIMIAR_RISCO, rotular_risco
from .features import construir_pipeline, modelos_candidatos, nomes_das_features

SEMENTE = 42
GRADE_C = [0.01, 0.1, 1.0, 10.0, 100.0]
DESTINO = cfg.RAIZ / "reports" / "model_card.json"

# A classe de risco é minoritária: a métrica que decide é a PR-AUC, não a acurácia.
METRICAS_CV = ["roc_auc", "average_precision", "recall", "precision"]


def _particoes(X, y):
    """Holdout estratificado de 25%, tocado só na avaliação final."""
    return train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEMENTE)


def _comparacao_de_algoritmos(X_treino, y_treino):
    """Todos os candidatos, mesma pipeline e a mesma validação estratificada.

    A comparação usa métricas **independentes de limiar** (ROC-AUC e PR-AUC):
    recall e precisão dependem do corte de decisão, que só é escolhido depois.
    """
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMENTE)
    resultados = []

    for nome, estimador in modelos_candidatos(random_state=SEMENTE).items():
        inicio = time.perf_counter()
        pontuacoes = cross_validate(
            construir_pipeline(estimador=estimador), X_treino, y_treino,
            cv=folds, scoring=METRICAS_CV, return_train_score=True, n_jobs=-1)
        resultados.append({
            "modelo": nome,
            "roc_auc": round(float(pontuacoes["test_roc_auc"].mean()), 4),
            "roc_auc_desvio": round(float(pontuacoes["test_roc_auc"].std()), 4),
            "pr_auc": round(float(pontuacoes["test_average_precision"].mean()), 4),
            "gap_treino_validacao": round(float(
                pontuacoes["train_roc_auc"].mean() - pontuacoes["test_roc_auc"].mean()), 4),
            "segundos": round(time.perf_counter() - inicio, 1),
        })

    return sorted(resultados, key=lambda r: r["pr_auc"], reverse=True)


def _busca_em_grade(X_treino, y_treino):
    """`GridSearchCV` sobre a regularização, otimizando PR-AUC."""
    busca = GridSearchCV(
        construir_pipeline(), {"modelo__C": GRADE_C},
        scoring="average_precision",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMENTE),
        n_jobs=-1)
    busca.fit(X_treino, y_treino)
    return busca


def gerar(destino: Path = DESTINO) -> dict:
    """Monta o model card e grava o JSON."""
    base = carregar_base_ml().drop(columns=["_gold_processed_at"])
    X, y = rotular_risco(base)
    X_treino, X_teste, y_treino, y_teste = _particoes(X, y)

    comparacao = _comparacao_de_algoritmos(X_treino, y_treino)
    busca = _busca_em_grade(X_treino, y_treino)
    melhor_C = busca.best_params_["modelo__C"]

    modelo_final = busca.best_estimator_
    probabilidade = modelo_final.predict_proba(X_teste)[:, 1]
    no_teste = avaliar(y_teste, modelo_final.predict(X_teste), probabilidade)

    # O modelo publicado é reajustado sobre todos os municípios do ciclo.
    modelo_publicado = construir_pipeline(C=melhor_C).fit(X, y)

    card = {
        "modelo": "Regressão Logística (L2) — classificação binária de risco municipal",
        "gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "premissa": (
            "não existe microdado público por aluno: nenhuma das fontes ingeridas traz "
            "identificador de estudante. A chance de a criança ser alfabetizada é tratada como "
            "dada pelo contexto socioeconômico e educacional do município em que ela estuda, e é "
            "esse contexto que o modelo descreve"
        ),
        "dados": {
            "origem": "camada Gold local — data/lake/gold/base_ml_alfabetizacao",
            "linhas_no_ciclo_modelado": int(len(X)),
            "municipios": int(X["id_municipio"].nunique()),
            "unidades_da_federacao": int(X["sigla_uf"].nunique()),
            "ano_modelado": ANO_MODELAGEM,
            "anos_disponiveis_na_base": sorted(int(a) for a in base["ano"].unique()),
            "rede": "municipal (código 3)",
            "cobertura_declarada": ("25 das 27 UFs — o Distrito Federal não tem rede "
                                    "municipal e Roraima está ausente do arquivo do INEP"),
        },
        "alvo": {
            "definicao": f"em_risco = taxa_alfabetizacao < {LIMIAR_RISCO:.0f}%",
            "justificativa_do_corte": (
                "o município onde menos da metade das crianças chega alfabetizada. Descartadas: "
                "a meta pactuada (deriva da taxa de 2023 do próprio município, logo circular) e a "
                "média nacional (relativa — metade do país estaria em risco melhorasse ou não)"
            ),
            "municipios_em_risco": int(y.sum()),
            "prevalencia_pct": round(float(y.mean() * 100), 2),
        },
        "features": {
            "colunas_de_entrada": len(F.colunas_do_modelo()),
            "features_apos_preprocessamento": len(nomes_das_features(modelo_publicado)),
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
            "C": melhor_C,
            "grade_testada": GRADE_C,
            "criterio_da_busca": "average_precision (PR-AUC)",
            "pr_auc_da_melhor_configuracao": round(float(busca.best_score_), 4),
            "penalidade": "L2 (padrão do LogisticRegression)",
            "solver": "lbfgs",
            "max_iter": 2000,
            "random_state": SEMENTE,
            "class_weight": (
                "não usado — testado e rejeitado: 'balanced' não altera o ROC-AUC e piora a "
                "calibração. O desbalanceamento é tratado na escolha do limiar"
            ),
        },
        "validacao": {
            "particao": "train_test_split estratificado 75/25",
            "selecao_e_ajuste": "StratifiedKFold de 5 folds sobre o treino",
            "observacao": (
                "modelagem restrita ao ciclo de 2024, uma linha por município — o Rio Grande do "
                "Sul caiu 20,2 p.p. nesse ciclo por choque exógeno (enchentes de 2024) que "
                "nenhuma variável da base descreve, de modo que o risco previsto para o estado "
                "está superestimado"
            ),
        },
        "metricas": {
            "no_teste": no_teste,
            "limiar_por_cobertura": {
                f"recall_{int(r * 100)}pct": limiar_por_recall(y_teste, probabilidade, r)
                for r in (0.6, 0.7, 0.8, 0.9)
            },
        },
        "comparacao_de_algoritmos": comparacao,
        "limitacoes": [
            "falácia ecológica — descreve o município, não uma criança: dois alunos do mesmo "
            "município recebem a mesma predição",
            "sem variáveis de política educacional — o maior efeito do modelo (a UF) fica sem "
            "explicação dentro da base",
            "INSE de 2023 replicado para 2024 como atributo estrutural",
            "AFD/ATU/IED usam o agregado Total, que inclui a rede privada",
            "um único ciclo modelado — não distingue tendência de choque, e o ciclo escolhido "
            "contém o choque do Rio Grande do Sul",
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
    print(f"model card gravado em {DESTINO}")
    print(f"  C escolhido      : {resultado['hiperparametros']['C']}")
    print(f"  ROC-AUC no teste : {resultado['metricas']['no_teste']['roc_auc']}")
    print(f"  PR-AUC no teste  : {resultado['metricas']['no_teste']['pr_auc']}")
    print(f"  melhor algoritmo : {resultado['comparacao_de_algoritmos'][0]['modelo']}")
