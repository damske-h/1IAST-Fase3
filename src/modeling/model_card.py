"""
Gerador do *model card* — a ficha técnica reproduzível do modelo.

Registra num arquivo versionável **o que foi treinado, com que dados, com que
resultado e sob quais limites**. Serve à proveniência (qual base produziu quais
números), à auditoria (métricas por execução, não só dentro de um notebook) e à
disciplina — **nenhum número da documentação é digitado à mão**.

Reproduz exatamente as etapas do `notebooks/03_modelagem.ipynb`.

Uso:

    python -m src.modeling.model_card
"""

import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn
from sklearn.model_selection import (GridSearchCV, StratifiedKFold, cross_validate,
                                     train_test_split)

from ..evaluation import avaliar, limiar_por_recall
from ..preprocessing import config as cfg
from . import features as F
from .alvo import ANO_ALVO, ANO_HISTORICO, LIMIAR_RISCO, montar_painel, rotular_risco
from .features import construir_pipeline, modelos_candidatos, nomes_das_features

SEMENTE = 42
GRADE_C = [0.01, 0.1, 1.0, 10.0, 100.0]
METRICAS = ["roc_auc", "average_precision", "recall", "precision", "accuracy"]
DESTINO = cfg.RAIZ / "reports" / "model_card.json"


def gerar(destino: Path = DESTINO) -> dict:
    """Monta o model card e grava o JSON."""
    painel = montar_painel()
    X, y = rotular_risco(painel)
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEMENTE)

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMENTE)

    # ── comparação dos candidatos ────────────────────────────────────────────
    algoritmos = []
    for nome, estimador in modelos_candidatos(SEMENTE).items():
        inicio = time.perf_counter()
        r = cross_validate(construir_pipeline(estimador=estimador), X_treino, y_treino,
                           cv=folds, scoring=METRICAS, return_train_score=True, n_jobs=-1)
        algoritmos.append({
            "modelo": nome,
            "roc_auc": round(float(r["test_roc_auc"].mean()), 4),
            "pr_auc": round(float(r["test_average_precision"].mean()), 4),
            "recall_risco": round(float(r["test_recall"].mean()), 4),
            "precisao_risco": round(float(r["test_precision"].mean()), 4),
            "acuracia": round(float(r["test_accuracy"].mean()), 4),
            "gap_treino_validacao": round(
                float(r["train_roc_auc"].mean() - r["test_roc_auc"].mean()), 4),
            "segundos": round(time.perf_counter() - inicio, 1),
        })
    algoritmos.sort(key=lambda linha: linha["pr_auc"], reverse=True)

    # ── contribuição de cada bloco de variáveis ──────────────────────────────
    blocos = {"so_historico": F.HISTORICO, "so_atualidade": F.ATUALIDADE,
              "historico_e_atualidade": F.HISTORICO + F.ATUALIDADE}
    por_bloco = {}
    for nome, colunas in blocos.items():
        r = cross_validate(construir_pipeline(numericas=colunas), X_treino, y_treino,
                           cv=folds, scoring=["roc_auc", "average_precision"], n_jobs=-1)
        por_bloco[nome] = {"roc_auc": round(float(r["test_roc_auc"].mean()), 4),
                           "pr_auc": round(float(r["test_average_precision"].mean()), 4)}

    # ── otimização de hiperparâmetros ───────────────────────────────────────
    busca = GridSearchCV(construir_pipeline(), {"modelo__C": GRADE_C}, cv=folds,
                         scoring="average_precision", return_train_score=True, n_jobs=-1)
    busca.fit(X_treino, y_treino)
    grade = [
        {"C": c, "pr_auc_validacao": round(float(v), 4), "pr_auc_treino": round(float(t), 4),
         "gap": round(float(t - v), 4)}
        for c, v, t in zip(GRADE_C, busca.cv_results_["mean_test_score"],
                           busca.cv_results_["mean_train_score"])
    ]

    # ── avaliação final, uma vez só ─────────────────────────────────────────
    modelo = busca.best_estimator_
    probabilidade = modelo.predict_proba(X_teste)[:, 1]
    no_teste = avaliar(y_teste, modelo.predict(X_teste), probabilidade)

    modelo_publicado = construir_pipeline(C=busca.best_params_["modelo__C"]).fit(X, y)
    coeficientes = dict(zip(nomes_das_features(modelo_publicado),
                            np.round(modelo_publicado.named_steps["modelo"].coef_[0], 4).tolist()))
    maiores = sorted(coeficientes.items(), key=lambda par: abs(par[1]), reverse=True)[:12]

    card = {
        "modelo": "Regressão Logística (L2) — classificação binária de risco municipal",
        "gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "premissa": (
            "não existe microdado público por aluno: nenhuma fonte traz identificador de "
            "estudante. Modelamos o contexto que responde pela criança — o município entra como "
            "em risco quando menos da metade das suas crianças chega alfabetizada ao fim do 2º ano"
        ),
        "dados": {
            "origem": "camada Gold local — data/lake/gold/base_ml_alfabetizacao",
            "ano_alvo": ANO_ALVO,
            "ano_historico": ANO_HISTORICO,
            "municipios": int(len(X)),
            "unidades_da_federacao": int(X["sigla_uf"].nunique()),
            "rede": "municipal (código 3)",
            "cobertura_declarada": ("25 das 27 UFs — o Distrito Federal não tem rede municipal "
                                    "e Roraima está ausente do arquivo do INEP"),
        },
        "alvo": {
            "definicao": f"em_risco = taxa_alfabetizacao < {LIMIAR_RISCO:.0f}%",
            "justificativa_do_corte": (
                "corte absoluto e interpretável sem contexto estatístico. Descartadas: a meta "
                "pactuada (deriva da taxa de 2023 do próprio município, logo circular) e a média "
                "nacional (relativa — metade do país estaria em risco por construção)"
            ),
            "municipios_em_risco": int(y.sum()),
            "prevalencia_pct": round(float(y.mean() * 100), 2),
            "acuracia_do_baseline_pct": round(float(max(y.mean(), 1 - y.mean()) * 100), 2),
        },
        "features": {
            "colunas_de_entrada": len(F.colunas_do_modelo()),
            "features_apos_preprocessamento": len(nomes_das_features(modelo_publicado)),
            "historico": F.HISTORICO,
            "atualidade": {
                "formacao_docente_afd": F.AFD,
                "esforco_docente_ied": F.IED,
                "niveis_inse": F.INSE_NIVEIS,
                "tamanho_de_turma_atu": F.ATU,
                "socioeconomico_outras": F.INSE_OUTRAS,
            },
            "categoricas": F.CATEGORICAS,
            "categorias_de_referencia_descartadas": F.REFERENCIAS,
            "excluidas_com_motivo": F.EXCLUIDAS,
            "contribuicao_por_bloco": por_bloco,
        },
        "tratamento_de_vazamento": {
            "criterio": ("a informação estaria disponível no momento da predição? Vazamento é "
                         "contemporâneo, não histórico — media_portugues de 2024 sai, a de 2023 fica"),
            "colunas_mapeadas": len(cfg.COLUNAS_VAZAMENTO),
            "aplicado_em": "src/preprocessing/gold.py — montar_base_ml()",
            "colunas": sorted(cfg.COLUNAS_VAZAMENTO),
        },
        "hiperparametros": {
            "C": busca.best_params_["modelo__C"],
            "grade_testada": grade,
            "criterio_da_busca": "average_precision (PR-AUC)",
            "pr_auc_da_melhor_configuracao": round(float(busca.best_score_), 4),
            "penalidade": "L2 (padrão do LogisticRegression)",
            "solver": "lbfgs",
            "max_iter": 2000,
            "random_state": SEMENTE,
        },
        "validacao": {
            "particao": "train_test_split estratificado 75/25",
            "selecao_e_ajuste": "StratifiedKFold de 5 folds sobre o treino",
            "municipios_treino": int(len(X_treino)),
            "municipios_teste": int(len(X_teste)),
            "observacao": ("o ciclo modelado contém o choque das enchentes de 2024 no Rio Grande "
                           "do Sul; o risco previsto para o estado está superestimado"),
        },
        "metricas": {
            "no_teste": no_teste,
            "limiar_por_cobertura": {
                f"recall_{int(r * 100)}pct": limiar_por_recall(y_teste, probabilidade, r)
                for r in (0.6, 0.7, 0.8, 0.9)
            },
        },
        "comparacao_de_algoritmos": algoritmos,
        "interpretabilidade": {"maiores_coeficientes": dict(maiores)},
        "limitacoes": [
            "falácia ecológica — descreve o município, não uma criança: duas crianças do mesmo "
            "município recebem a mesma predição",
            "sem variáveis de política educacional — o maior efeito do modelo (a UF) fica sem "
            "explicação dentro da base",
            "um único ciclo modelado, e ele contém o choque do Rio Grande do Sul",
            "INSE de 2023 replicado para 2024 como atributo estrutural",
            "AFD/ATU/IED usam o agregado Total, que inclui a rede privada",
            "associações condicionais, não efeitos causais",
        ],
        "reprodutibilidade": {
            "semente": SEMENTE,
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "comandos": ["python -m src.preprocessing.run_pipeline",
                         "python -m src.modeling.model_card"],
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
    print(f"  melhor PR-AUC CV : {resultado['comparacao_de_algoritmos'][0]['modelo']}")
