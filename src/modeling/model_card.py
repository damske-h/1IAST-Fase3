"""Treina o modelo final e gera artefatos reproduzíveis fora do notebook.

Todo número publicado no README, no relatório técnico e na apresentação sai
daqui ou dos notebooks — nenhum é digitado à mão.

O ponto delicado deste módulo é a **separação entre escolher e avaliar**:

* a comparação por validação cruzada acontece só em 2023 e é o que *poderia*
  ter sido usado para escolher o algoritmo;
* a comparação no holdout de 2024 é o que *foi* usado no trabalho de referência;
* as duas são gravadas lado a lado porque discordam, e a discordância é um dos
  achados do projeto (ver `generalizacao_temporal`).
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV

from ..evaluation.interpretabilidade import (
    importancia_nativa, importancia_por_permutacao, importancia_shap,
)
from ..evaluation.metricas import (
    avaliar, comparar_por_validacao_cruzada, desvio_de_calibracao,
    limiar_por_recall, particionador,
)
from ..preprocessing.config import RAIZ
from .alvo import TARGET, carregar_dataset
from .features import FEATURES, construir_pipeline, modelos_candidatos

REPORTS_DIR = RAIZ / "reports"
GRADE_C = [0.01, 0.1, 1.0, 10.0]


def _avaliar_no_holdout(nome, modelo, X, y) -> dict:
    probabilidade = modelo.predict_proba(X)[:, 1]
    return {"modelo": nome, **avaliar(y, modelo.predict(X), probabilidade)}


def gerar(destino: Path = REPORTS_DIR / "model_card.json") -> dict:
    df = carregar_dataset()
    treino, teste = df[df["ano"] == 2023], df[df["ano"] == 2024]
    X_treino, y_treino = treino[FEATURES], treino[TARGET]
    X_teste, y_teste = teste[FEATURES], teste[TARGET]

    candidatos = modelos_candidatos(com_baseline=True)

    # 1. Escolha honesta: só 2023.
    cv_2023 = comparar_por_validacao_cruzada(
        candidatos, construir_pipeline, X_treino, y_treino
    )

    # 2. Desempenho real no ciclo seguinte.
    holdout = pd.DataFrame([
        _avaliar_no_holdout(nome, construir_pipeline(est).fit(X_treino, y_treino),
                            X_teste, y_teste)
        for nome, est in candidatos.items()
    ]).sort_values("roc_auc", ascending=False)

    # 3. A discordância entre as duas: quanto cada modelo perde fora do ano de treino.
    generalizacao = (
        cv_2023[["modelo", "roc_auc_val"]]
        .merge(holdout[["modelo", "roc_auc"]], on="modelo")
        .rename(columns={"roc_auc_val": "roc_auc_cv_2023",
                         "roc_auc": "roc_auc_holdout_2024"})
        .assign(queda=lambda d: d["roc_auc_cv_2023"] - d["roc_auc_holdout_2024"])
        .sort_values("roc_auc_cv_2023", ascending=False)
    )

    # 4. Otimização do modelo escolhido, com score de treino ao lado do de validação.
    busca = GridSearchCV(
        construir_pipeline(), {"modelo__C": GRADE_C},
        scoring="f1", cv=particionador(), n_jobs=-1,
        refit=True, return_train_score=True,
    ).fit(X_treino, y_treino)
    modelo_final = busca.best_estimator_

    grade = pd.DataFrame(busca.cv_results_)[[
        "param_modelo__C", "mean_train_score", "mean_test_score", "std_test_score",
    ]].assign(gap=lambda d: d["mean_train_score"] - d["mean_test_score"])

    probabilidade = modelo_final.predict_proba(X_teste)[:, 1]
    metricas = avaliar(y_teste, modelo_final.predict(X_teste), probabilidade)

    # 5. Interpretabilidade: as três lentes pedidas pelo enunciado.
    coeficientes = importancia_nativa(modelo_final)
    permutacao = importancia_por_permutacao(modelo_final, X_teste, y_teste)
    shap_importancia = importancia_shap(modelo_final, X_treino, X_teste)

    REPORTS_DIR.mkdir(exist_ok=True)
    joblib.dump(modelo_final, REPORTS_DIR / "modelo_final.joblib")
    holdout.round(6).to_csv(REPORTS_DIR / "comparacao_modelos.csv", index=False)
    cv_2023.round(6).to_csv(REPORTS_DIR / "comparacao_modelos_cv2023.csv", index=False)

    card = {
        "unidade_analise": "municipio-ano-rede municipal",
        "alvo": "proxy: atingiu a meta municipal de 2025",
        "classe_positiva": "atingiu a meta (risco = 1 - probabilidade)",
        "features": FEATURES,
        "semente": 42,
        "treino": {"ano": 2023, "n": int(len(treino)),
                   "prevalencia": round(float(y_treino.mean()), 4)},
        "teste": {"ano": 2024, "n": int(len(teste)),
                  "prevalencia": round(float(y_teste.mean()), 4)},
        "melhores_parametros": busca.best_params_,
        "otimizacao": {
            "metrica": "f1",
            "grade_C": GRADE_C,
            "resultados": grade.round(4).to_dict(orient="records"),
        },
        "selecao_por_cv_2023": cv_2023.round(4).to_dict(orient="records"),
        "comparacao_no_holdout_2024": holdout.round(4).to_dict(orient="records"),
        "generalizacao_temporal": generalizacao.round(4).to_dict(orient="records"),
        "metricas_holdout": metricas,
        "desvio_de_calibracao": desvio_de_calibracao(y_teste, probabilidade),
        "limiares_de_cobertura": [
            limiar_por_recall(y_teste, probabilidade, r) for r in (0.6, 0.7, 0.8, 0.9)
        ],
        "interpretabilidade": {
            "coeficientes_top10": coeficientes.head(10).round(4).to_dict(orient="records"),
            "permutation_importance": permutacao.round(4).to_dict(orient="records"),
            "shap_top10": shap_importancia.head(10).round(4).to_dict(orient="records"),
        },
        "limitacao": (
            "indicadores contemporaneos; classificacao/nowcast municipal, "
            "nao previsao individual ou causal"
        ),
    }
    destino.write_text(
        json.dumps(card, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return card


if __name__ == "__main__":
    resumo = gerar()
    print(json.dumps({
        "melhores_parametros": resumo["melhores_parametros"],
        "metricas_holdout": resumo["metricas_holdout"],
        "generalizacao_temporal": resumo["generalizacao_temporal"],
    }, ensure_ascii=False, indent=2))
