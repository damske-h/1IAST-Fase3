"""Treina o modelo final e gera artefatos reproduzíveis fora do notebook."""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from ..preprocessing.config import RAIZ
from .alvo import TARGET, carregar_dataset
from .features import FEATURES, construir_pipeline, modelos_candidatos

REPORTS_DIR = RAIZ / "reports"


def _medir(nome, modelo, X, y):
    previsto = modelo.predict(X)
    prob = modelo.predict_proba(X)[:, 1]
    return {
        "modelo": nome,
        "acuracia": accuracy_score(y, previsto),
        "precisao": precision_score(y, previsto, zero_division=0),
        "recall": recall_score(y, previsto, zero_division=0),
        "f1": f1_score(y, previsto, zero_division=0),
        "roc_auc": roc_auc_score(y, prob),
        "pr_auc": average_precision_score(y, prob),
    }


def gerar(destino: Path = REPORTS_DIR / "model_card.json") -> dict:
    df = carregar_dataset()
    treino, teste = df[df["ano"] == 2023], df[df["ano"] == 2024]
    X_train, y_train = treino[FEATURES], treino[TARGET]
    X_test, y_test = teste[FEATURES], teste[TARGET]

    resultados = []
    for nome, estimador in modelos_candidatos().items():
        modelo = construir_pipeline(estimador).fit(X_train, y_train)
        resultados.append(_medir(nome, modelo, X_test, y_test))

    comparacao = pd.DataFrame(resultados).sort_values("roc_auc", ascending=False)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    busca = GridSearchCV(
        construir_pipeline(),
        {"modelo__C": [0.01, 0.1, 1.0, 10.0]},
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        refit=True,
    ).fit(X_train, y_train)
    modelo_final = busca.best_estimator_
    metricas = _medir("Regressão Logística otimizada", modelo_final, X_test, y_test)

    REPORTS_DIR.mkdir(exist_ok=True)
    joblib.dump(modelo_final, REPORTS_DIR / "modelo_final.joblib")
    comparacao.round(6).to_csv(REPORTS_DIR / "comparacao_modelos.csv", index=False)

    card = {
        "unidade_analise": "municipio-ano-rede municipal",
        "alvo": "proxy: atingiu a meta municipal de 2025",
        "features": FEATURES,
        "treino": {"ano": 2023, "n": int(len(treino))},
        "teste": {"ano": 2024, "n": int(len(teste))},
        "melhores_parametros": busca.best_params_,
        "metricas_holdout": {
            chave: round(float(valor), 4)
            for chave, valor in metricas.items()
            if chave != "modelo"
        },
        "limitacao": (
            "indicadores contemporaneos; classificacao/nowcast municipal, "
            "nao previsao individual ou causal"
        ),
    }
    destino.write_text(json.dumps(card, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return card


if __name__ == "__main__":
    print(json.dumps(gerar(), ensure_ascii=False, indent=2))
