"""
Métricas de classificação binária.

A classe positiva é **"em risco"**, e ela é a minoritária. Isso define o que
reportamos, e por quê:

* **Recall da classe de risco** — a prioritária. O erro caro é deixar de fora um
  município que precisa de apoio; um falso alarme custa uma visita técnica.
* **PR-AUC** — mais informativa que a ROC quando a classe de interesse é
  minoritária: não se deixa inflar pelos verdadeiros negativos, que aqui são
  maioria.
* **ROC-AUC** — capacidade de ordenar municípios por risco, comparável entre
  modelos.
* **Acurácia** — reportada **sempre ao lado do baseline**, justamente para
  deixar claro que sozinha ela não diz nada.

As funções são uma camada fina sobre o Scikit-learn: padronizam *quais* métricas
o projeto reporta, sem reimplementar nenhuma.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

NOMES_CLASSES = ["fora de risco", "em risco"]


def avaliar(y_verdadeiro, y_previsto, probabilidade=None) -> dict:
    """Conjunto de métricas do problema, com a classe de risco em destaque."""
    metricas = {
        "recall_risco": recall_score(y_verdadeiro, y_previsto, zero_division=0),
        "precisao_risco": precision_score(y_verdadeiro, y_previsto, zero_division=0),
        "f1_risco": f1_score(y_verdadeiro, y_previsto, zero_division=0),
        "acuracia": accuracy_score(y_verdadeiro, y_previsto),
    }
    if probabilidade is not None:
        metricas["roc_auc"] = roc_auc_score(y_verdadeiro, probabilidade)
        metricas["pr_auc"] = average_precision_score(y_verdadeiro, probabilidade)

    metricas["acuracia_baseline"] = max(np.mean(y_verdadeiro), 1 - np.mean(y_verdadeiro))
    return {k: round(float(v), 4) for k, v in metricas.items()}


def matriz_confusao(y_verdadeiro, y_previsto) -> pd.DataFrame:
    """Matriz de confusão rotulada — as células contam municípios."""
    return pd.DataFrame(
        confusion_matrix(y_verdadeiro, y_previsto),
        index=[f"real: {c}" for c in NOMES_CLASSES],
        columns=[f"previsto: {c}" for c in NOMES_CLASSES],
    )


def curva_calibracao(y_verdadeiro, probabilidade, n_faixas: int = 10) -> pd.DataFrame:
    """Probabilidade prevista contra frequência observada, por faixa.

    Um modelo calibrado tem `previsto ≈ observado` em todas as faixas — o que
    permite ler a saída como "chance de o município estar em risco", e não
    apenas como uma ordenação.
    """
    tabela = pd.DataFrame({"y": y_verdadeiro, "prob": probabilidade})
    tabela["faixa"] = pd.qcut(tabela["prob"], n_faixas, labels=False, duplicates="drop")

    return (tabela.groupby("faixa")
            .agg(municipios=("y", "size"),
                 previsto=("prob", "mean"),
                 observado=("y", "mean"))
            .round(3).reset_index())


def limiar_por_recall(y_verdadeiro, probabilidade, recall_desejado: float = 0.80) -> dict:
    """Menor limiar que atinge o recall pedido na classe de risco.

    O limiar de 0,5 é convenção do software, não decisão de política. Quem
    prioriza município define primeiro **quanta cobertura quer** — "não quero
    perder mais de 20% dos municípios em risco" — e o limiar sai daí.
    """
    ordem = np.argsort(probabilidade)[::-1]
    y_ordenado = np.asarray(y_verdadeiro)[ordem]
    prob_ordenada = np.asarray(probabilidade)[ordem]

    recall = y_ordenado.cumsum() / max(y_ordenado.sum(), 1)
    limiar = float(prob_ordenada[np.argmax(recall >= recall_desejado)])

    previsto = (np.asarray(probabilidade) >= limiar).astype(int)
    return {
        "recall_desejado": recall_desejado,
        "limiar": round(limiar, 4),
        "municipios_sinalizados": int(previsto.sum()),
        **avaliar(y_verdadeiro, previsto, probabilidade),
    }
