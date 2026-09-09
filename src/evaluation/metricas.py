"""Métricas de classificação e instrumentos de seleção de modelo.

**Convenção de classes.** A classe positiva do estimador é `alvo_alfabetizado = 1`,
isto é, **o município atingiu a meta**. A leitura de política pública é a
inversa — `prob_risco = 1 - prob_atingir_meta` — e vive em `estrategia.py`. As
duas convivem sem ambiguidade desde que se diga qual está em uso; aqui é sempre
a do estimador.

**O que reportamos, e por quê.**

* **ROC-AUC e PR-AUC** — independentes do limiar, comparáveis entre modelos.
* **Precisão, recall e F1** — dependem do limiar (0,5 por padrão), que é
  convenção de software e não decisão de política; por isso `limiar_por_recall`
  existe.
* **Acurácia** — reportada **sempre ao lado do baseline** da classe majoritária,
  justamente para deixar claro que sozinha ela não diz nada.

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
from sklearn.model_selection import StratifiedKFold, cross_validate

NOMES_CLASSES = ["nao atingiu a meta", "atingiu a meta"]
SEMENTE = 42


def particionador(n_folds: int = 5) -> StratifiedKFold:
    """Divisão estratificada e semeada — a mesma em todo o projeto."""
    return StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEMENTE)


def avaliar(y_verdadeiro, y_previsto, probabilidade=None) -> dict:
    """Métricas do problema para a classe positiva `atingiu a meta`."""
    metricas = {
        "acuracia": accuracy_score(y_verdadeiro, y_previsto),
        "precisao": precision_score(y_verdadeiro, y_previsto, zero_division=0),
        "recall": recall_score(y_verdadeiro, y_previsto, zero_division=0),
        "f1": f1_score(y_verdadeiro, y_previsto, zero_division=0),
    }
    if probabilidade is not None:
        metricas["roc_auc"] = roc_auc_score(y_verdadeiro, probabilidade)
        metricas["pr_auc"] = average_precision_score(y_verdadeiro, probabilidade)

    media = np.mean(y_verdadeiro)
    metricas["acuracia_baseline"] = max(media, 1 - media)
    return {chave: round(float(valor), 4) for chave, valor in metricas.items()}


def matriz_confusao(y_verdadeiro, y_previsto) -> pd.DataFrame:
    """Matriz de confusão rotulada — as células contam municípios."""
    return pd.DataFrame(
        confusion_matrix(y_verdadeiro, y_previsto),
        index=[f"real: {c}" for c in NOMES_CLASSES],
        columns=[f"previsto: {c}" for c in NOMES_CLASSES],
    )


def comparar_por_validacao_cruzada(candidatos: dict, construtor, X, y,
                                   n_folds: int = 5) -> pd.DataFrame:
    """Compara modelos **sem tocar no holdout**, com o gap de sobreajuste.

    Esta é a comparação metodologicamente correta para *escolher* o algoritmo:
    tudo acontece dentro do ano de treino. `gap_roc_auc` é a diferença entre o
    score de treino e o de validação em cada fold — quanto maior, mais o modelo
    está memorizando em vez de generalizar.
    """
    linhas = []
    for nome, estimador in candidatos.items():
        resultado = cross_validate(
            construtor(estimador), X, y,
            cv=particionador(n_folds),
            scoring=["roc_auc", "average_precision", "f1"],
            return_train_score=True, n_jobs=-1,
        )
        linhas.append({
            "modelo": nome,
            "roc_auc_val": resultado["test_roc_auc"].mean(),
            "roc_auc_desvio": resultado["test_roc_auc"].std(),
            "pr_auc_val": resultado["test_average_precision"].mean(),
            "f1_val": resultado["test_f1"].mean(),
            "roc_auc_treino": resultado["train_roc_auc"].mean(),
            "gap_roc_auc": (resultado["train_roc_auc"].mean()
                            - resultado["test_roc_auc"].mean()),
        })
    return (pd.DataFrame(linhas)
            .sort_values("roc_auc_val", ascending=False)
            .reset_index(drop=True))


def curva_calibracao(y_verdadeiro, probabilidade, n_faixas: int = 10) -> pd.DataFrame:
    """Probabilidade prevista contra frequência observada, por faixa.

    Um modelo calibrado tem `previsto ≈ observado` em todas as faixas — o que
    permite ler a saída como "chance de o município atingir a meta", e não
    apenas como uma ordenação.
    """
    tabela = pd.DataFrame({"y": np.asarray(y_verdadeiro), "prob": np.asarray(probabilidade)})
    tabela["faixa"] = pd.qcut(tabela["prob"], n_faixas, labels=False, duplicates="drop")

    return (tabela.groupby("faixa")
            .agg(municipios=("y", "size"),
                 previsto=("prob", "mean"),
                 observado=("y", "mean"))
            .round(3).reset_index())


def desvio_de_calibracao(y_verdadeiro, probabilidade, n_faixas: int = 10) -> float:
    """Erro médio absoluto entre previsto e observado nas faixas de probabilidade."""
    faixas = curva_calibracao(y_verdadeiro, probabilidade, n_faixas)
    return round(float((faixas["previsto"] - faixas["observado"]).abs().mean()), 4)


def limiar_por_recall(y_verdadeiro, probabilidade, recall_desejado: float = 0.80) -> dict:
    """Menor limiar que atinge o recall pedido na classe positiva.

    O limiar de 0,5 é convenção do software, não decisão de política. Quem
    prioriza município define primeiro **quanta cobertura quer** — "não quero
    perder mais de 20% dos casos" — e o limiar sai daí.
    """
    probabilidade = np.asarray(probabilidade)
    ordem = np.argsort(probabilidade)[::-1]
    y_ordenado = np.asarray(y_verdadeiro)[ordem]

    recall = y_ordenado.cumsum() / max(y_ordenado.sum(), 1)
    limiar = float(probabilidade[ordem][np.argmax(recall >= recall_desejado)])

    previsto = (probabilidade >= limiar).astype(int)
    return {
        "recall_desejado": recall_desejado,
        "limiar": round(limiar, 4),
        "municipios_sinalizados": int(previsto.sum()),
        **avaliar(y_verdadeiro, previsto, probabilidade),
    }
