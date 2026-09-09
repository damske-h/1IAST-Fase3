"""Interpretabilidade do modelo: Feature Importance e SHAP Values.

O enunciado pede explicitamente que o projeto identifique **quais variáveis
possuem maior influência na predição**, recomendando Feature Importance e SHAP.
Aqui as três lentes são calculadas sobre o mesmo Pipeline ajustado:

* **Coeficientes padronizados** — leitura nativa da Regressão Logística. Diz a
  direção e a magnitude relativa dentro do modelo, mas depende da escala e da
  forma funcional escolhidas.
* **Permutation importance** — embaralha uma coluna de cada vez e mede quanto o
  desempenho cai. É agnóstica ao modelo e opera sobre a **coluna original**
  (antes do one-hot), o que responde à pergunta "esta variável ajuda?" em vez de
  "esta dummy ajuda?".
* **SHAP** — decompõe cada predição individual na contribuição de cada variável.
  Como o estimador final é linear, usamos `LinearExplainer`, que calcula os
  valores exatos, sem amostragem.

Nenhuma das três estabelece causalidade: elas medem associação preditiva dentro
do modelo ajustado.
"""

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

SEMENTE = 42


def _matriz_densa(matriz):
    """One-hot devolve matriz esparsa; o SHAP linear precisa de densa."""
    return matriz.toarray() if hasattr(matriz, "toarray") else np.asarray(matriz)


def _transformar(modelo, X):
    return _matriz_densa(modelo.named_steps["preprocessamento"].transform(X))


def nomes_transformados(modelo) -> list[str]:
    """Nomes das colunas depois do one-hot, na ordem em que o estimador as vê."""
    return list(modelo.named_steps["preprocessamento"].get_feature_names_out())


def importancia_nativa(modelo) -> pd.DataFrame:
    """Coeficientes (ou ganho, nos ensembles) por variável transformada."""
    estimador = modelo.named_steps["modelo"]
    nomes = nomes_transformados(modelo)

    if hasattr(estimador, "coef_"):
        valores = estimador.coef_[0]
    elif hasattr(estimador, "feature_importances_"):
        valores = estimador.feature_importances_
    else:
        return pd.DataFrame(columns=["variavel", "coeficiente", "importancia"])

    return (
        pd.DataFrame({
            "variavel": nomes,
            "coeficiente": valores,
            "importancia": np.abs(valores),
        })
        .sort_values("importancia", ascending=False)
        .reset_index(drop=True)
    )


def importancia_por_permutacao(modelo, X, y, metrica: str = "roc_auc",
                               repeticoes: int = 10) -> pd.DataFrame:
    """Queda da métrica ao embaralhar cada **coluna de entrada** do modelo.

    Roda sobre o Pipeline inteiro, portanto o embaralhamento acontece antes do
    pré-processamento: uma UF embaralhada desmancha todas as suas dummies de uma
    vez. Valor negativo significa que a variável **atrapalha** naquele conjunto —
    o modelo vai melhor sem a informação que ela carrega.
    """
    resultado = permutation_importance(
        modelo, X, y, scoring=metrica, n_repeats=repeticoes,
        random_state=SEMENTE, n_jobs=-1,
    )
    return (
        pd.DataFrame({
            "variavel": list(X.columns),
            f"queda_{metrica}": resultado.importances_mean,
            "desvio": resultado.importances_std,
        })
        .sort_values(f"queda_{metrica}", ascending=False)
        .reset_index(drop=True)
    )


def valores_shap(modelo, X_fundo, X_explicar):
    """Valores SHAP do estimador linear, em log-odds.

    `X_fundo` define a referência (o "valor esperado"); usamos o treino, para
    que a explicação seja "quanto este município se afasta do município médio de
    2023". Devolve `(matriz_shap, matriz_transformada, nomes)`.
    """
    import shap

    fundo = _transformar(modelo, X_fundo)
    alvo = _transformar(modelo, X_explicar)
    explicador = shap.LinearExplainer(modelo.named_steps["modelo"], fundo)
    return explicador.shap_values(alvo), alvo, nomes_transformados(modelo)


def importancia_shap(modelo, X_fundo, X_explicar) -> pd.DataFrame:
    """Contribuição média absoluta de cada variável, no ranking do SHAP."""
    shap_valores, _, nomes = valores_shap(modelo, X_fundo, X_explicar)
    return (
        pd.DataFrame({
            "variavel": nomes,
            "shap_medio_abs": np.abs(shap_valores).mean(axis=0),
        })
        .sort_values("shap_medio_abs", ascending=False)
        .reset_index(drop=True)
    )
