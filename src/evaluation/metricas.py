"""
Métricas, validação cruzada e busca de hiperparâmetros com pesos amostrais.

**Por que a validação é escrita à mão em vez de usar `GridSearchCV` direto.**

O alvo deste projeto é binomial ponderado: cada município vira duas
observações com o *mesmo* vetor de características, uma com `y=1` e outra com
`y=0`. Toda a informação está nos **pesos**. Um scorer que ignore
`sample_weight` enxerga, para cada município, um par idêntico de exemplos com
rótulos opostos — e devolve AUC exatamente 0,5, independentemente do modelo.

Fazer o `sample_weight` chegar ao scorer dentro do `GridSearchCV` exigiria
ativar o *metadata routing* do Scikit-learn e declarar o roteamento em **cada**
etapa aninhada do `ColumnTransformer` (inclusive nas que apenas ignoram o
peso). O laço explícito abaixo é mais curto, mais legível e não deixa dúvida
sobre onde o peso entra — no ajuste e na avaliação.

A interface espelha a do `GridSearchCV` ensinado nas aulas: recebe uma grade,
um `cv` e devolve resultados por combinação, com `score` de treino e de
validação lado a lado — que é como se diagnostica overfitting.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

RANDOM_STATE = 42


# =============================================================================
# MÉTRICAS PONDERADAS
# =============================================================================

def ks_ponderado(y, probabilidade, peso) -> float:
    """Estatística KS: máxima distância entre as distribuições acumuladas das
    probabilidades previstas para y=1 e para y=0.

    Mede o poder de separação do modelo — quanto do espaço de probabilidade
    separa alfabetizados de não alfabetizados.
    """
    ordem = np.argsort(probabilidade)
    y, peso = np.asarray(y)[ordem], np.asarray(peso)[ordem]

    positivos = np.cumsum(peso * (y == 1))
    negativos = np.cumsum(peso * (y == 0))
    total_positivos, total_negativos = positivos[-1], negativos[-1]
    if total_positivos == 0 or total_negativos == 0:
        return float("nan")

    return float(np.max(np.abs(positivos / total_positivos - negativos / total_negativos)))


def avaliar(y, probabilidade, peso) -> dict:
    """Conjunto de métricas do problema, todas ponderadas.

    * **AUC-ROC** — capacidade de *ordenar* municípios por risco, que é o uso
      pretendido (priorização de política pública).
    * **Brier score** — erro quadrático da probabilidade; mede *calibração*,
      porque o número previsto será lido como "taxa esperada". Menor é melhor.
    * **Log loss** — a própria função otimizada; útil para comparar ajustes.
    * **KS** — separação entre as duas distribuições.
    """
    y = np.asarray(y)
    probabilidade = np.asarray(probabilidade)
    peso = np.asarray(peso)

    return {
        "auc": roc_auc_score(y, probabilidade, sample_weight=peso),
        "brier": brier_score_loss(y, probabilidade, sample_weight=peso),
        "log_loss": log_loss(y, probabilidade, sample_weight=peso),
        "ks": ks_ponderado(y, probabilidade, peso),
    }


# =============================================================================
# VALIDAÇÃO CRUZADA E BUSCA EM GRADE
# =============================================================================

def validar_cruzado(pipeline, X, y, peso, grupos, cv, metrica="auc") -> pd.DataFrame:
    """Validação cruzada com pesos, devolvendo uma linha por fold.

    O pré-processamento vive dentro do `pipeline`, então imputação,
    padronização e codificação são ajustadas **só no fold de treino** —
    é isso que impede o vazamento de estatística do conjunto de validação.
    """
    linhas = []
    for i, (treino, validacao) in enumerate(cv.split(X, y, grupos), start=1):
        modelo = clone(pipeline)
        modelo.fit(X.iloc[treino], y[treino], modelo__sample_weight=peso[treino])

        prob_treino = modelo.predict_proba(X.iloc[treino])[:, 1]
        prob_validacao = modelo.predict_proba(X.iloc[validacao])[:, 1]

        metricas_treino = avaliar(y[treino], prob_treino, peso[treino])
        metricas_validacao = avaliar(y[validacao], prob_validacao, peso[validacao])

        linhas.append({
            "fold": i,
            "municipios_treino": len(np.unique(grupos[treino])),
            "municipios_validacao": len(np.unique(grupos[validacao])),
            **{f"treino_{k}": v for k, v in metricas_treino.items()},
            **{f"validacao_{k}": v for k, v in metricas_validacao.items()},
        })

    return pd.DataFrame(linhas)


def buscar_em_grade(construtor, grade, X, y, peso, grupos, cv,
                    metrica="auc") -> pd.DataFrame:
    """Busca em grade com validação cruzada ponderada.

    `construtor` é uma função que recebe os hiperparâmetros e devolve um
    `Pipeline` novo. `grade` é um dicionário nome → lista de valores.

    Devolve, por combinação, a média e o desvio da métrica na validação e a
    média no treino. A diferença entre as duas colunas é o diagnóstico de
    overfitting — o mesmo papel de `return_train_score=True` no `GridSearchCV`.
    """
    from itertools import product

    nomes = list(grade)
    linhas = []

    for valores in product(*(grade[n] for n in nomes)):
        parametros = dict(zip(nomes, valores))
        resultados = validar_cruzado(construtor(**parametros), X, y, peso, grupos, cv)
        linhas.append({
            **parametros,
            f"validacao_{metrica}_media": resultados[f"validacao_{metrica}"].mean(),
            f"validacao_{metrica}_desvio": resultados[f"validacao_{metrica}"].std(),
            f"treino_{metrica}_media": resultados[f"treino_{metrica}"].mean(),
            "validacao_brier_media": resultados["validacao_brier"].mean(),
        })

    resultado = pd.DataFrame(linhas)
    resultado["diferenca_treino_validacao"] = (
        resultado[f"treino_{metrica}_media"] - resultado[f"validacao_{metrica}_media"]
    )
    return resultado.sort_values(f"validacao_{metrica}_media", ascending=False)


# =============================================================================
# CALIBRAÇÃO E LIMIAR
# =============================================================================

def curva_calibracao(y, probabilidade, peso, n_faixas: int = 10) -> pd.DataFrame:
    """Compara a probabilidade prevista com a frequência observada, por faixa.

    Um modelo bem calibrado tem `previsto ~= observado` em todas as faixas.
    Importa aqui porque a saída será lida como "taxa esperada de alfabetização",
    e não apenas como uma ordenação.
    """
    faixas = pd.qcut(probabilidade, n_faixas, labels=False, duplicates="drop")
    tabela = pd.DataFrame({"faixa": faixas, "y": y, "prob": probabilidade, "peso": peso})

    def agregar(grupo):
        peso_total = grupo["peso"].sum()
        return pd.Series({
            "previsto": np.average(grupo["prob"], weights=grupo["peso"]),
            "observado": grupo.loc[grupo["y"] == 1, "peso"].sum() / peso_total,
            "peso": peso_total,
        })

    return tabela.groupby("faixa").apply(agregar, include_groups=False).reset_index()


def matriz_confusao(y, probabilidade, peso, limiar: float = 0.5) -> pd.DataFrame:
    """Matriz de confusão ponderada — as células contam *alunos*, não linhas."""
    previsto = (probabilidade >= limiar).astype(int)
    matriz = np.zeros((2, 2))
    for real in (0, 1):
        for pred in (0, 1):
            matriz[real, pred] = peso[(y == real) & (previsto == pred)].sum()

    return pd.DataFrame(
        matriz.round(1),
        index=["real: não alfabetizado", "real: alfabetizado"],
        columns=["previsto: não alfabetizado", "previsto: alfabetizado"],
    )
