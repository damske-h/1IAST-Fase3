"""
Definição do alvo — classificação binária no grão do município.

**A premissa desta abordagem.** A alfabetização de um aluno é tratada como
*dada pelo contexto do município em que ele estuda*. Em vez de tentar modelar a
criança — que nenhuma base pública descreve individualmente —, modelamos o
município, e a leitura para o aluno é direta: *"uma criança deste município
está num contexto onde menos da metade dos colegas chega alfabetizada"*.

Isso troca uma construção estatística elaborada (dado binário agrupado com
pesos amostrais) por **uma linha, um município, um rótulo** — o formato que
todo classificador do Scikit-learn espera, e o que os hands-on da disciplina
usam do início ao fim.

**O rótulo.** `em_risco = taxa_alfabetizacao < 50%` — o município onde **menos
da metade das crianças** chega alfabetizada ao fim do 2º ano.

Três razões para este corte:

1. **É absoluto.** Não deriva da taxa passada do próprio município (o que seria
   circular) nem da média dos demais (o que faria metade do país estar sempre
   "em risco" por construção, melhorasse o país ou não).
2. **A classe positiva é a minoritária e a acionável** — 27,3% dos municípios.
   É o caso em que *recall* e `class_weight` fazem diferença, e o erro que
   custa caro é deixar um município em risco passar despercebido.
3. **É comunicável.** "Menos da metade das crianças" é uma frase que um gestor
   entende sem nota de rodapé.

**Recorte temporal.** Apenas o ciclo de **2024**: uma linha por município, sem
repetição. Não há grupo a proteger na validação cruzada, o que dispensa o
`GroupKFold` e permite usar `train_test_split` estratificado e
`StratifiedKFold` diretamente.
"""

import pandas as pd

LIMIAR_RISCO = 50.0
ANO_MODELAGEM = 2024
COLUNA_TAXA = "taxa_alfabetizacao"


def rotular_risco(base: pd.DataFrame,
                  ano: int = ANO_MODELAGEM,
                  limiar: float = LIMIAR_RISCO):
    """Recorta o ciclo e devolve `(X, y)`.

    `X` é a base do ciclo, uma linha por município. `y` vale 1 quando o
    município está **em risco** — menos de `limiar`% das crianças alfabetizadas.
    """
    corte = base[base["ano"] == ano].reset_index(drop=True)
    if corte.empty:
        raise ValueError(f"nenhuma linha para o ciclo {ano}")
    if corte["id_municipio"].duplicated().any():
        raise ValueError("há municípios repetidos no ciclo — o grão deveria ser único")

    y = (corte[COLUNA_TAXA] < limiar).astype(int).to_numpy()
    return corte, y


def resumir_rotulo(X: pd.DataFrame, y, limiar: float = LIMIAR_RISCO) -> pd.DataFrame:
    """Sumário de conferência do rótulo, para exibir no notebook."""
    em_risco = X.loc[y == 1, COLUNA_TAXA]
    fora = X.loc[y == 0, COLUNA_TAXA]

    return pd.DataFrame([
        {"classe": f"em risco (taxa < {limiar:.0f}%)",
         "municípios": int(y.sum()),
         "% do total": round(float(y.mean()) * 100, 1),
         "taxa média": round(float(em_risco.mean()), 1),
         "taxa mínima": round(float(em_risco.min()), 1),
         "taxa máxima": round(float(em_risco.max()), 1)},
        {"classe": f"fora de risco (taxa >= {limiar:.0f}%)",
         "municípios": int((y == 0).sum()),
         "% do total": round(float((y == 0).mean()) * 100, 1),
         "taxa média": round(float(fora.mean()), 1),
         "taxa mínima": round(float(fora.min()), 1),
         "taxa máxima": round(float(fora.max()), 1)},
    ])
