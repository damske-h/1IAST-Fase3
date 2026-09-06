"""
Tradução do modelo em instrumentos de decisão.

Enquanto `metricas.py` responde "o modelo é bom?", este módulo responde "o que
se faz com ele?". São quatro instrumentos:

* **efeitos marginais** — o efeito de cada variável em *pontos percentuais* da
  taxa esperada, que é a unidade em que um gestor pensa (o *odds ratio* é
  correto, mas não é acionável numa reunião);
* **taxa esperada e resíduo** — quanto o município alfabetiza, comparado ao que
  a estrutura dele levaria a esperar;
* **segmentação** — agrupamento de municípios por perfil, para desenhar
  intervenção por tipo em vez de por território;
* **estabilidade do ranking** — quanto uma lista de prioridade muda de um ciclo
  para outro. É o instrumento que impede o uso indevido dos demais.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEMENTE = 42


# =============================================================================
# TAXA ESPERADA E RESÍDUO
# =============================================================================

def tabela_municipal(modelo, base: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta `taxa_esperada` e `residuo` à base agregada.

    Como as duas linhas da expansão binomial compartilham o mesmo vetor de
    características, prever direto sobre a base agregada devolve exatamente a
    taxa esperada daquele município — em percentual, para leitura direta.

    O **resíduo** (observado − esperado) é a quantidade interessante para
    política pública: mede quanto o município entrega *além* — ou *aquém* — do
    que a estrutura dele explicaria.
    """
    tabela = base.copy()
    tabela["taxa_esperada"] = modelo.predict_proba(base)[:, 1] * 100
    tabela["residuo"] = tabela["taxa_alfabetizacao"] - tabela["taxa_esperada"]
    return tabela


def efeitos_marginais(modelo, base: pd.DataFrame, colunas) -> pd.DataFrame:
    """Efeito de somar um desvio-padrão a cada variável, em pontos percentuais.

    Diferente do coeficiente, que vive em log-odds, este número responde à
    pergunta do gestor: *"se eu melhorar isto, quanto sobe a taxa esperada?"*.
    Continua sendo uma leitura **associativa**, não causal.
    """
    referencia = modelo.predict_proba(base)[:, 1]

    linhas = []
    for coluna in colunas:
        alterada = base.copy()
        alterada[coluna] = alterada[coluna] + alterada[coluna].std()
        efeito = (modelo.predict_proba(alterada)[:, 1] - referencia).mean() * 100
        linhas.append({"variavel": coluna, "efeito_pp": efeito})

    return (pd.DataFrame(linhas)
            .set_index("variavel")
            .sort_values("efeito_pp", key=np.abs, ascending=False))


# =============================================================================
# SEGMENTAÇÃO
# =============================================================================

def preparar_segmentacao(base: pd.DataFrame, colunas):
    """Imputa e padroniza o perfil municipal para o agrupamento."""
    preparo = Pipeline([
        ("imputacao", SimpleImputer(strategy="median")),
        ("padronizacao", StandardScaler()),
    ])
    return preparo.fit_transform(base[colunas])


def avaliar_k(matriz, valores_de_k=range(2, 9), amostra: int = 3000) -> pd.DataFrame:
    """Inércia e silhueta por número de grupos.

    A silhueta é calculada numa amostra porque é O(n²). Valores baixos (abaixo
    de ~0,2) indicam que **não há grupos naturais** — os perfis formam um
    contínuo, e qualquer partição é uma conveniência descritiva, não uma
    estrutura descoberta nos dados.
    """
    indices = np.random.default_rng(SEMENTE).choice(
        len(matriz), min(amostra, len(matriz)), replace=False)

    linhas = []
    for k in valores_de_k:
        agrupador = KMeans(n_clusters=k, n_init=10, random_state=SEMENTE).fit(matriz)
        linhas.append({
            "k": k,
            "inercia": agrupador.inertia_,
            "silhueta": silhouette_score(matriz[indices], agrupador.labels_[indices]),
            "menor_grupo": int(np.bincount(agrupador.labels_).min()),
        })
    return pd.DataFrame(linhas)


def segmentar(matriz, k: int) -> np.ndarray:
    """Rótulo de segmento por município."""
    return KMeans(n_clusters=k, n_init=10, random_state=SEMENTE).fit_predict(matriz)


# =============================================================================
# ESTABILIDADE DE RANKING
# =============================================================================

def estabilidade_ranking(tabela: pd.DataFrame, coluna: str,
                         tamanhos=(300, 500, 1000),
                         chave: str = "id_municipio",
                         periodo: str = "ano") -> pd.DataFrame:
    """Quanto uma lista dos "N piores" muda de um ciclo para o outro.

    Uma lista de prioridade só é utilizável se for razoavelmente estável: se os
    N piores de um ciclo forem outros no ciclo seguinte, o critério está
    medindo ruído, não o problema. A sobreposição entre os dois ciclos é a
    forma mais direta de verificar isso antes de publicar a lista.
    """
    largo = tabela.pivot_table(index=chave, columns=periodo, values=coluna).dropna()
    if largo.shape[1] != 2:
        raise ValueError("são necessários exatamente dois ciclos para comparar")

    primeiro, segundo = largo.columns
    linhas = []
    for n in tamanhos:
        a = set(largo[primeiro].nsmallest(n).index)
        b = set(largo[segundo].nsmallest(n).index)
        linhas.append({
            "tamanho_da_lista": n,
            "municípios em comum": len(a & b),
            "sobreposição_%": round(len(a & b) / n * 100, 1),
        })

    resultado = pd.DataFrame(linhas)
    resultado.attrs["correlacao_entre_ciclos"] = round(
        largo[primeiro].corr(largo[segundo]), 3)
    return resultado
