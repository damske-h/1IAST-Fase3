"""
Construção do alvo e do painel de modelagem.

**Alvo:** `em_risco = taxa_alfabetizacao < 50%` no ciclo de 2024 — o município
onde **menos da metade das crianças** chega alfabetizada ao fim do 2º ano.

A leitura para a pergunta do Tech Challenge é direta: uma criança que estuda num
município sinalizado em risco tem chance substancialmente menor de ser
considerada alfabetizada. Como nenhuma fonte pública desce ao aluno, é o
contexto do município que responde por ela.

**Duas naturezas de informação entram como preditores:**

* **histórico** — como a rede vinha indo (taxa e nota de 2023, IDEB de 2021);
* **atualidade** — como o município é hoje (nível socioeconômico, corpo
  docente, turmas, ruralidade, porte, UF).

Ambas são anteriores ao resultado previsto, então nenhuma vaza o alvo.
"""

import numpy as np
import pandas as pd

from ..preprocessing import carregar_base_ml, ler_particionado
from ..preprocessing.config import LAKE_DIR

ANO_ALVO = 2024
ANO_HISTORICO = 2023
LIMIAR_RISCO = 50.0
COLUNA_TAXA = "taxa_alfabetizacao"


def montar_painel(ano_alvo: int = ANO_ALVO, ano_historico: int = ANO_HISTORICO) -> pd.DataFrame:
    """Uma linha por município: a atualidade do ciclo alvo + o histórico do anterior.

    O histórico vem da Silver, não da base de modelagem: `taxa_alfabetizacao` e
    `media_portugues` do ciclo **corrente** são vazamento e por isso ficam fora
    da Gold analítica — **defasadas**, deixam de ser.
    """
    atual = carregar_base_ml().drop(columns=["_gold_processed_at"])
    atual = atual[atual["ano"] == ano_alvo]

    indicador = ler_particionado(LAKE_DIR / "silver" / "pass" / "indicador_municipio")
    historico = (
        indicador[(indicador["rede_desc"] == "municipal")
                  & (indicador["ano"] == ano_historico)]
        [["id_municipio", "taxa_alfabetizacao", "media_portugues"]]
        .rename(columns={"taxa_alfabetizacao": f"taxa_{ano_historico}",
                         "media_portugues": f"media_portugues_{ano_historico}"})
    )

    painel = atual.merge(historico, on="id_municipio", how="inner")
    if painel["id_municipio"].duplicated().any():
        raise ValueError("o painel deveria ter uma linha por município")
    return painel.reset_index(drop=True)


def rotular_risco(painel: pd.DataFrame, limiar: float = LIMIAR_RISCO):
    """Aplica o corte e devolve `(X, y)`.

    O corte em 50% é **absoluto e interpretável sem contexto estatístico** — um
    gestor entende na hora. Duas alternativas foram descartadas:

    * a **meta pactuada**, calculada a partir da taxa de 2023 do próprio
      município: usá-la tornaria o alvo função da própria história e a meta,
      um preditor circular;
    * a **média nacional do ano**, que é relativa — metade do país estaria em
      risco por construção, melhorasse ou piorasse.
    """
    y = (painel[COLUNA_TAXA] < limiar).astype(int).to_numpy()
    return painel, y


def resumir_rotulo(painel: pd.DataFrame, y: np.ndarray,
                   limiar: float = LIMIAR_RISCO) -> pd.DataFrame:
    """Conferência do balanceamento — a classe de risco é a minoritária."""
    return pd.DataFrame([
        {"medida": "municípios", "valor": len(painel)},
        {"medida": f"em risco (taxa < {limiar:.0f}%)", "valor": int(y.sum())},
        {"medida": "fora de risco", "valor": int((y == 0).sum())},
        {"medida": "prevalência do risco (%)", "valor": round(float(y.mean() * 100), 2)},
        {"medida": "acurácia do baseline (%)",
         "valor": round(float(max(y.mean(), 1 - y.mean()) * 100), 2)},
    ])
