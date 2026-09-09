"""Avaliação: métricas de classificação e instrumentos de decisão."""

from .estrategia import (
    efeitos_marginais, perfil_por_regiao, ranking_de_risco, situacao_frente_a_meta,
)
from .metricas import (
    avaliar, curva_calibracao, limiar_por_recall, matriz_confusao,
)

__all__ = [
    "avaliar", "matriz_confusao", "curva_calibracao", "limiar_por_recall",
    "efeitos_marginais", "ranking_de_risco", "perfil_por_regiao",
    "situacao_frente_a_meta",
]
