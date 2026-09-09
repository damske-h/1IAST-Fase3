"""Avaliação: métricas de classificação e instrumentos de decisão."""

from .estrategia import importancia_nativa, perfil_por_regiao, ranking_de_risco
from .metricas import (
    avaliar, curva_calibracao, limiar_por_recall, matriz_confusao,
)

__all__ = [
    "avaliar", "matriz_confusao", "curva_calibracao", "limiar_por_recall",
    "importancia_nativa", "ranking_de_risco", "perfil_por_regiao",
]
