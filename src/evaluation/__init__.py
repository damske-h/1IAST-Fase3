"""Avaliação: métricas ponderadas, validação cruzada e calibração."""

from .metricas import (
    avaliar, buscar_em_grade, curva_calibracao, ks_ponderado,
    matriz_confusao, validar_cruzado,
)

__all__ = [
    "avaliar", "ks_ponderado", "validar_cruzado", "buscar_em_grade",
    "curva_calibracao", "matriz_confusao",
]
