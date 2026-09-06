"""Avaliação: métricas ponderadas, validação cruzada e calibração."""

from .estrategia import (
    avaliar_k, efeitos_marginais, estabilidade_ranking, preparar_segmentacao,
    segmentar, tabela_municipal,
)
from .metricas import (
    avaliar, buscar_em_grade, curva_calibracao, ks_ponderado,
    matriz_confusao, validar_cruzado,
)

__all__ = [
    "avaliar", "ks_ponderado", "validar_cruzado", "buscar_em_grade",
    "curva_calibracao", "matriz_confusao",
    "tabela_municipal", "efeitos_marginais", "preparar_segmentacao",
    "avaliar_k", "segmentar", "estabilidade_ranking",
]
