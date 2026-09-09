"""Modelagem supervisionada: construção do alvo e pipeline do modelo."""

from .alvo import ANO_MODELAGEM, LIMIAR_RISCO, resumir_rotulo, rotular_risco
from .features import (
    CATEGORICAS, EXCLUIDAS, NUMERICAS_GERAIS, NUMERICAS_IDEB, REFERENCIAS,
    colunas_do_modelo, construir_pipeline, construir_preprocessamento,
    modelos_candidatos, nomes_das_features,
)

__all__ = [
    "rotular_risco", "resumir_rotulo", "LIMIAR_RISCO", "ANO_MODELAGEM",
    "construir_pipeline", "construir_preprocessamento", "colunas_do_modelo",
    "nomes_das_features", "modelos_candidatos",
    "NUMERICAS_GERAIS", "NUMERICAS_IDEB", "CATEGORICAS", "EXCLUIDAS", "REFERENCIAS",
]
