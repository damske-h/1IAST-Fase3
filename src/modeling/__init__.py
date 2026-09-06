"""Modelagem supervisionada: construção do alvo e pipeline do modelo."""

from .alvo import expandir_binomial, resumir_expansao
from .features import (
    CATEGORICAS, EXCLUIDAS, NUMERICAS_GERAIS, NUMERICAS_IDEB, REFERENCIAS,
    colunas_do_modelo, construir_pipeline, nomes_das_features,
)

__all__ = [
    "expandir_binomial", "resumir_expansao",
    "construir_pipeline", "colunas_do_modelo", "nomes_das_features",
    "NUMERICAS_GERAIS", "NUMERICAS_IDEB", "CATEGORICAS", "EXCLUIDAS", "REFERENCIAS",
]
