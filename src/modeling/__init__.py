"""Modelagem supervisionada da proxy municipal de alfabetização."""

from .alvo import ID_COLS, TARGET, carregar_dataset, resumir_alvo, separar_xy
from .features import (
    CATEGORICAS, EXCLUIDAS_POR_VAZAMENTO, FEATURES, NUMERICAS,
    construir_pipeline, construir_preprocessamento, modelos_candidatos,
    nomes_das_features,
)

__all__ = [
    "carregar_dataset", "separar_xy", "resumir_alvo", "TARGET", "ID_COLS",
    "construir_pipeline", "construir_preprocessamento", "modelos_candidatos",
    "nomes_das_features", "FEATURES", "NUMERICAS", "CATEGORICAS",
    "EXCLUIDAS_POR_VAZAMENTO",
]
