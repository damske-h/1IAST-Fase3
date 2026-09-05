"""
Pipeline medalhão local (Bronze → Silver → Gold), sem AWS.

Reproduz em pandas a arquitetura de dados construída na Fase 2 em AWS Glue +
S3 + Athena, e a estende com as fontes externas de enriquecimento.

Uso típico:

    from src.preprocessing import run_bronze, run_silver, run_gold
    run_bronze(); run_silver(); run_gold()
"""

from .bronze import run_bronze
from .gold import carregar_base_ml, montar_base_ml, run_gold
from .lake import comparar_snapshots, ler_particionado, snapshot_camada
from .silver import run_silver

__all__ = [
    "run_bronze", "run_silver", "run_gold",
    "montar_base_ml", "carregar_base_ml",
    "snapshot_camada", "comparar_snapshots", "ler_particionado",
]
