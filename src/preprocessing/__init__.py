"""Pipeline medalhão local e didática, implementada com pandas."""

from .bronze import run_bronze
from .gold import run_gold
from .lake import comparar_snapshots, ler_particionado, snapshot_camada
from .silver import run_silver

__all__ = [
    "run_bronze", "run_silver", "run_gold",
    "snapshot_camada", "comparar_snapshots", "ler_particionado",
]
