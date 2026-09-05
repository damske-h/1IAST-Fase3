"""
Executor da pipeline medalhão local.

Roda as três camadas em sequência e imprime o sumário de cada uma — o mesmo
papel dos SUMÁRIOS que os jobs Glue da Fase 2 gravavam no CloudWatch.

    python -m src.preprocessing.run_pipeline
"""

import logging
import sys

from .bronze import run_bronze
from .gold import run_gold
from .silver import run_silver


def configurar_log(nivel=logging.INFO) -> None:
    logging.basicConfig(
        level=nivel,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


def main() -> None:
    configurar_log()

    for titulo, camada in (("BRONZE", run_bronze),
                           ("SILVER", run_silver),
                           ("GOLD", run_gold)):
        print("\n" + "=" * 78)
        print(f"SUMÁRIO {titulo}")
        print("=" * 78)
        print(camada().to_string(index=False))


if __name__ == "__main__":
    main()
