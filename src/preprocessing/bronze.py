"""Camada Bronze: ingestão fiel dos cinco CSVs da Fase 2."""

import logging
from datetime import datetime, timezone

import pandas as pd

from . import config as cfg
from .lake import escrever_particionado, hash_registro

log = logging.getLogger(__name__)


def _validar(df: pd.DataFrame, entidade: str) -> None:
    if df["ano"].isna().any():
        raise ValueError(f"{entidade}: ano nulo")
    chave = "id_municipio" if "id_municipio" in df else "sigla_uf" if "sigla_uf" in df else None
    if chave and df[chave].isna().any():
        raise ValueError(f"{entidade}: {chave} nulo")
    if "taxa_alfabetizacao" in df:
        valores = df["taxa_alfabetizacao"].dropna()
        if not valores.between(0, 100).all():
            raise ValueError(f"{entidade}: taxa_alfabetizacao fora de [0, 100]")


def run_bronze() -> pd.DataFrame:
    """Lê, tipa, adiciona linhagem e grava partições ``ano=YYYY``."""
    ts = datetime.now(timezone.utc).isoformat()
    linhas = []
    for entidade, arquivo in cfg.ARQUIVOS_INEP.items():
        df = pd.read_csv(cfg.DADOS_DIR / arquivo, dtype=cfg.DTYPES[entidade], encoding="utf-8")
        _validar(df, entidade)
        df["_record_hash"] = hash_registro(df, cfg.CHAVES_HASH[entidade])
        df["_ingestion_timestamp"] = ts
        df["_source_entity"] = entidade
        df["_source_file"] = arquivo
        escrever_particionado(df, cfg.LAKE_DIR / "bronze" / entidade)
        linhas.append({"entidade": entidade, "registros": len(df), "colunas": df.shape[1]})
        log.info("[BRONZE] %s: %s registros", entidade, len(df))
    return pd.DataFrame(linhas)


run = run_bronze


if __name__ == "__main__":
    print(run_bronze().to_string(index=False))
