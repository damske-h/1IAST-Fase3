"""Camada Silver: deduplicação, padronização e qualidade por linha."""

import logging
from datetime import datetime, timezone

import pandas as pd

from . import config as cfg
from .lake import escrever_particionado, ler_particionado

log = logging.getLogger(__name__)


def _tratar(df: pd.DataFrame, entidade: str) -> pd.DataFrame:
    df = df.sort_values("_ingestion_timestamp").drop_duplicates("_record_hash", keep="last").copy()
    if "rede" in df:
        df["rede_desc"] = df["rede"].map(cfg.REDE_MAP).fillna(df["rede"])
    for coluna in [c for c in df if c == "taxa_alfabetizacao" or c.startswith("meta_alfabetizacao_")]:
        df[coluna] = df[coluna].round(2)
    if "media_portugues" in df:
        df["media_portugues"] = df["media_portugues"].round(2)

    valido = df["ano"].between(2020, 2030)
    if "id_municipio" in df:
        valido &= df["id_municipio"].str.fullmatch(r"\d{7}", na=False)
    if "sigla_uf" in df:
        valido &= df["sigla_uf"].str.fullmatch(r"[A-Z]{2}", na=False)
    if entidade == "indicador_municipio":
        valido &= df["taxa_alfabetizacao"].between(0, 100) & df["taxa_alfabetizacao"].notna()
    df["_dq_passou"] = valido
    df["_silver_processed_at"] = datetime.now(timezone.utc).isoformat()
    return df


def run_silver() -> pd.DataFrame:
    linhas = []
    for entidade in cfg.ARQUIVOS_INEP:
        df = _tratar(ler_particionado(cfg.LAKE_DIR / "bronze" / entidade), entidade)
        passou, quarentena = df[df["_dq_passou"]].copy(), df[~df["_dq_passou"]].copy()
        escrever_particionado(passou, cfg.LAKE_DIR / "silver" / "pass" / entidade)
        if not quarentena.empty:
            quarentena["_quarentena_motivo"] = "chave, ano ou taxa inválida"
            escrever_particionado(quarentena, cfg.LAKE_DIR / "silver" / "quarentena" / entidade)
        linhas.append({"entidade": entidade, "pass": len(passou), "quarentena": len(quarentena)})
        log.info("[SILVER] %s: pass=%s, quarentena=%s", entidade, len(passou), len(quarentena))
    return pd.DataFrame(linhas)


run = run_silver


if __name__ == "__main__":
    print(run_silver().to_string(index=False))
