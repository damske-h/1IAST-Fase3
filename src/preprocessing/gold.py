"""Camada Gold: quatro visões analíticas e uma base enxuta de modelagem."""

from datetime import datetime, timezone

import pandas as pd

from . import config as cfg
from .lake import escrever_particionado, ler_particionado


def _silver(entidade: str) -> pd.DataFrame:
    return ler_particionado(cfg.LAKE_DIR / "silver" / "pass" / entidade)


def _carimbar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_gold_processed_at"] = datetime.now(timezone.utc).isoformat()
    df["_ingestion_date"] = datetime.now(timezone.utc).date().isoformat()
    return df


def alfabetizacao_por_municipio(indicador: pd.DataFrame, metas: pd.DataFrame) -> pd.DataFrame:
    meta = metas[[
        "id_municipio", "ano", "meta_alfabetizacao_2025",
        "meta_alfabetizacao_2030", "nivel_alfabetizacao",
    ]].rename(columns={
        "meta_alfabetizacao_2025": "meta_2025",
        "meta_alfabetizacao_2030": "meta_2030",
    })
    df = indicador[indicador["rede"] == cfg.REDE_ALVO].merge(
        meta, on=["id_municipio", "ano"], how="left"
    )
    df = df[[
        "id_municipio", "ano", "serie", "rede_desc", "taxa_alfabetizacao",
        "media_portugues", "meta_2025", "meta_2030", "nivel_alfabetizacao",
    ]].rename(columns={"rede_desc": "rede"})
    df["gap_meta_2025"] = (df["taxa_alfabetizacao"] - df["meta_2025"]).round(2)
    df["status_meta_2025"] = pd.NA
    conhecidos = df["taxa_alfabetizacao"].notna() & df["meta_2025"].notna()
    df.loc[conhecidos, "status_meta_2025"] = "NAO_ATINGIU"
    df.loc[conhecidos & (df["taxa_alfabetizacao"] >= df["meta_2025"]), "status_meta_2025"] = "ATINGIU"
    return df


def evolucao_temporal(indicador_uf: pd.DataFrame) -> pd.DataFrame:
    return (
        indicador_uf[indicador_uf["rede"] == cfg.REDE_ALVO]
        .groupby(["sigla_uf", "ano", "serie"], as_index=False)
        .agg(
            taxa_media=("taxa_alfabetizacao", "mean"),
            taxa_min=("taxa_alfabetizacao", "min"),
            taxa_max=("taxa_alfabetizacao", "max"),
            taxa_desvio=("taxa_alfabetizacao", "std"),
            media_portugues_media=("media_portugues", "mean"),
        )
        .round(2)
    )


def ranking_municipios(indicador: pd.DataFrame) -> pd.DataFrame:
    df = indicador[indicador["rede"] == cfg.REDE_ALVO].copy()
    df["sigla_uf"] = df["id_municipio"].str[:2].map(cfg.IBGE_UF)
    df["ranking_uf"] = (
        df.groupby(["sigla_uf", "ano"])["taxa_alfabetizacao"]
        .rank(method="min", ascending=False).astype("Int64")
    )
    return df[[
        "id_municipio", "sigla_uf", "ano", "serie", "rede_desc",
        "taxa_alfabetizacao", "media_portugues", "ranking_uf",
    ]].rename(columns={"rede_desc": "rede"})


def comparacao_metas(indicador_uf: pd.DataFrame, meta_br: pd.DataFrame,
                     meta_uf: pd.DataFrame) -> pd.DataFrame:
    taxas = indicador_uf[indicador_uf["rede"] == "0"][[
        "sigla_uf", "ano", "taxa_alfabetizacao",
    ]].rename(columns={"taxa_alfabetizacao": "taxa_uf"})
    nacional = meta_br[[
        "ano", "meta_alfabetizacao_2025", "meta_alfabetizacao_2030",
    ]].rename(columns={
        "meta_alfabetizacao_2025": "meta_nacional_2025",
        "meta_alfabetizacao_2030": "meta_nacional_2030",
    })
    estadual = meta_uf[["ano", "sigla_uf", "meta_alfabetizacao_2025"]].rename(
        columns={"meta_alfabetizacao_2025": "meta_uf_2025"}
    )
    df = taxas.merge(nacional, on="ano", how="left").merge(
        estadual, on=["ano", "sigla_uf"], how="left"
    )
    df["gap_meta_nacional"] = (df["taxa_uf"] - df["meta_nacional_2025"]).round(2)
    df["gap_meta_uf"] = (df["taxa_uf"] - df["meta_uf_2025"]).round(2)
    df["status_meta"] = pd.NA
    conhecidos = df["taxa_uf"].notna() & df["meta_nacional_2025"].notna()
    df.loc[conhecidos, "status_meta"] = "NAO_ATINGIU"
    df.loc[conhecidos & (df["taxa_uf"] >= df["meta_nacional_2025"]), "status_meta"] = "ATINGIU"
    return df.sort_values(["ano", "taxa_uf"]).reset_index(drop=True)


def run_gold() -> pd.DataFrame:
    indicador = _silver("indicador_municipio")
    indicador_uf = _silver("indicador_uf")
    meta_br, meta_uf, meta_mun = _silver("meta_brasil"), _silver("meta_uf"), _silver("meta_municipio")

    alf = alfabetizacao_por_municipio(indicador, meta_mun)
    visoes = {
        "alfabetizacao_por_municipio": alf,
        "evolucao_temporal": evolucao_temporal(indicador_uf),
        "ranking_municipios": ranking_municipios(indicador),
        "comparacao_metas_nacionais": comparacao_metas(indicador_uf, meta_br, meta_uf),
    }
    resumo = []
    for nome, df in visoes.items():
        saida = _carimbar(df)
        escrever_particionado(saida, cfg.LAKE_DIR / "gold" / nome)
        resumo.append({"visao": nome, "registros": len(df), "colunas": df.shape[1]})
    return pd.DataFrame(resumo)


run = run_gold


if __name__ == "__main__":
    print(run_gold().to_string(index=False))
