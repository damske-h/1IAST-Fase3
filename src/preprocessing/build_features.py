"""Preparação analítica executada depois da Gold.

Este módulo não faz parte da reprodução Bronze/Silver/Gold. Aqui, depois de a
EDA demonstrar o risco de leakage, são escolhidas as mesmas nove colunas do
projeto de referência e materializado o dataset usado pelos modelos.
"""

import pandas as pd

from . import config as cfg
from .lake import ler_particionado

FEATURES_DIR = cfg.DADOS_DIR / "features"
DATASET_PATH = FEATURES_DIR / "dataset_modelagem.parquet"


def _loo_media_uf(df: pd.DataFrame) -> pd.Series:
    soma = df.groupby(["sigla_uf", "ano"])["taxa_alfabetizacao"].transform("sum")
    contagem = df.groupby(["sigla_uf", "ano"])["taxa_alfabetizacao"].transform("count")
    return ((soma - df["taxa_alfabetizacao"]) / (contagem - 1)).round(2)


def construir_dataset_modelagem() -> pd.DataFrame:
    gold = ler_particionado(cfg.LAKE_DIR / "gold" / "alfabetizacao_por_municipio")
    indicador = ler_particionado(cfg.LAKE_DIR / "silver" / "pass" / "indicador_municipio")
    meta_uf = ler_particionado(cfg.LAKE_DIR / "silver" / "pass" / "meta_uf")

    # A taxa, o gap, o status, o nível e as proporções SAEB permanecem nas
    # camadas de dados. Só aqui deixam de ser candidatas a preditoras.
    df = gold.dropna(subset=["status_meta_2025"]).copy()
    df["alvo_alfabetizado"] = (df["status_meta_2025"] == "ATINGIU").astype(int)
    df["sigla_uf"] = df["id_municipio"].str[:2].map(cfg.IBGE_UF)
    df["regiao"] = df["sigla_uf"].map(cfg.UF_REGIAO)

    metas = meta_uf[["ano", "sigla_uf", "meta_alfabetizacao_2025"]].rename(
        columns={"meta_alfabetizacao_2025": "meta_uf_2025"}
    )
    df = df.merge(metas, on=["ano", "sigla_uf"], how="left")

    pares = indicador[indicador["rede"] == cfg.REDE_ALVO][[
        "id_municipio", "ano", "taxa_alfabetizacao",
    ]].copy()
    pares["sigla_uf"] = pares["id_municipio"].str[:2].map(cfg.IBGE_UF)
    pares["taxa_media_uf_loo"] = _loo_media_uf(pares)
    df = df.merge(
        pares[["id_municipio", "ano", "taxa_media_uf_loo"]],
        on=["id_municipio", "ano"], how="left",
    )

    colunas = [
        "id_municipio", "ano", "sigla_uf", "regiao", "media_portugues",
        "meta_2025", "meta_uf_2025", "taxa_media_uf_loo", "alvo_alfabetizado",
    ]
    return df[colunas].drop_duplicates(["id_municipio", "ano"]).reset_index(drop=True)


def salvar_dataset_modelagem() -> pd.DataFrame:
    df = construir_dataset_modelagem()
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATASET_PATH, index=False)
    return df


def carregar_dataset_modelagem() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        return salvar_dataset_modelagem()
    return pd.read_parquet(DATASET_PATH)


if __name__ == "__main__":
    dataset = salvar_dataset_modelagem()
    print(f"{DATASET_PATH}: {dataset.shape[0]} linhas x {dataset.shape[1]} colunas")
