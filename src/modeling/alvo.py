"""Separação das variáveis e da proxy municipal de alfabetização."""

import pandas as pd

from ..preprocessing.build_features import carregar_dataset_modelagem

TARGET = "alvo_alfabetizado"
ID_COLS = ["id_municipio", "ano"]


def carregar_dataset() -> pd.DataFrame:
    """Carrega uma linha por município-ano, já materializada na Gold."""
    return carregar_dataset_modelagem()


def separar_xy(df: pd.DataFrame, features: list[str]):
    """Devolve X e y sem permitir que identificadores entrem no modelo."""
    return df[features].copy(), df[TARGET].astype(int).copy()


def resumir_alvo(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("ano")[TARGET]
        .agg(registros="size", proporcao_atingiu_meta="mean")
        .assign(proporcao_atingiu_meta=lambda x: x["proporcao_atingiu_meta"].round(3))
        .reset_index()
    )
