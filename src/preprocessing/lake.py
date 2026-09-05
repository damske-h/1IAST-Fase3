"""
Data lake local — leitura/escrita particionada e verificação de idempotência.

Reproduz, em disco local com pandas + Parquet, a semântica de escrita que a
Fase 2 usava no S3 via Spark:

  * particionamento Hive-style `ano=YYYY/` dentro de cada entidade;
  * `partitionOverwriteMode=dynamic` — reprocessar um ano sobrescreve **apenas**
    aquela partição, preservando o histórico dos demais ciclos avaliativos.

O par `snapshot_camada()` / `comparar_snapshots()` permite provar idempotência:
executar a mesma camada duas vezes deve produzir partições, contagens e hashes
de conteúdo idênticos.
"""

import hashlib
import shutil
from pathlib import Path

import pandas as pd

from .config import COLUNAS_VOLATEIS


def escrever_particionado(df: pd.DataFrame, destino: Path) -> None:
    """Grava `df` em `destino/ano=YYYY/dados.parquet`.

    Sobrescreve somente as partições de ano presentes neste lote — as demais
    permanecem intactas (overwrite dinâmico).
    """
    destino = Path(destino)
    if df.empty:
        return

    for ano, df_ano in df.groupby("ano"):
        particao = destino / f"ano={int(ano)}"
        if particao.exists():
            shutil.rmtree(particao)
        particao.mkdir(parents=True, exist_ok=True)
        (df_ano.drop(columns=["ano"])
               .reset_index(drop=True)
               .to_parquet(particao / "dados.parquet", index=False))


def ler_particionado(origem: Path) -> pd.DataFrame:
    """Lê todas as partições `ano=YYYY` de uma entidade.

    A coluna `ano` é reconstruída a partir do nome do diretório — é assim que
    um leitor Hive-style (Spark, Athena) enxerga a partição.
    """
    origem = Path(origem)
    if not origem.exists():
        raise FileNotFoundError(
            f"Partição não encontrada: {origem}. Rode a camada anterior primeiro."
        )

    partes = []
    for particao in sorted(origem.glob("ano=*")):
        arquivo = particao / "dados.parquet"
        if not arquivo.exists():
            continue
        df = pd.read_parquet(arquivo)
        df["ano"] = int(particao.name.split("=")[1])
        partes.append(df)

    if not partes:
        raise FileNotFoundError(f"Nenhuma partição legível em {origem}")

    return pd.concat(partes, ignore_index=True)


def escrever_sem_particao(df: pd.DataFrame, destino: Path, nome: str) -> None:
    """Grava um diretório sem partição de ano (usado pela quarentena, que é
    particionada pela data de processamento)."""
    destino = Path(destino)
    if destino.exists():
        shutil.rmtree(destino)
    destino.mkdir(parents=True, exist_ok=True)
    df.reset_index(drop=True).to_parquet(destino / f"{nome}.parquet", index=False)


def snapshot_camada(prefixo: Path, raiz: Path) -> pd.DataFrame:
    """Fingerprint do estado de uma camada do lake.

    Devolve uma linha por partição física: caminho relativo, nº de linhas e
    hash MD5 do conteúdo de negócio (colunas voláteis removidas, colunas e
    linhas ordenadas para que a comparação não dependa de ordem).
    """
    prefixo, raiz = Path(prefixo), Path(raiz)
    linhas = []

    for arquivo in sorted(prefixo.rglob("*.parquet")):
        df = pd.read_parquet(arquivo)
        df = df.drop(columns=[c for c in COLUNAS_VOLATEIS if c in df.columns])
        df = df.reindex(sorted(df.columns), axis=1)
        df = df.sort_values(by=list(df.columns)).reset_index(drop=True)
        linhas.append({
            "particao": str(arquivo.parent.relative_to(raiz)).replace("\\", "/"),
            "n_linhas": len(df),
            "hash_negocio": hashlib.md5(df.to_csv(index=False).encode()).hexdigest(),
        })

    return pd.DataFrame(linhas).sort_values("particao").reset_index(drop=True)


def comparar_snapshots(snap_a: pd.DataFrame, snap_b: pd.DataFrame) -> pd.DataFrame:
    """Compara dois snapshots partição a partição.

    A coluna `idempotente` é True quando a partição existe nos dois snapshots
    com a mesma contagem e o mesmo hash de conteúdo.
    """
    comp = snap_a.merge(
        snap_b, on="particao", how="outer",
        suffixes=("_exec1", "_exec2"), indicator=True,
    )
    comp["idempotente"] = (
        (comp["_merge"] == "both")
        & (comp["n_linhas_exec1"] == comp["n_linhas_exec2"])
        & (comp["hash_negocio_exec1"] == comp["hash_negocio_exec2"])
    )
    return comp.drop(columns="_merge")


def hash_registro(df: pd.DataFrame, chaves: list) -> pd.Series:
    """MD5 da chave de negócio composta — base da deduplicação no Silver."""
    return (df[chaves].astype("string").fillna("")
            .agg("|".join, axis=1)
            .map(lambda s: hashlib.md5(s.encode()).hexdigest()))
