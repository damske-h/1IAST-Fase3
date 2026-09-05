"""
Camada Silver (SOT) — limpeza, padronização, deduplicação e quarentena.

Espelha `glue_jobs/etl_silver.py` da Fase 2. Para cada entidade do Bronze:

  1. **deduplicação** pelo `_record_hash` calculado no Bronze;
  2. transformações de padronização (decode de domínios, arredondamentos);
  3. regras de qualidade materializadas como **colunas booleanas `_dq_*`**,
     consolidadas em `_dq_passou`;
  4. roteamento: aprovados vão para `silver/pass/<entidade>/ano=YYYY/`;
     reprovados vão para `silver/quarentena/<entidade>/anomesdia=YYYYMMDD/`
     com `_quarentena_motivo` — registro reprovado **não é descartado**,
     fica auditável e reprocessável.

A quarentena é particionada pela data de processamento, e não por `ano`,
porque o `ano` pode ser justamente o campo inválido que causou a reprovação.
"""

import logging
from datetime import datetime, timezone

import pandas as pd

from . import config as cfg
from .lake import escrever_particionado, escrever_sem_particao, ler_particionado

log = logging.getLogger(__name__)

ENTIDADES = (
    list(cfg.ARQUIVOS_INEP)
    + list(cfg.ARQUIVOS_CENSO)
    + ["inse", "ideb_anos_iniciais"]
)


# =============================================================================
# TRANSFORMAÇÕES POR ENTIDADE
# =============================================================================

def _transformar(df: pd.DataFrame, entidade: str) -> pd.DataFrame:
    """Padronizações específicas de cada entidade."""
    if entidade in ("indicador_municipio", "indicador_uf"):
        # Decode do código de rede do INEP (0/2/3/5) para texto legível.
        df["rede_desc"] = df["rede"].map(cfg.REDE_MAP).fillna(df["rede"])
        for coluna in ("taxa_alfabetizacao", "media_portugues"):
            df[coluna] = pd.to_numeric(df[coluna], errors="coerce").round(2)

    elif entidade.startswith("meta_"):
        metas = [c for c in df.columns if c.startswith("meta_alfabetizacao_")]
        df[metas] = df[metas].apply(pd.to_numeric, errors="coerce").round(2)

    elif entidade.startswith("censo_"):
        metricas = list(cfg.ARQUIVOS_CENSO[entidade]["colunas"].values())
        df[metricas] = df[metricas].apply(pd.to_numeric, errors="coerce").round(2)

    elif entidade == "inse":
        # Decode dos domínios documentados na aba `Dicionário` do arquivo.
        df["tipo_rede_desc"] = df["tipo_rede"].map(cfg.INSE_TIPO_REDE)
        df["localizacao_desc"] = df["localizacao"].map(cfg.INSE_LOCALIZACAO)
        df["capital_desc"] = df["capital"].map(cfg.INSE_CAPITAL)
        numericas = ["media_inse", "qtd_alunos_inse"] + [
            f"inse_pc_nivel_{i}" for i in range(1, 9)
        ]
        df[numericas] = df[numericas].apply(pd.to_numeric, errors="coerce")

    elif entidade == "ideb_anos_iniciais":
        metricas = list(cfg.METRICAS_IDEB.values())
        df[metricas] = df[metricas].apply(pd.to_numeric, errors="coerce")

    return df


# =============================================================================
# REGRAS DE QUALIDADE — colunas booleanas _dq_*
# =============================================================================
#
# Cada regra é uma tupla (nome da coluna _dq_, função, descrição do motivo).
# A descrição é o texto que aparece em `_quarentena_motivo`.

def _regras(entidade: str):
    ano_valido = lambda d: d["ano"].notna() & d["ano"].between(2000, 2030)
    municipio_valido = lambda d: (
        d["id_municipio"].notna() & (d["id_municipio"].astype("string").str.len() == 7)
    )

    comuns = [("_dq_ano_valido", ano_valido, "ano fora de [2000, 2030] ou nulo")]

    if entidade == "indicador_municipio":
        return comuns + [
            ("_dq_municipio_valido", municipio_valido,
             "id_municipio ausente ou fora do padrão de 7 dígitos IBGE"),
            ("_dq_taxa_valida",
             lambda d: d["taxa_alfabetizacao"].notna()
                       & d["taxa_alfabetizacao"].between(0, 100),
             "taxa_alfabetizacao nula ou fora de [0, 100]"),
        ]

    if entidade == "indicador_uf":
        return comuns + [
            ("_dq_uf_valida",
             lambda d: d["sigla_uf"].notna() & (d["sigla_uf"].str.len() == 2),
             "sigla_uf ausente ou fora do padrão de 2 caracteres"),
        ]

    if entidade == "meta_brasil":
        return comuns

    if entidade == "meta_uf":
        return comuns + [
            ("_dq_uf_valida", lambda d: d["sigla_uf"].notna(), "sigla_uf nula"),
        ]

    if entidade in ("meta_municipio", "ideb_anos_iniciais"):
        return comuns + [
            ("_dq_municipio_valido", municipio_valido,
             "id_municipio ausente ou fora do padrão de 7 dígitos IBGE"),
        ]

    if entidade.startswith("censo_"):
        return comuns + [
            ("_dq_municipio_valido", municipio_valido,
             "id_municipio ausente ou fora do padrão de 7 dígitos IBGE"),
            ("_dq_dimensoes_validas",
             lambda d: d["localizacao"].notna() & d["dependencia"].notna(),
             "localização ou dependência administrativa ausente"),
        ]

    if entidade == "inse":
        return comuns + [
            ("_dq_municipio_valido", municipio_valido,
             "id_municipio ausente ou fora do padrão de 7 dígitos IBGE"),
            ("_dq_inse_valido",
             lambda d: d["media_inse"].notna() & d["media_inse"].between(0, 10),
             "media_inse nula ou fora da escala [0, 10]"),
        ]

    return comuns


def _aplicar_dq(df: pd.DataFrame, entidade: str) -> tuple:
    """Materializa as colunas `_dq_*` e consolida em `_dq_passou`."""
    regras = _regras(entidade)
    for coluna, funcao, _ in regras:
        df[coluna] = funcao(df).fillna(False).astype(bool)

    df["_dq_passou"] = df[[c for c, _, _ in regras]].all(axis=1)
    return df, regras


def _motivo_quarentena(df: pd.DataFrame, regras) -> pd.Series:
    """Concatena as descrições de todas as regras violadas na linha."""
    return pd.Series(
        [
            "; ".join(desc for coluna, _, desc in regras if not linha[coluna])
            for _, linha in df.iterrows()
        ],
        index=df.index,
        dtype="string",
    )


# =============================================================================
# EXECUÇÃO DA CAMADA
# =============================================================================

def run_silver(verbose: bool = True) -> pd.DataFrame:
    """Bronze → Silver. Devolve o sumário por entidade."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    anomesdia = datetime.now(timezone.utc).strftime("%Y%m%d")

    sumario = []
    for entidade in ENTIDADES:
        if verbose:
            log.info(f"[SILVER] Processando: {entidade}")

        df = ler_particionado(cfg.LAKE_DIR / "bronze" / entidade)
        lidos = len(df)

        # 1. Deduplicação pelo hash de negócio calculado no Bronze.
        df = df.drop_duplicates(subset="_record_hash", keep="first").reset_index(drop=True)
        duplicados = lidos - len(df)

        # 2. Padronização e 3. regras de qualidade.
        df = _transformar(df, entidade)
        df, regras = _aplicar_dq(df, entidade)
        df["_silver_processed_at"] = timestamp

        # 4. Roteamento pass / quarentena.
        aprovados = df[df["_dq_passou"]].copy()
        reprovados = df[~df["_dq_passou"]].copy()

        escrever_particionado(aprovados, cfg.LAKE_DIR / "silver" / "pass" / entidade)
        if not reprovados.empty:
            reprovados["_quarentena_motivo"] = _motivo_quarentena(reprovados, regras)
            reprovados["_quarentena_ts"] = timestamp
            escrever_sem_particao(
                reprovados,
                cfg.LAKE_DIR / "silver" / "quarentena" / entidade / f"anomesdia={anomesdia}",
                "dados",
            )

        total = len(df)
        sumario.append({
            "entidade": entidade,
            "lidos_bronze": lidos,
            "duplicados_removidos": duplicados,
            "pass": len(aprovados),
            "quarentena": len(reprovados),
            "score_dq": round(len(aprovados) / total * 100, 1) if total else 0.0,
        })

    return pd.DataFrame(sumario)
