"""
Camada Gold (SPEC) — visões analíticas e base de modelagem.

Reproduz as 4 visões da Fase 2 (para que a continuidade com o projeto anterior
fique explícita) e acrescenta a entrega própria da Fase 3:

  5. `base_ml_alfabetizacao` — tabela analítica no grão **município × ano**,
     restrita à **rede municipal**, com os dados externos integrados e **sem
     nenhuma coluna que vaze o alvo**.

O tratamento de data leakage acontece **aqui**, na construção da base, e não
dentro do notebook: as colunas proibidas estão listadas em
`config.COLUNAS_VAZAMENTO`, com o motivo de cada exclusão, e a função
`montar_base_ml()` as remove de forma programática. Assim a decisão é
auditável e não depende de ninguém lembrar de excluí-las.
"""

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import config as cfg
from .lake import escrever_particionado, ler_particionado

log = logging.getLogger(__name__)


def _ler_pass(entidade: str) -> pd.DataFrame:
    return ler_particionado(cfg.LAKE_DIR / "silver" / "pass" / entidade)


def _derivar_territorio(df: pd.DataFrame) -> pd.DataFrame:
    """UF e região a partir dos 2 primeiros dígitos do código IBGE.

    Evita depender de uma tabela auxiliar de municípios: o próprio
    `id_municipio` carrega a informação territorial.
    """
    df["sigla_uf"] = df["id_municipio"].str[:2].map(cfg.IBGE_UF).astype("string")
    df["regiao"] = df["sigla_uf"].map(cfg.UF_REGIAO).astype("string")
    return df


# =============================================================================
# VISÕES HERDADAS DA FASE 2
# =============================================================================

def _visao_alfabetizacao_municipio(df_mun, df_meta_mun) -> pd.DataFrame:
    """Rede municipal × meta municipal: gap e status da meta 2025."""
    meta = df_meta_mun[["id_municipio", "ano", "meta_alfabetizacao_2025",
                        "meta_alfabetizacao_2030", "nivel_alfabetizacao"]].rename(
        columns={"meta_alfabetizacao_2025": "meta_2025",
                 "meta_alfabetizacao_2030": "meta_2030"})

    df = (df_mun[df_mun["rede"] == cfg.REDE_ALVO]
          .merge(meta, on=["id_municipio", "ano"], how="left"))

    df = df[["id_municipio", "ano", "serie", "rede_desc", "taxa_alfabetizacao",
             "media_portugues", "meta_2025", "meta_2030",
             "nivel_alfabetizacao"]].rename(columns={"rede_desc": "rede"})
    df["gap_meta_2025"] = (df["taxa_alfabetizacao"] - df["meta_2025"]).round(2)
    df["status_meta_2025"] = np.where(
        df["taxa_alfabetizacao"] >= df["meta_2025"], "ATINGIU", "NAO_ATINGIU")
    return df


def _visao_evolucao_temporal(df_uf) -> pd.DataFrame:
    """Série histórica agregada por UF/ano."""
    return (df_uf[df_uf["rede"] == cfg.REDE_ALVO]
            .groupby(["sigla_uf", "ano", "serie"], as_index=False)
            .agg(taxa_media=("taxa_alfabetizacao", "mean"),
                 taxa_min=("taxa_alfabetizacao", "min"),
                 taxa_max=("taxa_alfabetizacao", "max"),
                 taxa_desvio=("taxa_alfabetizacao", "std"),
                 media_portugues_media=("media_portugues", "mean"))
            .round(2))


def _visao_ranking_municipios(df_mun) -> pd.DataFrame:
    """Ranking dos municípios por taxa dentro de cada UF e ano."""
    df = _derivar_territorio(df_mun[df_mun["rede"] == cfg.REDE_ALVO].copy())
    df["ranking_uf"] = (df.groupby(["sigla_uf", "ano"])["taxa_alfabetizacao"]
                          .rank(ascending=False, method="min").astype("Int64"))
    return df[["id_municipio", "sigla_uf", "regiao", "ano", "serie", "rede_desc",
               "taxa_alfabetizacao", "media_portugues", "ranking_uf"]].rename(
        columns={"rede_desc": "rede"})


def _visao_comparacao_metas(df_uf, df_meta_br, df_meta_uf) -> pd.DataFrame:
    """Taxa por UF × meta nacional × meta estadual."""
    base = (df_uf[df_uf["rede"] == cfg.REDE_ALVO]
            .groupby(["sigla_uf", "ano"], as_index=False)
            .agg(taxa_uf=("taxa_alfabetizacao", "mean")).round(2))

    meta_br = df_meta_br[["ano", "meta_alfabetizacao_2025", "meta_alfabetizacao_2030"]].rename(
        columns={"meta_alfabetizacao_2025": "meta_nacional_2025",
                 "meta_alfabetizacao_2030": "meta_nacional_2030"}).drop_duplicates("ano")
    meta_uf = df_meta_uf[["ano", "sigla_uf", "meta_alfabetizacao_2025"]].rename(
        columns={"meta_alfabetizacao_2025": "meta_uf_2025"}).drop_duplicates(["ano", "sigla_uf"])

    df = base.merge(meta_br, on="ano", how="left").merge(
        meta_uf, on=["ano", "sigla_uf"], how="left")
    df["gap_meta_nacional"] = (df["taxa_uf"] - df["meta_nacional_2025"]).round(2)
    df["gap_meta_uf"] = (df["taxa_uf"] - df["meta_uf_2025"]).round(2)
    df["status_meta"] = np.where(
        df["taxa_uf"] >= df["meta_nacional_2025"], "ATINGIU", "NAO_ATINGIU")
    return df


# =============================================================================
# BASE DE MODELAGEM — a entrega da Fase 3
# =============================================================================

def _fatia_censo(entidade: str) -> pd.DataFrame:
    """Recorte `Localização=Total` + `Dependência=Total` de um indicador do
    Censo Escolar: uma linha por município × ano.

    Trade-off assumido: o agregado *Total* inclui a rede privada, enquanto o
    alvo é da rede municipal. A leitura correta destas features é, portanto,
    "contexto educacional do município", não "característica da rede municipal".
    """
    metricas = list(cfg.ARQUIVOS_CENSO[entidade]["colunas"].values())
    df = _ler_pass(entidade)
    fatia = df[(df["localizacao"] == "Total") & (df["dependencia"] == "Total")]
    return fatia[["id_municipio", "ano"] + metricas].drop_duplicates(["id_municipio", "ano"])


def _fatia_inse() -> pd.DataFrame:
    """Perfil socioeconômico da rede municipal + proporção de alunos rurais.

    Duas informações saem do mesmo arquivo:

    * o **nível socioeconômico** da rede municipal do município
      (`tipo_rede=3`, `localizacao=0`, isto é, urbana + rural);
    * a **proporção rural**, calculada com a contagem de alunos
      (`localizacao=2` sobre `localizacao=0`). Município sem linha rural tem
      proporção **0** — é ausência estrutural de oferta rural, não dado faltante.

    O INSE só existe para 2023 e é replicado para 2024: nível socioeconômico é
    um atributo estrutural do município, que não muda em um ciclo. A
    simplificação está declarada nas limitações do projeto.
    """
    df = _ler_pass("inse")
    municipal = df[df["tipo_rede"] == 3]

    niveis = [f"inse_pc_nivel_{i}" for i in range(1, 9)]
    perfil = municipal[municipal["localizacao"] == 0][
        ["id_municipio", "media_inse", "qtd_alunos_inse", "capital_desc"] + niveis
    ].drop_duplicates("id_municipio").copy()

    # No arquivo do INEP, o percentual de um nível sem nenhum aluno vem em
    # branco, não como 0 — verificado: com os brancos tratados como zero, os
    # oito níveis somam 100% em todas as linhas. É ausência estrutural, e
    # imputá-la pela média distorceria a distribuição socioeconômica.
    perfil[niveis] = perfil[niveis].fillna(0.0)

    alunos = (municipal.pivot_table(index="id_municipio", columns="localizacao",
                                    values="qtd_alunos_inse", aggfunc="max")
                       .rename(columns={0: "alunos_total", 2: "alunos_rural"}))
    alunos["proporcao_rural"] = (
        alunos.get("alunos_rural", pd.Series(dtype=float)).fillna(0)
        / alunos["alunos_total"]
    ).clip(0, 1).round(4)

    return perfil.merge(
        alunos[["proporcao_rural"]].reset_index(), on="id_municipio", how="left"
    )


def _fatia_ideb() -> pd.DataFrame:
    """IDEB da rede municipal no ciclo de referência (2021).

    A defasagem é o mecanismo de controle de leakage: 2021 é **anterior** aos
    dois ciclos do alvo (2023 e 2024), logo estava publicado antes de o alvo
    existir. Os ciclos 2023 e 2025 do mesmo arquivo são deliberadamente
    ignorados — usá-los seria prever o presente com o presente (ou com o futuro).
    """
    df = _ler_pass("ideb_anos_iniciais")
    fatia = df[(df["ano"] == cfg.ANO_IDEB_REFERENCIA) & (df["rede"] == "Municipal")]

    sufixo = f"_{cfg.ANO_IDEB_REFERENCIA}"
    metricas = list(cfg.METRICAS_IDEB.values())
    return (fatia[["id_municipio"] + metricas]
            .rename(columns={m: m + sufixo for m in metricas})
            .drop_duplicates("id_municipio"))


def montar_base_ml() -> pd.DataFrame:
    """Constrói a tabela analítica do modelo supervisionado.

    Grão: município × ano, rede municipal, ciclos 2023 e 2024.
    """
    df = _ler_pass("indicador_municipio")
    df = df[(df["rede"] == cfg.REDE_ALVO) & (df["ano"].isin(cfg.ANOS_CICLO))].copy()
    df = _derivar_territorio(df)

    base = df[["id_municipio", "ano", "sigla_uf", "regiao", "taxa_alfabetizacao"]]

    # Enriquecimento por ano (Censo Escolar) e estrutural (INSE, IDEB defasado).
    for entidade in cfg.ARQUIVOS_CENSO:
        base = base.merge(_fatia_censo(entidade), on=["id_municipio", "ano"], how="left")
    base = base.merge(_fatia_inse(), on="id_municipio", how="left")
    base = base.merge(_fatia_ideb(), on="id_municipio", how="left")

    # Guarda de leakage: nenhuma coluna proibida pode sobreviver até aqui.
    vazamentos = [c for c in base.columns if c in cfg.COLUNAS_VAZAMENTO]
    if vazamentos:
        log.warning(f"[GOLD] Removendo colunas com vazamento do alvo: {vazamentos}")
        base = base.drop(columns=vazamentos)

    return base.sort_values(["ano", "id_municipio"]).reset_index(drop=True)


# =============================================================================
# EXECUÇÃO DA CAMADA
# =============================================================================

def run_gold(verbose: bool = True) -> pd.DataFrame:
    """Silver → Gold. Devolve o sumário por visão."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    df_mun = _ler_pass("indicador_municipio")
    df_uf = _ler_pass("indicador_uf")
    df_meta_br = _ler_pass("meta_brasil")
    df_meta_uf = _ler_pass("meta_uf")
    df_meta_mun = _ler_pass("meta_municipio")

    visoes = {
        "alfabetizacao_por_municipio": _visao_alfabetizacao_municipio(df_mun, df_meta_mun),
        "evolucao_temporal": _visao_evolucao_temporal(df_uf),
        "ranking_municipios": _visao_ranking_municipios(df_mun),
        "comparacao_metas_nacionais": _visao_comparacao_metas(df_uf, df_meta_br, df_meta_uf),
        "base_ml_alfabetizacao": montar_base_ml(),
    }

    sumario = []
    for nome, df in visoes.items():
        if verbose:
            log.info(f"[GOLD] Gerando: {nome}")
        df = df.copy()
        df["_gold_processed_at"] = timestamp
        escrever_particionado(df, cfg.LAKE_DIR / "gold" / nome)
        sumario.append({
            "visao": nome,
            "registros": len(df),
            "colunas": df.shape[1],
            "anos": sorted(df["ano"].dropna().unique().tolist()),
        })

    return pd.DataFrame(sumario)


def carregar_base_ml() -> pd.DataFrame:
    """Lê a base de modelagem já materializada na Gold."""
    return ler_particionado(cfg.LAKE_DIR / "gold" / "base_ml_alfabetizacao")
