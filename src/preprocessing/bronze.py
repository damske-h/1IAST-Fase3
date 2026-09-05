"""
Camada Bronze (SOR) — ingestão bruta com schema explícito e linhagem.

Espelha `glue_jobs/etl_bronze.py` da Fase 2, agora em pandas e sem AWS.
Para cada entidade:

  1. leitura com **tipagem explícita** (nada de inferência — o código IBGE
     precisa continuar sendo texto de 7 dígitos);
  2. `_record_hash` MD5 da chave de negócio (insumo da deduplicação no Silver);
  3. colunas de linhagem com prefixo `_`;
  4. checks de qualidade com níveis FAIL / WARN e score percentual;
  5. escrita Parquet particionada por `ano=YYYY`.

Novidade em relação à Fase 2: além dos 5 CSVs do INEP, ingerimos 5 entidades
externas de enriquecimento (Censo Escolar AFD/ATU/IED, INSE e IDEB). O Bronze
as recebe **na granularidade original**, sem filtro e sem deduplicação — quem
recorta e limpa é o Silver. Isso preserva a rastreabilidade: qualquer linha da
base analítica pode ser reconduzida à linha bruta que a originou.
"""

import logging
from datetime import datetime, timezone

import pandas as pd

from . import config as cfg
from .lake import escrever_particionado, hash_registro

log = logging.getLogger(__name__)


# =============================================================================
# LEITORES — um por formato de origem
# =============================================================================

def _ler_csv_inep(entidade: str) -> pd.DataFrame:
    """Lê um dos 5 CSVs do INEP com tipagem explícita."""
    caminho = cfg.DADOS_DIR / cfg.ARQUIVOS_INEP[entidade]
    return pd.read_csv(
        caminho,
        dtype={c: "string" for c in cfg.COLUNAS_TEXTO},
        encoding="utf-8",
    )


def _normalizar_codigo_municipio(serie: pd.Series) -> pd.Series:
    """`CO_MUNICIPIO` vem como float nas planilhas (5300108.0). Converte para
    texto de 7 dígitos, formato do `id_municipio` dos CSVs do INEP."""
    return (pd.to_numeric(serie, errors="coerce")
              .astype("Int64")
              .astype("string")
              .str.zfill(7))


def _ler_planilha_inep(caminho, linha_cabecalho: int, aba=0) -> pd.DataFrame:
    """Lê uma planilha do INEP com cabeçalho multi-nível.

    `linha_cabecalho` é a linha do Excel (1-indexada) com os nomes técnicos.
    As linhas de rótulo em português acima dela são descartadas, assim como os
    rodapés de nota do fim do arquivo (identificados por `CO_MUNICIPIO` nulo).
    """
    df = pd.read_excel(
        caminho,
        sheet_name=aba,
        skiprows=linha_cabecalho - 1,
        na_values=cfg.NULOS_INEP,
    )
    df = df[df["CO_MUNICIPIO"].notna()].copy()
    df["id_municipio"] = _normalizar_codigo_municipio(df["CO_MUNICIPIO"])
    return df[df["id_municipio"].str.len() == 7]


def _ler_censo(entidade: str) -> pd.DataFrame:
    """Concatena os dois anos de um indicador do Censo Escolar (AFD/ATU/IED),
    mantendo todas as fatias de localização e dependência administrativa."""
    spec = cfg.ARQUIVOS_CENSO[entidade]
    partes = []

    for ano, arquivo in spec["arquivos"].items():
        df = _ler_planilha_inep(cfg.EXTERNOS_DIR / arquivo, spec["linha_cabecalho"])
        df = df.rename(columns=spec["colunas"])
        df = df.assign(
            ano=ano,
            sigla_uf=df["SG_UF"].astype("string"),
            regiao=df["NO_REGIAO"].astype("string"),
            localizacao=df["NO_CATEGORIA"].astype("string"),
            dependencia=df["NO_DEPENDENCIA"].astype("string"),
        )
        colunas = (["ano", "id_municipio", "sigla_uf", "regiao",
                    "localizacao", "dependencia"] + list(spec["colunas"].values()))
        partes.append(df[colunas])

    return pd.concat(partes, ignore_index=True)


def _ler_inse() -> pd.DataFrame:
    """Lê o INSE municipal de 2023.

    O arquivo é ingerido **com as duplicatas** que ele traz de fábrica (a mesma
    chave município × rede × localização se repete até 7 vezes). Removê-las é
    tarefa do Silver, via `_record_hash` — assim a duplicidade fica registrada
    e contabilizada, em vez de sumir silenciosamente na leitura.
    """
    caminho = cfg.EXTERNOS_DIR / cfg.ARQUIVO_INSE
    df = pd.read_excel(caminho, sheet_name=cfg.ABA_INSE, na_values=cfg.NULOS_INEP)
    df["id_municipio"] = _normalizar_codigo_municipio(df["CO_MUNICIPIO"])
    df = df[df["id_municipio"].str.len() == 7]

    renomear = {
        "NU_ANO_SAEB": "ano", "SG_UF": "sigla_uf",
        "TP_TIPO_REDE": "tipo_rede", "TP_LOCALIZACAO": "localizacao",
        "TP_CAPITAL": "capital", "QTD_ALUNOS_INSE": "qtd_alunos_inse",
        "MEDIA_INSE": "media_inse",
        **{f"PC_NIVEL_{i}": f"inse_pc_nivel_{i}" for i in range(1, 9)},
    }
    df = df.rename(columns=renomear)
    return df[list(renomear.values()) + ["id_municipio"]]


def _ler_ideb() -> pd.DataFrame:
    """Lê o IDEB dos Anos Iniciais e o converte de formato largo para longo.

    O arquivo original tem uma coluna por métrica × ciclo (`VL_OBSERVADO_2005`,
    `VL_OBSERVADO_2007`, ...). Sem despivotar não há como particionar por ano,
    que é a chave física do lake. O resultado tem uma linha por
    município × rede × ciclo — o mesmo grão das demais entidades.
    """
    df = _ler_planilha_inep(
        cfg.EXTERNOS_DIR / cfg.ARQUIVO_IDEB, cfg.LINHA_CABECALHO_IDEB
    )
    df = df[df["REDE"].notna()].copy()
    df["rede"] = df["REDE"].astype("string")
    df["sigla_uf"] = df["SG_UF"].astype("string")

    # Ciclos presentes no arquivo, deduzidos do sufixo das colunas.
    anos = sorted({
        int(str(c)[-4:]) for c in df.columns
        if str(c).startswith("VL_OBSERVADO_")
    })

    partes = []
    for ano in anos:
        bloco = df[["id_municipio", "sigla_uf", "rede"]].copy()
        bloco["ano"] = ano
        for prefixo, nome in cfg.METRICAS_IDEB.items():
            coluna = prefixo % ano if "%s" in prefixo else f"{prefixo}_{ano}"
            bloco[nome] = pd.to_numeric(df[coluna], errors="coerce") if coluna in df else pd.NA
        partes.append(bloco)

    return pd.concat(partes, ignore_index=True)


# =============================================================================
# CHECKS DE QUALIDADE (níveis FAIL / WARN, como na Fase 2)
# =============================================================================

CHECKS = {
    "indicador_municipio": [
        ("id_municipio", "not_null", None, "FAIL"),
        ("ano", "not_null", None, "FAIL"),
        ("taxa_alfabetizacao", "not_null", None, "WARN"),
        ("taxa_alfabetizacao", "range", (0, 100), "FAIL"),
        ("rede", "in_set", {"0", "2", "3", "5"}, "WARN"),
    ],
    "indicador_uf": [
        ("sigla_uf", "not_null", None, "FAIL"),
        ("ano", "not_null", None, "FAIL"),
        ("taxa_alfabetizacao", "range", (0, 100), "WARN"),
    ],
    "meta_brasil": [("ano", "not_null", None, "FAIL")],
    "meta_uf": [("sigla_uf", "not_null", None, "FAIL"), ("ano", "not_null", None, "FAIL")],
    "meta_municipio": [("id_municipio", "not_null", None, "FAIL"), ("ano", "not_null", None, "FAIL")],
    "censo_afd": [
        ("id_municipio", "not_null", None, "FAIL"),
        ("afd_ai_grupo_1", "range", (0, 100), "FAIL"),
    ],
    "censo_atu": [
        ("id_municipio", "not_null", None, "FAIL"),
        ("atu_anos_iniciais", "range", (0, 80), "WARN"),
    ],
    "censo_ied": [
        ("id_municipio", "not_null", None, "FAIL"),
        ("ied_ai_nivel_1", "range", (0, 100), "FAIL"),
    ],
    "inse": [
        ("id_municipio", "not_null", None, "FAIL"),
        ("media_inse", "range", (0, 10), "FAIL"),
    ],
    "ideb_anos_iniciais": [
        ("id_municipio", "not_null", None, "FAIL"),
        ("ideb", "range", (0, 10), "FAIL"),
    ],
}


def _checar_qualidade(df: pd.DataFrame, entidade: str) -> float:
    """Roda os checks da entidade. Levanta exceção em falha crítica (FAIL)."""
    checks = CHECKS.get(entidade, [])
    if not checks:
        return 100.0

    aprovados = criticos = 0
    for coluna, tipo, parametro, nivel in checks:
        if coluna not in df.columns:
            ok, detalhe = False, "coluna ausente"
        elif tipo == "not_null":
            n = int(df[coluna].isna().sum())
            ok, detalhe = n == 0, f"{n} nulos"
        elif tipo == "range":
            minimo, maximo = parametro
            valores = pd.to_numeric(df[coluna], errors="coerce")
            n = int((valores.notna() & ~valores.between(minimo, maximo)).sum())
            ok, detalhe = n == 0, f"{n} fora de [{minimo}, {maximo}]"
        elif tipo == "in_set":
            n = int((~df[coluna].isin(parametro) & df[coluna].notna()).sum())
            ok, detalhe = n == 0, f"{n} fora de {sorted(parametro)}"
        else:
            ok, detalhe = False, "tipo de check desconhecido"

        status = "PASS" if ok else nivel
        mensagem = f"[DQ:BRONZE] {entidade} | {status} | {tipo}({coluna}) | {detalhe}"
        if ok:
            aprovados += 1
            log.info(mensagem)
        elif nivel == "FAIL":
            criticos += 1
            log.error(mensagem)
        else:
            log.warning(mensagem)

    score = round(aprovados / len(checks) * 100, 1)
    log.info(f"[DQ:BRONZE] {entidade} | score={score}% | criticos={criticos}")
    if criticos:
        raise ValueError(f"[DQ:BRONZE] {criticos} check(s) crítico(s) em '{entidade}'")
    return score


# =============================================================================
# EXECUÇÃO DA CAMADA
# =============================================================================

def run_bronze(verbose: bool = True) -> pd.DataFrame:
    """RAW → Bronze. Devolve o sumário por entidade."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    data = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    leitores = {
        **{ent: (lambda e=ent: _ler_csv_inep(e)) for ent in cfg.ARQUIVOS_INEP},
        **{ent: (lambda e=ent: _ler_censo(e)) for ent in cfg.ARQUIVOS_CENSO},
        "inse": _ler_inse,
        "ideb_anos_iniciais": _ler_ideb,
    }

    sumario = []
    for entidade, leitor in leitores.items():
        if verbose:
            log.info(f"[BRONZE] Ingerindo: {entidade}")

        df = leitor()
        df["_record_hash"] = hash_registro(df, cfg.CHAVES_HASH[entidade])
        df["_ingestion_timestamp"] = timestamp
        df["_ingestion_date"] = data
        df["_source_entity"] = entidade

        score = _checar_qualidade(df, entidade)
        escrever_particionado(df, cfg.LAKE_DIR / "bronze" / entidade)

        sumario.append({
            "entidade": entidade,
            "registros": len(df),
            "anos": sorted(df["ano"].dropna().unique().tolist()),
            "hashes_distintos": int(df["_record_hash"].nunique()),
            "score_dq": score,
        })

    return pd.DataFrame(sumario)
