"""Configuração única da reimplementação local da pipeline da Fase 2.

A Fase 3 usa deliberadamente apenas os cinco CSVs do Indicador Criança
Alfabetizada já utilizados na Fase 2. As planilhas em ``data/external`` são
opcionais no enunciado e não participam desta versão.
"""

from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
DADOS_DIR = RAIZ / "data"
LAKE_DIR = DADOS_DIR / "lake"
ANOS_CICLO = [2023, 2024]

ARQUIVOS_INEP = {
    "indicador_municipio": "br_inep_avaliacao_alfabetizacao_municipio.csv",
    "indicador_uf": "br_inep_avaliacao_alfabetizacao_uf.csv",
    "meta_brasil": "br_inep_avaliacao_alfabetizacao_meta_alfabetizacao_brasil.csv",
    "meta_uf": "br_inep_avaliacao_alfabetizacao_meta_alfabetizacao_uf.csv",
    "meta_municipio": "br_inep_avaliacao_alfabetizacao_meta_alfabetizacao_municipio.csv",
}

DTYPES = {
    "indicador_municipio": {
        "ano": "Int64", "id_municipio": "string", "serie": "Int64", "rede": "string",
        "taxa_alfabetizacao": "float64", "media_portugues": "float64",
        **{f"proporcao_aluno_nivel_{i}": "float64" for i in range(9)},
    },
    "indicador_uf": {
        "ano": "Int64", "sigla_uf": "string", "serie": "Int64", "rede": "string",
        "taxa_alfabetizacao": "float64", "media_portugues": "float64",
        **{f"proporcao_aluno_nivel_{i}": "float64" for i in range(9)},
    },
    "meta_brasil": {
        "ano": "Int64", "rede": "string", "taxa_alfabetizacao": "float64",
        **{f"meta_alfabetizacao_{ano}": "float64" for ano in range(2024, 2031)},
        "percentual_participacao": "float64",
    },
    "meta_uf": {
        "ano": "Int64", "sigla_uf": "string", "rede": "string",
        "taxa_alfabetizacao": "float64",
        **{f"meta_alfabetizacao_{ano}": "float64" for ano in range(2024, 2031)},
        "percentual_participacao": "float64",
    },
    "meta_municipio": {
        "ano": "Int64", "id_municipio": "string", "rede": "string",
        "taxa_alfabetizacao": "float64",
        **{f"meta_alfabetizacao_{ano}": "float64" for ano in range(2024, 2031)},
        "nivel_alfabetizacao": "string", "percentual_participacao": "float64",
    },
}

CHAVES_HASH = {
    "indicador_municipio": ["ano", "id_municipio", "serie", "rede"],
    "indicador_uf": ["ano", "sigla_uf", "serie", "rede"],
    "meta_brasil": ["ano", "rede"],
    "meta_uf": ["ano", "sigla_uf", "rede"],
    "meta_municipio": ["ano", "id_municipio", "rede"],
}

REDE_MAP = {"0": "total", "2": "estadual", "3": "municipal", "5": "privada"}
REDE_ALVO = "3"

IBGE_UF = {
    "11": "RO", "12": "AC", "13": "AM", "14": "RR", "15": "PA", "16": "AP",
    "17": "TO", "21": "MA", "22": "PI", "23": "CE", "24": "RN", "25": "PB",
    "26": "PE", "27": "AL", "28": "SE", "29": "BA", "31": "MG", "32": "ES",
    "33": "RJ", "35": "SP", "41": "PR", "42": "SC", "43": "RS", "50": "MS",
    "51": "MT", "52": "GO", "53": "DF",
}

UF_REGIAO = {
    **{uf: "Norte" for uf in ("RO", "AC", "AM", "RR", "PA", "AP", "TO")},
    **{uf: "Nordeste" for uf in ("MA", "PI", "CE", "RN", "PB", "PE", "AL", "SE", "BA")},
    **{uf: "Sudeste" for uf in ("MG", "ES", "RJ", "SP")},
    **{uf: "Sul" for uf in ("PR", "SC", "RS")},
    **{uf: "Centro-Oeste" for uf in ("MS", "MT", "GO", "DF")},
}

COLUNAS_VAZAMENTO = {
    "taxa_alfabetizacao", "gap_meta_2025", "status_meta_2025",
    "nivel_alfabetizacao", "meta_2030", "ranking_uf",
    *{f"proporcao_aluno_nivel_{i}" for i in range(9)},
}

COLUNAS_VOLATEIS = [
    "_ingestion_timestamp", "_ingestion_date", "_source_entity", "_source_file",
    "_silver_processed_at", "_quarentena_ts", "_gold_processed_at",
]
