"""
Configuração central da pipeline medalhão local.

Concentra caminhos, mapas de domínio e listas de colunas usados pelas três
camadas (Bronze, Silver, Gold). Manter tudo aqui evita que os módulos
divirjam entre si e torna explícito, num único lugar, *o que* entra na base
analítica e *o que* é deliberadamente descartado.
"""

from pathlib import Path

# =============================================================================
# CAMINHOS
# =============================================================================

RAIZ = Path(__file__).resolve().parents[2]

DADOS_DIR = RAIZ / "data"
EXTERNOS_DIR = DADOS_DIR / "external"
LAKE_DIR = DADOS_DIR / "lake"

# =============================================================================
# CICLOS AVALIATIVOS
# =============================================================================

# O Indicador Criança Alfabetizada só tem dois ciclos publicados.
ANOS_CICLO = [2023, 2024]

# Ciclo do IDEB usado como preditor defasado. É anterior a 2023, portanto
# estava disponível antes do alvo existir — condição para não ser vazamento.
ANO_IDEB_REFERENCIA = 2021

# =============================================================================
# ENTIDADES BRONZE — 5 CSVs do INEP herdados da Fase 2
# =============================================================================

ARQUIVOS_INEP = {
    "indicador_municipio": "br_inep_avaliacao_alfabetizacao_municipio.csv",
    "indicador_uf": "br_inep_avaliacao_alfabetizacao_uf.csv",
    "meta_brasil": "br_inep_avaliacao_alfabetizacao_meta_alfabetizacao_brasil.csv",
    "meta_uf": "br_inep_avaliacao_alfabetizacao_meta_alfabetizacao_uf.csv",
    "meta_municipio": "br_inep_avaliacao_alfabetizacao_meta_alfabetizacao_municipio.csv",
}

# Colunas lidas como texto para preservar o formato original
# (o código IBGE tem 7 dígitos e não pode virar inteiro com zero à esquerda perdido).
COLUNAS_TEXTO = ["id_municipio", "sigla_uf", "rede", "nivel_alfabetizacao"]

# Chave de negócio de cada entidade — base do _record_hash (deduplicação).
CHAVES_HASH = {
    "indicador_municipio": ["ano", "id_municipio", "serie", "rede"],
    "indicador_uf": ["ano", "sigla_uf", "serie", "rede"],
    "meta_brasil": ["ano", "rede"],
    "meta_uf": ["ano", "sigla_uf", "rede"],
    "meta_municipio": ["ano", "id_municipio", "rede"],
    "censo_afd": ["ano", "id_municipio", "localizacao", "dependencia"],
    "censo_atu": ["ano", "id_municipio", "localizacao", "dependencia"],
    "censo_ied": ["ano", "id_municipio", "localizacao", "dependencia"],
    "inse": ["ano", "id_municipio", "tipo_rede", "localizacao"],
    "ideb_anos_iniciais": ["ano", "id_municipio", "rede"],
}

# =============================================================================
# ENTIDADES BRONZE — fontes externas de enriquecimento (INEP / Censo Escolar)
# =============================================================================
#
# `linha_cabecalho` é a linha (1-indexada, como no Excel) que contém os nomes
# técnicos das colunas (NU_ANO_CENSO, CO_MUNICIPIO, ...). As linhas acima são
# rótulos multi-nível em português, que não servem como cabeçalho.

ARQUIVOS_CENSO = {
    "censo_afd": {
        "arquivos": {2023: "AFD_MUNICIPIOS_2023.xlsx", 2024: "AFD_MUNICIPIOS_2024.xlsx"},
        "linha_cabecalho": 11,
        # Adequação da Formação Docente nos Anos Iniciais do Fundamental.
        # Grupo 1 = licenciatura na área que leciona ... Grupo 5 = sem curso superior.
        "colunas": {
            "FUN_AI_CAT_1": "afd_ai_grupo_1",
            "FUN_AI_CAT_2": "afd_ai_grupo_2",
            "FUN_AI_CAT_3": "afd_ai_grupo_3",
            "FUN_AI_CAT_4": "afd_ai_grupo_4",
            "FUN_AI_CAT_5": "afd_ai_grupo_5",
        },
    },
    "censo_atu": {
        "arquivos": {2023: "ATU_MUNICIPIOS_2023.xlsx", 2024: "ATU_MUNICIPIOS_2024.xlsx"},
        "linha_cabecalho": 9,
        # Média de Alunos por Turma. O 2º ano é a série avaliada pelo indicador.
        "colunas": {
            "CRE_CAT_0": "atu_creche",
            "PRE_CAT_0": "atu_pre_escola",
            "FUN_AI_CAT_0": "atu_anos_iniciais",
            "FUN_01_CAT_0": "atu_1_ano",
            "FUN_02_CAT_0": "atu_2_ano",
        },
    },
    "censo_ied": {
        "arquivos": {2023: "IED_MUNICIPIOS_2023.xlsx", 2024: "IED_MUNICIPIOS_2024.xlsx"},
        "linha_cabecalho": 11,
        # Esforço Docente nos Anos Iniciais. Nível 1 = menor esforço (poucos turnos,
        # escolas e alunos) ... Nível 6 = maior esforço.
        "colunas": {
            "FUN_AI_CAT_1": "ied_ai_nivel_1",
            "FUN_AI_CAT_2": "ied_ai_nivel_2",
            "FUN_AI_CAT_3": "ied_ai_nivel_3",
            "FUN_AI_CAT_4": "ied_ai_nivel_4",
            "FUN_AI_CAT_5": "ied_ai_nivel_5",
            "FUN_AI_CAT_6": "ied_ai_nivel_6",
        },
    },
}

ARQUIVO_INSE = "INSE_2023_municipios.xlsx"
ABA_INSE = "INSE_MUN_2023"

ARQUIVO_IDEB = "divulgacao_anos_iniciais_municipios_2025.xlsx"
LINHA_CABECALHO_IDEB = 10

# Métricas do IDEB que existem por ciclo — usadas no "despivotamento" do arquivo
# largo (uma coluna por métrica × ciclo) para o formato longo (uma linha por ciclo).
METRICAS_IDEB = {
    "VL_OBSERVADO": "ideb",
    "VL_NOTA_PORTUGUES": "nota_portugues",
    "VL_NOTA_MATEMATICA": "nota_matematica",
    "VL_INDICADOR_REND": "indicador_rendimento",
    "VL_APROVACAO_%s_SI_4": "taxa_aprovacao",
}

# Valores que representam ausência nas planilhas do INEP.
NULOS_INEP = ["--", "-", "", " "]

# =============================================================================
# MAPAS DE DOMÍNIO
# =============================================================================

# Códigos de rede do INEP no arquivo do indicador.
REDE_MAP = {"0": "total", "2": "estadual", "3": "municipal", "5": "privada"}

# Rede alvo da modelagem: única com cobertura idêntica em 2023 e 2024.
REDE_ALVO = "3"

# Dicionário do INSE (aba `Dicionário` do arquivo).
INSE_TIPO_REDE = {1: "federal", 2: "estadual", 3: "municipal",
                  5: "total_estadual_municipal", 6: "total_geral"}
INSE_LOCALIZACAO = {0: "total", 1: "urbana", 2: "rural"}
INSE_CAPITAL = {1: "capital", 2: "interior"}

# Dois primeiros dígitos do código IBGE do município → sigla da UF.
IBGE_UF = {
    "11": "RO", "12": "AC", "13": "AM", "14": "RR", "15": "PA", "16": "AP",
    "17": "TO", "21": "MA", "22": "PI", "23": "CE", "24": "RN", "25": "PB",
    "26": "PE", "27": "AL", "28": "SE", "29": "BA", "31": "MG", "32": "ES",
    "33": "RJ", "35": "SP", "41": "PR", "42": "SC", "43": "RS", "50": "MS",
    "51": "MT", "52": "GO", "53": "DF",
}

UF_REGIAO = {
    "RO": "Norte", "AC": "Norte", "AM": "Norte", "RR": "Norte", "PA": "Norte",
    "AP": "Norte", "TO": "Norte",
    "MA": "Nordeste", "PI": "Nordeste", "CE": "Nordeste", "RN": "Nordeste",
    "PB": "Nordeste", "PE": "Nordeste", "AL": "Nordeste", "SE": "Nordeste",
    "BA": "Nordeste",
    "MG": "Sudeste", "ES": "Sudeste", "RJ": "Sudeste", "SP": "Sudeste",
    "PR": "Sul", "SC": "Sul", "RS": "Sul",
    "MS": "Centro-Oeste", "MT": "Centro-Oeste", "GO": "Centro-Oeste",
    "DF": "Centro-Oeste",
}

# =============================================================================
# TRATAMENTO DE DATA LEAKAGE
# =============================================================================
#
# Todas estas colunas são função aritmética do alvo (taxa_alfabetizacao) e,
# por isso, NÃO podem entrar na base de modelagem. A lista é explícita para
# que a decisão fique auditável no repositório — e é aplicada por
# `gold.montar_base_ml()`, não "lembrada" caso a caso nos notebooks.

COLUNAS_VAZAMENTO = {
    "media_portugues":
        "mesma escala Saeb da qual a taxa é o percentual de alunos com 743+ pontos",
    "proporcao_aluno_nivel_0": "as proporções por nível somam exatamente para a taxa",
    "proporcao_aluno_nivel_1": "as proporções por nível somam exatamente para a taxa",
    "proporcao_aluno_nivel_2": "as proporções por nível somam exatamente para a taxa",
    "proporcao_aluno_nivel_3": "as proporções por nível somam exatamente para a taxa",
    "proporcao_aluno_nivel_4": "as proporções por nível somam exatamente para a taxa",
    "proporcao_aluno_nivel_5": "as proporções por nível somam exatamente para a taxa",
    "proporcao_aluno_nivel_6": "as proporções por nível somam exatamente para a taxa",
    "proporcao_aluno_nivel_7": "as proporções por nível somam exatamente para a taxa",
    "proporcao_aluno_nivel_8": "as proporções por nível somam exatamente para a taxa",
    "nivel_alfabetizacao": "é a própria taxa discretizada em faixas",
    "meta_alfabetizacao_2024": "meta derivada aritmeticamente da taxa observada em 2023",
    "meta_alfabetizacao_2025": "meta derivada aritmeticamente da taxa observada em 2023",
    "meta_alfabetizacao_2026": "meta derivada aritmeticamente da taxa observada em 2023",
    "meta_alfabetizacao_2027": "meta derivada aritmeticamente da taxa observada em 2023",
    "meta_alfabetizacao_2028": "meta derivada aritmeticamente da taxa observada em 2023",
    "meta_alfabetizacao_2029": "meta derivada aritmeticamente da taxa observada em 2023",
    "meta_alfabetizacao_2030": "meta derivada aritmeticamente da taxa observada em 2023",
    "gap_meta_2025": "diferença entre a taxa e a meta — função direta do alvo",
    "status_meta_2025": "classificação binária da própria taxa",
    "ranking_uf": "ordenação da própria taxa dentro da UF",
    "ideb_2025": "ciclo posterior aos anos do alvo — vazamento do futuro",
    "ideb_2023": "ciclo contemporâneo ao alvo — não disponível no momento da predição",
}

# =============================================================================
# COLUNAS VOLÁTEIS (excluídas da comparação de idempotência)
# =============================================================================
#
# Timestamps de execução mudam legitimamente a cada rodada: idempotência é
# definida sobre o CONTEÚDO DE NEGÓCIO do lake, não sobre metadados.

COLUNAS_VOLATEIS = [
    "_ingestion_timestamp", "_ingestion_date", "_job_name",
    "_silver_processed_at", "_quarentena_ts", "_gold_processed_at",
]
