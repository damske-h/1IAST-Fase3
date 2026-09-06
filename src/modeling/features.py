"""
Seleção de variáveis e pipeline de pré-processamento integrada ao modelo.

Este módulo materializa as decisões tomadas na análise exploratória
(`notebooks/02_analise_exploratoria.ipynb`). Elas ficam aqui, em código, e não
espalhadas pelo notebook, para que a escolha de cada variável seja auditável e
reproduzível.

O pré-processamento é montado **dentro** de um `Pipeline` do Scikit-learn,
como o enunciado exige. A consequência prática é que imputação, padronização e
codificação são ajustadas **apenas no fold de treino** de cada validação — a
média usada para padronizar nunca vê o conjunto de validação.
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# =============================================================================
# BLOCOS COMPOSICIONAIS — categoria de referência descartada
# =============================================================================
#
# AFD, IED e os níveis do INSE são percentuais que somam 100% por construção
# (verificado na seção 5 da EDA). Manter todas as categorias cria colinearidade
# perfeita: a última é `100 - (soma das demais)`, informação zero. Descartamos
# uma categoria de referência por bloco, escolhida pela interpretação — os
# coeficientes passam a ser lidos como "efeito de deslocar um ponto percentual
# da categoria de referência para esta".

REFERENCIAS = {
    "afd_ai_grupo_5": "docentes sem curso superior",
    "ied_ai_nivel_1": "menor nível de esforço docente",
    "inse_pc_nivel_1": "nível socioeconômico mais baixo",
}

AFD = [f"afd_ai_grupo_{i}" for i in (1, 2, 3, 4)]
IED = [f"ied_ai_nivel_{i}" for i in (2, 3, 4, 5, 6)]
INSE_NIVEIS = [f"inse_pc_nivel_{i}" for i in range(2, 9)]

# Médias de alunos por turma: não são composicionais, entram todas.
ATU = ["atu_creche", "atu_pre_escola", "atu_anos_iniciais", "atu_1_ano", "atu_2_ano"]

INSE_OUTRAS = ["media_inse", "qtd_alunos_inse", "proporcao_rural"]

# =============================================================================
# BLOCO IDEB — ausência informativa
# =============================================================================
#
# Do bloco original de cinco variáveis restam duas: `ideb_2021` (índice
# composto, o mais interpretável para um gestor) e `taxa_aprovacao_2021` (a
# dimensão de fluxo escolar, distinta da proficiência). As notas de Português e
# Matemática correlacionam 0,95+ com o IDEB, e `indicador_rendimento_2021`
# correlaciona 0,996 com a taxa de aprovação — redundância pura.
#
# Estas duas têm 13,3% e 5,0% de ausência, concentrada em municípios pequenos e
# isolados: a ausência **é informativa**, e por isso a imputação vem com
# `add_indicator=True`.

IDEB = ["ideb_2021", "taxa_aprovacao_2021"]

NUMERICAS_GERAIS = AFD + IED + INSE_NIVEIS + ATU + INSE_OUTRAS
NUMERICAS_IDEB = IDEB

# =============================================================================
# CATEGÓRICAS
# =============================================================================
#
# `sigla_uf` é o controle geográfico obrigatório (ver o paradoxo de Simpson na
# seção 4 da EDA). `regiao` **não entra**: é função determinística da UF, de
# modo que as dummies de UF já a contêm — incluí-la só somaria colinearidade.
# Ela permanece na base para agregação e relatório.

CATEGORICAS = ["sigla_uf", "capital_desc"]

# =============================================================================
# EXCLUÍDAS — com o motivo, para a decisão ficar auditável
# =============================================================================

EXCLUIDAS = {
    "id_municipio": "identificador; entra como grupo do GroupKFold, não como preditor",
    "taxa_alfabetizacao": "é o alvo",
    "regiao": "função determinística de sigla_uf — as dummies de UF já a codificam",
    "ano": ("o split temporal treina em 2023 e testa em 2024; um coeficiente de ano "
            "estimado só com 2023 não se aplica a 2024. O efeito de ciclo fica fora "
            "do escopo: o modelo descreve a estrutura municipal, não a tendência nacional"),
    "nota_portugues_2021": "correlaciona 0,956 com ideb_2021 — redundante",
    "nota_matematica_2021": "correlaciona 0,961 com ideb_2021 — redundante",
    "indicador_rendimento_2021": "correlaciona 0,996 com taxa_aprovacao_2021 — redundante",
    **{coluna: f"categoria de referência do bloco composicional ({desc})"
       for coluna, desc in REFERENCIAS.items()},
}


def colunas_do_modelo():
    """Lista, na ordem, as colunas que alimentam a pipeline."""
    return NUMERICAS_GERAIS + NUMERICAS_IDEB + CATEGORICAS


def construir_pipeline(C: float = 1.0, max_iter: int = 2000,
                       random_state: int = 42) -> Pipeline:
    """Monta o `Pipeline` completo: pré-processamento + Regressão Logística.

    Três ramos no `ColumnTransformer`:

    * numéricas gerais — imputação pela mediana + padronização;
    * numéricas do IDEB — imputação pela mediana **com indicador de ausência**,
      porque não ter IDEB divulgado é sinal, não ruído;
    * categóricas — imputação pela categoria mais frequente + *dummy encoding*
      (`drop="first"`), que remove uma categoria para evitar multicolinearidade
      entre as dummies.

    `handle_unknown="ignore"` cobre o caso real do split temporal: os municípios
    do Acre só aparecem em 2024, então a UF `AC` é desconhecida para um modelo
    treinado em 2023.
    """
    numericas_gerais = Pipeline([
        ("imputacao", SimpleImputer(strategy="median")),
        ("padronizacao", StandardScaler()),
    ])

    numericas_ideb = Pipeline([
        ("imputacao", SimpleImputer(strategy="median", add_indicator=True)),
        ("padronizacao", StandardScaler()),
    ])

    categoricas = Pipeline([
        ("imputacao", SimpleImputer(strategy="most_frequent")),
        ("codificacao", OneHotEncoder(drop="first", handle_unknown="ignore",
                                      sparse_output=False)),
    ])

    preprocessamento = ColumnTransformer([
        ("numericas", numericas_gerais, NUMERICAS_GERAIS),
        ("ideb", numericas_ideb, NUMERICAS_IDEB),
        ("categoricas", categoricas, CATEGORICAS),
    ], remainder="drop", verbose_feature_names_out=False)

    # A penalidade L2 é o padrão do LogisticRegression e é o que queremos:
    # estabiliza os coeficientes sob a colinearidade residual entre blocos.
    # (O argumento `penalty` foi depreciado no scikit-learn 1.8; omiti-lo mantém L2.)
    modelo = LogisticRegression(
        C=C,
        solver="lbfgs",
        max_iter=max_iter,
        random_state=random_state,
    )

    return Pipeline([("preprocessamento", preprocessamento), ("modelo", modelo)])


def nomes_das_features(pipeline: Pipeline):
    """Nomes das colunas na saída do pré-processamento, para ler os coeficientes."""
    return list(pipeline.named_steps["preprocessamento"].get_feature_names_out())
