"""
Seleção de variáveis e pipeline de pré-processamento integrada ao modelo.

As decisões vêm da EDA (`notebooks/02`) e moram aqui, em código, para ficarem
auditáveis. O pré-processamento é montado **dentro** do `Pipeline`, então
imputação, padronização e codificação são ajustadas só no fold de treino.

As variáveis se organizam em duas naturezas, e a distinção é o eixo do projeto:

* **HISTÓRICO** — como a rede vinha indo. Defasado, logo legítimo.
* **ATUALIDADE** — como o município é hoje: contexto socioeconômico, corpo
  docente, turmas, ruralidade, porte e UF.
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ── Histórico do próprio município ───────────────────────────────────────────
# Anteriores ao ciclo previsto: em 2024 já estavam publicados. Vazamento é
# contemporâneo, não histórico.
HISTORICO = ["taxa_2023", "media_portugues_2023", "ideb_2021", "taxa_aprovacao_2021"]

# ── Blocos composicionais: uma categoria de referência descartada ────────────
# AFD, IED e os níveis do INSE somam 100% por construção. A última categoria é
# `100 - (soma das demais)` — colinearidade perfeita. Cada coeficiente passa a
# ser lido como "efeito de deslocar um ponto percentual da referência para esta".
REFERENCIAS = {
    "afd_ai_grupo_5": "docentes sem curso superior",
    "ied_ai_nivel_1": "menor nível de esforço docente",
    "inse_pc_nivel_1": "nível socioeconômico mais baixo",
}

AFD = [f"afd_ai_grupo_{i}" for i in (1, 2, 3, 4)]
IED = [f"ied_ai_nivel_{i}" for i in (2, 3, 4, 5, 6)]
INSE_NIVEIS = [f"inse_pc_nivel_{i}" for i in range(2, 9)]
ATU = ["atu_creche", "atu_pre_escola", "atu_anos_iniciais", "atu_1_ano", "atu_2_ano"]
INSE_OUTRAS = ["media_inse", "qtd_alunos_inse", "proporcao_rural"]

ATUALIDADE = AFD + IED + INSE_NIVEIS + ATU + INSE_OUTRAS

# `sigla_uf` é o controle geográfico obrigatório (paradoxo de Simpson, EDA §4).
# `regiao` fica fora: é função determinística da UF, já contida nas dummies.
CATEGORICAS = ["sigla_uf", "capital_desc"]

# Ausência informativa: IDEB e taxa de aprovação faltam em 13,3% e 5,0% dos
# municípios, concentrados nos pequenos e isolados — daí `add_indicator=True`.
COM_INDICADOR = ["ideb_2021", "taxa_aprovacao_2021"]

EXCLUIDAS = {
    "id_municipio": "identificador; serve para separar municípios na validação",
    "taxa_alfabetizacao": "é o alvo (do ciclo de 2024)",
    "regiao": "função determinística de sigla_uf — as dummies já a codificam",
    "ano": "constante: a modelagem usa um único ciclo",
    "nota_portugues_2021": "correlaciona 0,956 com ideb_2021 — redundante",
    "nota_matematica_2021": "correlaciona 0,961 com ideb_2021 — redundante",
    "indicador_rendimento_2021": "correlaciona 0,996 com taxa_aprovacao_2021 — redundante",
    **{c: f"categoria de referência do bloco composicional ({d})"
       for c, d in REFERENCIAS.items()},
}


def colunas_do_modelo(numericas=None):
    """Colunas que alimentam a pipeline, na ordem."""
    return (HISTORICO + ATUALIDADE if numericas is None else list(numericas)) + CATEGORICAS


def construir_preprocessamento(numericas=None) -> ColumnTransformer:
    """Pré-processamento isolado, para reuso entre modelos candidatos.

    `numericas` permite treinar sobre um subconjunto — é assim que o notebook
    compara "só histórico" contra "só atualidade" pela mesma pipeline.
    """
    numericas = HISTORICO + ATUALIDADE if numericas is None else list(numericas)
    simples = [c for c in numericas if c not in COM_INDICADOR]
    com_indicador = [c for c in numericas if c in COM_INDICADOR]

    def ramo(indicador):
        return Pipeline([
            ("imputacao", SimpleImputer(strategy="median", add_indicator=indicador)),
            ("padronizacao", StandardScaler()),
        ])

    categoricas = Pipeline([
        ("imputacao", SimpleImputer(strategy="most_frequent")),
        ("codificacao", OneHotEncoder(drop="first", handle_unknown="ignore",
                                      sparse_output=False)),
    ])

    ramos = [("numericas", ramo(False), simples)]
    if com_indicador:
        ramos.append(("ausencia_informativa", ramo(True), com_indicador))
    ramos.append(("categoricas", categoricas, CATEGORICAS))

    return ColumnTransformer(ramos, remainder="drop", verbose_feature_names_out=False)


def construir_pipeline(C: float = 1.0, max_iter: int = 2000, random_state: int = 42,
                       estimador=None, numericas=None) -> Pipeline:
    """`Pipeline` completo: pré-processamento + modelo.

    Sem `estimador`, usa a Regressão Logística com L2 — o modelo da entrega. O
    parâmetro existe para que a comparação de algoritmos rode todos os
    candidatos pelo mesmo pré-processamento; trocar só o estimador é o que
    torna a comparação justa.
    """
    if estimador is None:
        estimador = LogisticRegression(C=C, solver="lbfgs", max_iter=max_iter,
                                       random_state=random_state)
    return Pipeline([("preprocessamento", construir_preprocessamento(numericas)),
                     ("modelo", estimador)])


def modelos_candidatos(random_state: int = 42) -> dict:
    """Catálogo dos algoritmos comparados no notebook de modelagem.

    Os quatro da aula de classificação supervisionada — Regressão Logística,
    Árvore de Decisão, SVM e Naive Bayes — mais os dois ensembles da aula de
    otimização (Random Forest e Gradient Boosting) e o baseline obrigatório.

    Todos passam pelo **mesmo** pré-processamento e pela mesma validação; só o
    estimador muda, e é isso que torna a comparação justa.
    """
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    return {
        "Baseline (classe majoritária)": DummyClassifier(strategy="most_frequent"),
        "Regressão Logística": LogisticRegression(
            C=1.0, solver="lbfgs", max_iter=2000, random_state=random_state),
        "Árvore de Decisão": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=30, random_state=random_state),
        "SVM (RBF)": SVC(kernel="rbf", C=1.0, probability=True,
                         random_state=random_state),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=10,
            n_jobs=-1, random_state=random_state),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.1,
            random_state=random_state),
    }


def nomes_das_features(pipeline: Pipeline):
    """Nomes das colunas na saída do pré-processamento, para ler os coeficientes."""
    return list(pipeline.named_steps["preprocessamento"].get_feature_names_out())
