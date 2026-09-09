"""Pipeline Scikit-learn com as mesmas variáveis do projeto de referência."""

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERICAS = ["media_portugues", "meta_2025", "meta_uf_2025", "taxa_media_uf_loo"]
CATEGORICAS = ["sigla_uf", "regiao"]
FEATURES = NUMERICAS + CATEGORICAS

EXCLUIDAS_POR_VAZAMENTO = [
    "taxa_alfabetizacao", "gap_meta_2025", "status_meta_2025",
    "nivel_alfabetizacao", *[f"proporcao_aluno_nivel_{i}" for i in range(9)],
]


def construir_preprocessamento() -> ColumnTransformer:
    numericas = Pipeline([
        ("imputacao", SimpleImputer(strategy="median")),
        ("padronizacao", StandardScaler()),
    ])
    categoricas = Pipeline([
        ("imputacao", SimpleImputer(strategy="most_frequent")),
        ("codificacao", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numericas", numericas, NUMERICAS),
        ("categoricas", categoricas, CATEGORICAS),
    ], verbose_feature_names_out=False)


def construir_pipeline(estimador=None) -> Pipeline:
    if estimador is None:
        estimador = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)
    return Pipeline([
        ("preprocessamento", construir_preprocessamento()),
        ("modelo", estimador),
    ])


def modelos_candidatos(com_baseline: bool = False) -> dict:
    """Um modelo linear e dois ensembles, opcionalmente com o baseline.

    O baseline prevê sempre a classe majoritária: ele não é um concorrente, é a
    régua que torna a acurácia legível. Fica fora por padrão para não poluir a
    comparação entre algoritmos de fato.
    """
    candidatos = {
        "Regressão Logística": LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250, max_depth=8, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }
    if com_baseline:
        candidatos["Baseline (classe majoritária)"] = DummyClassifier(
            strategy="most_frequent"
        )
    return candidatos


def nomes_das_features(pipeline: Pipeline) -> list[str]:
    return list(pipeline.named_steps["preprocessamento"].get_feature_names_out())
