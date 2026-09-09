"""Conversão das probabilidades do modelo em respostas de negócio."""

import pandas as pd


def ranking_de_risco(modelo, X: pd.DataFrame, contexto: pd.DataFrame) -> pd.DataFrame:
    tabela = contexto[["id_municipio", "ano", "sigla_uf", "regiao"]].copy()
    tabela["prob_atingir_meta"] = modelo.predict_proba(X)[:, 1]
    tabela["prob_risco"] = 1 - tabela["prob_atingir_meta"]
    return tabela.sort_values("prob_risco", ascending=False).reset_index(drop=True)


def importancia_nativa(modelo) -> pd.DataFrame:
    estimador = modelo.named_steps["modelo"]
    nomes = modelo.named_steps["preprocessamento"].get_feature_names_out()
    if hasattr(estimador, "feature_importances_"):
        valores = estimador.feature_importances_
    elif hasattr(estimador, "coef_"):
        valores = abs(estimador.coef_[0])
    else:
        return pd.DataFrame(columns=["variavel", "importancia"])
    return (
        pd.DataFrame({"variavel": nomes, "importancia": valores})
        .sort_values("importancia", ascending=False).reset_index(drop=True)
    )


def perfil_por_regiao(ranking: pd.DataFrame) -> pd.DataFrame:
    return (
        ranking.groupby("regiao", as_index=False)
        .agg(municipios=("id_municipio", "size"), risco_medio=("prob_risco", "mean"))
        .sort_values("risco_medio", ascending=False)
    )
