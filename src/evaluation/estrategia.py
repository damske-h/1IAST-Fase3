"""Conversão das probabilidades do modelo em respostas de negócio.

A classe positiva do estimador é "atingiu a meta". Para política pública o que
interessa é o complemento, e ele é nomeado explicitamente aqui — `prob_risco` —
para que nenhuma tabela do projeto precise ser lida de trás para frente.

**Alinhamento.** `ranking_de_risco` ordena as linhas por risco. Por isso o rótulo
observado é anexado **antes** da ordenação e viaja junto com ela; as funções
seguintes leem a coluna `atingiu_observado` do próprio ranking, nunca um vetor
externo — que estaria na ordem original e produziria uma comparação silenciosamente
errada.
"""

import numpy as np
import pandas as pd

CONTEXTO = ["id_municipio", "ano", "sigla_uf", "regiao"]
COLUNA_OBSERVADA = "atingiu_observado"


def ranking_de_risco(modelo, X: pd.DataFrame, contexto: pd.DataFrame,
                     observado=None) -> pd.DataFrame:
    """Ordena municípios do maior para o menor risco de não atingir a meta.

    `observado` é opcional e deve estar na mesma ordem de `contexto`; ele é
    anexado antes da ordenação para não se desalinhar.
    """
    tabela = contexto[CONTEXTO].copy()
    tabela["prob_atingir_meta"] = modelo.predict_proba(X)[:, 1]
    tabela["prob_risco"] = 1 - tabela["prob_atingir_meta"]
    if observado is not None:
        tabela[COLUNA_OBSERVADA] = np.asarray(observado)
    return tabela.sort_values("prob_risco", ascending=False).reset_index(drop=True)


def _exigir_observado(ranking: pd.DataFrame) -> None:
    if COLUNA_OBSERVADA not in ranking.columns:
        raise ValueError(
            "o ranking precisa do rótulo observado: chame ranking_de_risco(..., "
            f"observado=y) para que a coluna '{COLUNA_OBSERVADA}' acompanhe a ordenação"
        )


def perfil_por_regiao(ranking: pd.DataFrame) -> pd.DataFrame:
    """Risco médio previsto por região, ao lado do resultado observado.

    Ver as duas colunas juntas é o que permite dizer onde o modelo acerta o
    patamar e onde ele apenas reproduz a geografia do ano de treino.
    """
    _exigir_observado(ranking)
    return (
        ranking.groupby("regiao", as_index=False)
        .agg(municipios=("id_municipio", "size"),
             risco_medio_previsto=("prob_risco", "mean"),
             risco_observado=(COLUNA_OBSERVADA, lambda s: 1 - s.mean()))
        .assign(diferenca=lambda d: d["risco_medio_previsto"] - d["risco_observado"])
        .sort_values("risco_medio_previsto", ascending=False)
        .reset_index(drop=True)
    )


def cobertura_por_limiar(ranking: pd.DataFrame,
                         limiares=(0.5, 0.7, 0.8, 0.9)) -> pd.DataFrame:
    """Quantos municípios cada corte sinaliza, e com que qualidade.

    Traduz o limiar em capacidade de atendimento: quantas equipes a secretaria
    consegue mobilizar define o corte, não o contrário.
    """
    _exigir_observado(ranking)
    em_risco_real = (1 - ranking[COLUNA_OBSERVADA]).to_numpy()

    linhas = []
    for limiar in limiares:
        sinalizado = (ranking["prob_risco"] >= limiar).to_numpy()
        acertos = int((sinalizado & (em_risco_real == 1)).sum())
        linhas.append({
            "limiar": limiar,
            "municipios_sinalizados": int(sinalizado.sum()),
            "cobertura_do_risco": round(acertos / max(int(em_risco_real.sum()), 1), 4),
            "precisao_do_alerta": round(acertos / max(int(sinalizado.sum()), 1), 4),
        })
    return pd.DataFrame(linhas)
