"""
Tradução do classificador em instrumentos de decisão.

`metricas.py` responde "o modelo é bom?"; este responde "o que se faz com ele?".
Tudo sai do **mesmo modelo supervisionado** — não há um segundo modelo por trás
de nenhuma resposta.

| Instrumento | Pergunta de negócio |
|---|---|
| `efeitos_marginais` | quais fatores mais impactam a alfabetização |
| `ranking_de_risco` | quais municípios apresentam maior risco |
| `perfil_por_regiao` | quais regiões possuem padrões semelhantes |
| `situacao_frente_a_meta` | quem pode não atingir a meta pactuada |
"""

import numpy as np
import pandas as pd


def ranking_de_risco(modelo, painel: pd.DataFrame,
                     colunas_extras=("sigla_uf", "regiao", "taxa_alfabetizacao")) -> pd.DataFrame:
    """Probabilidade de risco por município, da maior para a menor.

    É a resposta direta a "quais municípios apresentam maior risco educacional":
    a saída nativa do classificador, sem construção intermediária.
    """
    colunas = ["id_municipio", *[c for c in colunas_extras if c in painel.columns]]
    tabela = painel[colunas].copy()
    tabela["prob_risco"] = modelo.predict_proba(painel)[:, 1]
    return tabela.sort_values("prob_risco", ascending=False).reset_index(drop=True)


def efeitos_marginais(modelo, painel: pd.DataFrame, colunas) -> pd.DataFrame:
    """Efeito de somar um desvio-padrão a cada variável, em pontos percentuais
    de **probabilidade de risco**.

    Diferente do coeficiente, que vive em log-odds, este número responde à
    pergunta do gestor: *"se eu melhorar isto, quanto cai o risco?"*. Continua
    sendo leitura **associativa**, não causal.
    """
    referencia = modelo.predict_proba(painel)[:, 1]

    linhas = []
    for coluna in colunas:
        alterado = painel.copy()
        alterado[coluna] = alterado[coluna] + alterado[coluna].std()
        efeito = (modelo.predict_proba(alterado)[:, 1] - referencia).mean() * 100
        linhas.append({"variavel": coluna, "efeito_pp_no_risco": efeito})

    return (pd.DataFrame(linhas).set_index("variavel")
            .sort_values("efeito_pp_no_risco", key=np.abs, ascending=False))


def perfil_por_regiao(modelo, painel: pd.DataFrame, valores_shap: pd.DataFrame,
                      n_fatores: int = 3) -> pd.DataFrame:
    """Nível de risco e **motores do risco** em cada região.

    Duas regiões "possuem padrões semelhantes" quando têm risco parecido *e* o
    risco é empurrado pelos mesmos fatores. Ficam fora as dummies de UF, que
    identificam o território em vez de explicá-lo, e os indicadores de ausência,
    que sinalizam falta de dado e não um fator do município.
    """
    fatores = [c for c in valores_shap.columns
               if not c.startswith(("sigla_uf", "missingindicator_"))]
    contribuicoes = valores_shap[fatores].copy()
    contribuicoes["regiao"] = painel["regiao"].to_numpy()
    media = contribuicoes.groupby("regiao").mean()

    risco = pd.Series(modelo.predict_proba(painel)[:, 1] * 100, index=painel.index)

    linhas = []
    for regiao, contribuicao in media.iterrows():
        recorte = painel["regiao"] == regiao
        linhas.append({
            "regiao": regiao,
            "municipios": int(recorte.sum()),
            "risco_medio_pct": round(float(risco[recorte].mean()), 1),
            "motores_do_risco": " · ".join(contribuicao.nlargest(n_fatores).index),
        })

    return (pd.DataFrame(linhas)
            .sort_values("risco_medio_pct", ascending=False)
            .reset_index(drop=True))


def situacao_frente_a_meta(ranking: pd.DataFrame, metas: pd.DataFrame,
                           limiar_risco: float = 0.5) -> pd.DataFrame:
    """Cruza a probabilidade de risco com a meta pactuada de cada município.

    A meta **nunca entrou no modelo** — ela deriva da taxa de 2023 do próprio
    município, o que a torna vazamento como preditor. Entra aqui só como régua
    de comparação, depois da predição.
    """
    juncao = ranking.merge(metas, on="id_municipio", how="left").dropna(subset=["meta_2025"])
    juncao["atingiu_meta"] = juncao["taxa_alfabetizacao"] >= juncao["meta_2025"]
    juncao["sinalizado_em_risco"] = juncao["prob_risco"] >= limiar_risco
    juncao["distancia_pp"] = juncao["taxa_alfabetizacao"] - juncao["meta_2025"]

    juncao["situacao"] = np.where(
        juncao["atingiu_meta"], "1. Meta já atingida",
        np.where(~juncao["sinalizado_em_risco"],
                 "2. Abaixo da meta, sem sinal de risco",
                 "3. Abaixo da meta e sinalizado em risco"))
    return juncao
