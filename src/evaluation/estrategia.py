"""
Tradução do classificador em instrumentos de decisão.

Enquanto `metricas.py` responde "o modelo é bom?", este módulo responde "o que
se faz com ele?". Tudo aqui sai do **próprio modelo supervisionado** — não há
um segundo modelo por trás de nenhuma das respostas.

| Instrumento | Pergunta de negócio que atende |
|---|---|
| `efeitos_marginais` | quais fatores mais impactam a alfabetização |
| `ranking_de_risco` | quais municípios apresentam maior risco educacional |
| `perfil_por_regiao` | quais regiões possuem padrões semelhantes |
| `situacao_frente_a_meta` | quem pode não atingir a meta pactuada |
"""

import numpy as np
import pandas as pd

SEMENTE = 42


# =============================================================================
# PROBABILIDADE DE RISCO POR MUNICÍPIO
# =============================================================================

def ranking_de_risco(modelo, X: pd.DataFrame,
                     colunas_extras=("sigla_uf", "regiao", "taxa_alfabetizacao")) -> pd.DataFrame:
    """Probabilidade de risco por município, ordenada da maior para a menor.

    É a resposta direta a "quais municípios apresentam maior risco educacional":
    a saída nativa do classificador, sem nenhuma construção intermediária.
    """
    colunas = ["id_municipio", *[c for c in colunas_extras if c in X.columns]]
    tabela = X[colunas].copy()
    tabela["prob_risco"] = modelo.predict_proba(X)[:, 1]
    return tabela.sort_values("prob_risco", ascending=False).reset_index(drop=True)


# =============================================================================
# EFEITOS MARGINAIS
# =============================================================================

def efeitos_marginais(modelo, X: pd.DataFrame, colunas) -> pd.DataFrame:
    """Efeito de somar um desvio-padrão a cada variável, em pontos percentuais
    de **probabilidade de risco**.

    Diferente do coeficiente, que vive em log-odds, este número responde à
    pergunta do gestor: *"se eu melhorar isto, quanto cai o risco?"*. Continua
    sendo leitura **associativa**, não causal.
    """
    referencia = modelo.predict_proba(X)[:, 1]

    linhas = []
    for coluna in colunas:
        alterada = X.copy()
        alterada[coluna] = alterada[coluna] + alterada[coluna].std()
        efeito = (modelo.predict_proba(alterada)[:, 1] - referencia).mean() * 100
        linhas.append({"variavel": coluna, "efeito_pp_no_risco": efeito})

    return (pd.DataFrame(linhas)
            .set_index("variavel")
            .sort_values("efeito_pp_no_risco", key=np.abs, ascending=False))


# =============================================================================
# PADRÕES REGIONAIS — SEM MODELO ADICIONAL
# =============================================================================

def perfil_por_regiao(modelo, X: pd.DataFrame, valores_shap: pd.DataFrame,
                      coluna_regiao: str = "regiao", n_fatores: int = 3) -> pd.DataFrame:
    """Nível de risco e **motores do risco** em cada região.

    Duas regiões "possuem padrões semelhantes" quando têm risco parecido *e* o
    risco é empurrado pelos mesmos fatores. Ambas as leituras saem do modelo
    supervisionado: a primeira da probabilidade prevista, a segunda da
    contribuição média (SHAP) de cada variável naquela região.

    `valores_shap` deve vir com as mesmas linhas de `X` e uma coluna por
    feature já transformada.
    """
    contribuicoes = valores_shap.copy()
    contribuicoes[coluna_regiao] = X[coluna_regiao].to_numpy()

    # As dummies de UF ficam de fora: elas identificam o território em vez de
    # explicá-lo, e tornariam a resposta circular.
    fatores = [c for c in valores_shap.columns if not c.startswith("sigla_uf")]
    media_por_regiao = contribuicoes.groupby(coluna_regiao)[fatores].mean()

    probabilidade = pd.Series(modelo.predict_proba(X)[:, 1], index=X.index)

    linhas = []
    for regiao, contribuicao in media_por_regiao.iterrows():
        recorte = X[coluna_regiao] == regiao
        principais = contribuicao.sort_values(ascending=False).head(n_fatores)
        linhas.append({
            "regiao": regiao,
            "municipios": int(recorte.sum()),
            "risco_medio_previsto": round(float(probabilidade[recorte].mean()), 3),
            "motores_do_risco": " · ".join(principais.index),
        })

    return (pd.DataFrame(linhas)
            .sort_values("risco_medio_previsto", ascending=False)
            .reset_index(drop=True))


# =============================================================================
# SITUAÇÃO FRENTE À META
# =============================================================================

def situacao_frente_a_meta(ranking: pd.DataFrame, metas: pd.DataFrame,
                           limiar_risco: float = 0.5) -> pd.DataFrame:
    """Cruza a probabilidade de risco com a meta pactuada de cada município.

    A meta **nunca entrou no modelo** — ela é derivada da taxa de 2023 do
    próprio município, o que a torna vazamento como preditor. Aqui ela entra só
    como régua de comparação, depois da predição.
    """
    juncao = ranking.merge(metas, on="id_municipio", how="left").dropna(subset=["meta_2025"])
    juncao["atingiu_meta"] = juncao["taxa_alfabetizacao"] >= juncao["meta_2025"]
    juncao["sinalizado_em_risco"] = juncao["prob_risco"] >= limiar_risco

    def classificar(linha):
        if linha["atingiu_meta"]:
            return "1. Meta já atingida"
        if not linha["sinalizado_em_risco"]:
            return "2. Abaixo da meta, sem sinal de risco estrutural"
        return "3. Abaixo da meta e sinalizado em risco"

    juncao["situacao"] = juncao.apply(classificar, axis=1)
    return juncao
