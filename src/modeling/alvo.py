"""
Construção do alvo supervisionado a partir de dados agregados.

O enunciado pede prever se **um aluno** será alfabetizado. A base disponível,
porém, é agregada por município: cada linha traz a *taxa* de alfabetização,
não o desfecho de cada criança. Não existem microdados de aluno publicados.

A ponte entre os dois grãos é o **dado binário agrupado** (*grouped binary
data*), que é exatamente o formato que a regressão logística binomial estima:
cada linha município × ano vira **duas observações** com o mesmo vetor de
características,

    y = 1  com peso  taxa / 100          (a fração de alunos alfabetizados)
    y = 0  com peso  1 - taxa / 100      (a fração não alfabetizada)

treinadas com `sample_weight`. A verossimilhança resultante é idêntica à de
uma logística ajustada sobre os alunos individuais daquele município, e a
probabilidade prevista tem a leitura direta: *"probabilidade de um aluno
daquele município estar alfabetizado"*.

**Limitação a declarar (falácia ecológica).** O modelo estima a probabilidade
média do aluno *dado o município*. Ele não observa nenhuma característica
individual da criança, e portanto não pode ser usado para prever o desfecho de
um aluno específico — apenas o de um aluno típico daquele contexto municipal.

**Peso por município, não por aluno.** Cada município contribui com peso total
1, independentemente de quantos alunos tenha. É a escolha coerente com o uso
pretendido (priorizar municípios para política pública) e com o dado
disponível — a base não traz o número de alunos avaliados. A alternativa
(ponderar pelo porte) deslocaria as estimativas para os grandes municípios.
"""

import numpy as np
import pandas as pd

COLUNA_TAXA = "taxa_alfabetizacao"
COLUNA_GRUPO = "id_municipio"


def expandir_binomial(base: pd.DataFrame,
                      coluna_taxa: str = COLUNA_TAXA,
                      coluna_grupo: str = COLUNA_GRUPO):
    """Expande a base agregada em observações binárias ponderadas.

    Devolve `(X, y, peso, grupos)`, onde `X` repete cada linha duas vezes,
    `y` alterna 1 e 0, `peso` traz a fração correspondente e `grupos` carrega
    o `id_municipio` de cada observação — insumo do `GroupKFold`, que impede
    o mesmo município de aparecer em treino e validação.
    """
    valida = base[base[coluna_taxa].notna()].reset_index(drop=True)
    if len(valida) < len(base):
        raise ValueError(
            f"{len(base) - len(valida)} linha(s) sem taxa de alfabetização — "
            "o alvo deve estar completo antes da expansão."
        )

    X = valida.loc[valida.index.repeat(2)].reset_index(drop=True)
    y = np.tile([1, 0], len(valida))

    taxa = X[coluna_taxa].to_numpy() / 100.0
    peso = np.where(y == 1, taxa, 1.0 - taxa)
    grupos = X[coluna_grupo].to_numpy()

    return X, y, peso, grupos


def resumir_expansao(base: pd.DataFrame, y, peso) -> pd.DataFrame:
    """Sumário de conferência da expansão, para exibir no notebook."""
    return pd.DataFrame([
        {"verificação": "linhas na base agregada", "valor": len(base)},
        {"verificação": "observações após a expansão", "valor": len(y)},
        {"verificação": "soma dos pesos (= linhas da base)", "valor": round(float(peso.sum()), 2)},
        {"verificação": "peso total em y=1 (alunos alfabetizados)",
         "valor": round(float(peso[y == 1].sum()), 2)},
        {"verificação": "taxa média implícita (%)",
         "valor": round(float(peso[y == 1].sum() / peso.sum() * 100), 2)},
        {"verificação": "observações com peso zero (taxa 0% ou 100%)",
         "valor": int((peso == 0).sum())},
    ])
