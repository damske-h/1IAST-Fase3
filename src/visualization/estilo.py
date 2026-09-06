"""
Estilo visual compartilhado pelos notebooks.

Centraliza paleta, cromo dos eixos e formatação numérica para que os gráficos
do projeto formem um sistema coerente, em vez de cada notebook inventar o seu.

**Paleta.** Os oito slots categóricos são usados em **ordem fixa** — a cor segue
a entidade, nunca o ranking, de modo que filtrar uma série não repinta as
demais. Para magnitude usamos uma escala sequencial de **um único matiz**; para
polaridade (positivo/negativo), um par divergente quente/frio com cinza neutro
no meio. Cores de status são reservadas e nunca viram "série 4".

**Cromo recessivo.** Grade e eixos em *hairline* sólido, uma tonalidade acima da
superfície; sem molduras supérfluas; rótulo direto apenas nos extremos que
importam, nunca um número em cada barra.
"""

import matplotlib.pyplot as plt

# =============================================================================
# PALETA
# =============================================================================

# Slots categóricos, em ordem fixa.
SERIE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
         "#008300", "#4a3aa7", "#e34948"]

# Par divergente: polos que leem como opostos (frio/quente).
POSITIVO = "#2a78d6"
NEGATIVO = "#e34948"

# Tintas e cromo.
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
MUDO = "#898781"
GRADE = "#e1e0d9"
EIXO = "#c3c2b7"
SUPERFICIE = "#fcfcfb"


def aplicar_estilo() -> None:
    """Configura o matplotlib com o cromo do projeto."""
    plt.rcParams.update({
        "figure.facecolor": SUPERFICIE, "axes.facecolor": SUPERFICIE,
        "savefig.facecolor": SUPERFICIE,
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 9.5,
        "axes.edgecolor": EIXO, "axes.labelcolor": TINTA_2,
        "xtick.color": MUDO, "ytick.color": MUDO,
        "axes.grid": True, "grid.color": GRADE, "grid.linewidth": 0.8,
        "axes.axisbelow": True, "legend.frameon": False,
    })


def limpar(eixo, grade: str = "y"):
    """Remove molduras supérfluas e deixa a grade só no eixo que ajuda a ler."""
    eixo.spines[["top", "right"]].set_visible(False)
    eixo.grid(axis=grade, color=GRADE, linewidth=0.8)
    eixo.grid(axis="x" if grade == "y" else "y", visible=False)
    return eixo


def num(valor, casas: int = 2) -> str:
    """Formata número no padrão brasileiro (vírgula decimal)."""
    return f"{valor:.{casas}f}".replace(".", ",")


def salvar(figura, nome: str, diretorio) -> None:
    """Grava a figura em `diretorio/<nome>.png`."""
    figura.savefig(diretorio / f"{nome}.png", dpi=150,
                   bbox_inches="tight", facecolor=SUPERFICIE)
