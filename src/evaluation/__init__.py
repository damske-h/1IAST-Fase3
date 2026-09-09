"""Avaliação: métricas, interpretabilidade e instrumentos de decisão."""

from .estrategia import cobertura_por_limiar, perfil_por_regiao, ranking_de_risco
from .interpretabilidade import (
    importancia_nativa, importancia_por_permutacao, importancia_shap,
    nomes_transformados, valores_shap,
)
from .metricas import (
    avaliar, comparar_por_validacao_cruzada, curva_calibracao,
    desvio_de_calibracao, limiar_por_recall, matriz_confusao, particionador,
)

__all__ = [
    # métricas
    "avaliar", "matriz_confusao", "particionador", "curva_calibracao",
    "desvio_de_calibracao", "limiar_por_recall", "comparar_por_validacao_cruzada",
    # interpretabilidade
    "importancia_nativa", "importancia_por_permutacao", "importancia_shap",
    "valores_shap", "nomes_transformados",
    # estratégia
    "ranking_de_risco", "perfil_por_regiao", "cobertura_por_limiar",
]
