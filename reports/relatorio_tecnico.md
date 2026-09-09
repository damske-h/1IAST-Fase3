# Relatório técnico - classificação municipal de alfabetização

## 1. Delimitação do problema

O Tech Challenge solicita um modelo supervisionado para prever alfabetização. Os dados da Fase 2 são agregados por município, ano e rede, sem observações individuais. O projeto, portanto, não treina um classificador de alunos.

A unidade de análise adotada é município-ano da rede municipal. O alvo binário é uma proxy:

- 1: taxa de alfabetização observada maior ou igual à meta municipal de 2025;
- 0: taxa observada abaixo da meta;
- registros sem meta conhecida são excluídos do dataset de modelagem.

Essa formulação reproduz o trabalho de referência, mas a interpretação correta é atingimento de meta municipal.

## 2. Reimplementação da Fase 2

### 2.1 Fontes

A solução usa somente os cinco arquivos originais:

| Fonte | Registros |
|---|---:|
| Indicador município | 23.995 |
| Indicador UF | 145 |
| Meta Brasil | 3 |
| Meta UF | 54 |
| Meta município | 10.704 |

Fontes externas foram removidas do fluxo porque o enunciado as apresenta como enriquecimento opcional.

### 2.2 Correspondência tecnológica

| Pipeline original | Reimplementação |
|---|---|
| AWS Glue / Spark | funções pandas |
| S3 Bronze, Silver e Gold | data/lake/bronze, silver e gold |
| Parquet particionado por ano | Parquet local com ano=YYYY |
| CloudWatch | sumários retornados como DataFrame e logs |
| Athena | leitura analítica com pandas |

### 2.3 Responsabilidade de cada camada

**Bronze:** tipagem explícita, hash, linhagem e persistência fiel.

**Silver:** padronização de rede, arredondamento, deduplicação defensiva e qualidade.

**Gold:** quatro visões analíticas da Fase 2. Nenhuma feature é excluída por leakage.

Essa fronteira é proposital: a engenharia de dados disponibiliza dados auditáveis; a EDA justifica a seleção de variáveis.

## 3. Saídas da Gold

| Visão | Registros | Colunas de negócio |
|---|---:|---:|
| alfabetizacao_por_municipio | 10.896 | 11 |
| evolucao_temporal | 49 | 8 |
| ranking_municipios | 10.896 | 8 |
| comparacao_metas_nacionais | 1 | 9 |

A visão principal conserva taxa, nota de Português, metas, nível, gap e status. Ela é apropriada para análise, mas não pode ser entregue diretamente ao estimador.

## 4. Análise exploratória e leakage

O alvo é calculado comparando taxa_alfabetizacao e meta_2025. Assim:

| Coluna | Decisão |
|---|---|
| taxa_alfabetizacao | excluída: define o alvo |
| gap_meta_2025 | excluída: função direta da taxa e meta |
| status_meta_2025 | excluída: origem do rótulo |
| nivel_alfabetizacao | excluída: faixa da taxa |
| proporcao_aluno_nivel_0 a 8 | excluídas: decomposição da avaliação |
| id_municipio e ano | identificação/split, não entram em X |

A variável taxa_media_uf_loo é calculada pela média dos pares da mesma UF e ano, sem incluir o município da linha.

A seleção acontece em src/preprocessing/build_features.py depois de demonstrada no notebook 02. O resultado tem 10.654 registros e nove colunas.

## 5. Variáveis finais

| Tipo | Variáveis |
|---|---|
| Numéricas | media_portugues, meta_2025, meta_uf_2025, taxa_media_uf_loo |
| Categóricas | sigla_uf, regiao |
| Identificadores | id_municipio, ano |
| Alvo | alvo_alfabetizado |

As seis features são exatamente as usadas pelo projeto de referência.

## 6. Pré-processamento e validação

O ColumnTransformer executa:

- SimpleImputer com mediana e StandardScaler nas numéricas;
- SimpleImputer com moda e OneHotEncoder nas categóricas.

Todas as transformações estão dentro do Pipeline, portanto são ajustadas separadamente em cada fold.

Estratégia:

- treino temporal: 2023, 5.302 registros;
- holdout: 2024, 5.352 registros;
- tuning: StratifiedKFold com 5 folds apenas no treino;
- seleção por F1 no treino;
- holdout usado para avaliação final.

Foram comparados Regressão Logística, Gradient Boosting e Random Forest.

## 7. Resultados

| Modelo | Acurácia | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|
| Regressão Logística | 0,717 | 0,584 | 0,819 | 0,788 |
| Gradient Boosting | 0,650 | 0,441 | 0,815 | 0,734 |
| Random Forest | 0,586 | 0,322 | 0,710 | 0,613 |

A Regressão Logística, com C=1,0, foi escolhida. No holdout:

- precisão: 0,808;
- recall: 0,457;
- F1: 0,584;
- ROC-AUC: 0,819;
- PR-AUC: 0,788.

O modelo ordena os municípios razoavelmente bem, mas o recall baixo no limiar padrão impede uso automático. Para política pública, o limiar deve refletir orçamento e custo dos falsos negativos.

## 8. Aplicação estratégica

### Fatores associados

Os coeficientes padronizados permitem ordenar influência dentro do modelo. Eles não estimam impacto causal. Dummies territoriais refletem contexto institucional, histórico e composição não observada.

### Municípios em maior risco

A probabilidade operacional é:

    prob_risco = 1 - prob_atingir_meta

O arquivo reports/ranking_risco_2024.csv contém o ranking completo do holdout.

### Padrões regionais

O risco é agregado por região para facilitar comparação. Similaridade de média regional não implica que os municípios tenham causas iguais.

### Metas futuras

A base atual não sustenta previsão ex ante. media_portugues e taxa_media_uf_loo são medidas do próprio ciclo. O ranking deve ser descrito como classificação ou nowcast. Uma previsão futura exigiria:

1. features defasadas;
2. pelo menos três ciclos comparáveis;
3. validação temporal em ciclos nunca usados;
4. idealmente, variáveis contextuais disponíveis antes da avaliação.

## 9. Limitações

1. A proxy municipal não equivale a um rótulo individual.
2. Não há nomes de municípios, somente códigos IBGE.
3. A prevalência da classe muda de 18,1% em 2023 para 43,4% em 2024.
4. As features contemporâneas limitam o uso prospectivo.
5. Duas variáveis territoriais relacionadas podem introduzir redundância.
6. A ausência de fontes externas reduz capacidade explicativa socioeconômica.
7. Importância preditiva não autoriza recomendações causais.

## 10. Conclusão

A simplificação atende ao núcleo acadêmico do desafio: reproduz a Gold da Fase 2, realiza EDA, trata leakage na etapa correta, integra pré-processamento e modelo, compara algoritmos, valida temporalmente e gera um ranking aplicável à triagem.

O principal ganho não foi maximizar métricas, mas tornar o encadeamento auditável:

    cinco CSVs -> Bronze -> Silver -> quatro Golds -> EDA/leakage
    -> dataset de 9 colunas -> Pipeline Scikit-learn -> ranking municipal
