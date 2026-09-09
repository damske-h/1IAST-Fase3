# Tech Challenge - Fase 3

## Predição e inteligência analítica para alfabetização no Brasil

Projeto acadêmico de Ciência de Dados que reproduz, em Python e pandas, a pipeline medalhão construída na Fase 2 e desenvolve um modelo supervisionado para classificar o risco educacional dos municípios brasileiros.

A solução foi deliberadamente simplificada para usar somente os cinco arquivos originais da Fase 2. As fontes externas são opcionais no enunciado e não participam desta versão.

**Recorte da entrega:** analisamos fatores por **município**, devido às limitações dos arquivos disponíveis. Eles não contêm ID de aluno, características individuais ou histórico que permita acompanhar uma criança. O código IBGE identifica o município. O resultado estima o atingimento de uma meta municipal e não pode ser convertido na chance de alfabetização de um aluno daquele lugar.

## Sumário

1. [Contexto do problema](#1-contexto-do-problema)
2. [Objetivo analítico](#2-objetivo-analítico)
3. [Descrição da base utilizada](#3-descrição-da-base-utilizada)
4. [Etapas de modelagem](#4-etapas-de-modelagem)
5. [Escolha do algoritmo](#5-escolha-do-algoritmo)
6. [Métricas de avaliação](#6-métricas-de-avaliação)
7. [Interpretação dos resultados](#7-interpretação-dos-resultados)
8. [Insights encontrados](#8-insights-encontrados)
9. [Limitações do projeto](#9-limitações-do-projeto)
10. [Aplicação prática para políticas públicas](#10-aplicação-prática-para-políticas-públicas)
11. [Possíveis evoluções futuras](#11-possíveis-evoluções-futuras)
12. [Perguntas norteadoras](#12-respostas-às-perguntas-norteadoras)
13. [Reprodução](#13-como-reproduzir)

---

## 1. Contexto do problema

A alfabetização infantil é um indicador central do desenvolvimento educacional e social. Conhecer somente a situação corrente não é suficiente para orientar recursos públicos: gestores precisam identificar territórios vulneráveis, compreender padrões regionais e priorizar municípios que demandam acompanhamento.

O Tech Challenge propõe transformar os dados produzidos pela pipeline da Fase 2 em análise exploratória, modelagem supervisionada e inteligência aplicável à tomada de decisão.

Há uma restrição essencial: os arquivos disponíveis não contêm microdados de alunos. A menor granularidade é **município × ano × rede de ensino**. Consequentemente, este trabalho não prevê se uma criança específica será alfabetizada.

---

## 2. Objetivo analítico

O objetivo técnico é estimar a probabilidade de uma rede municipal atingir sua meta de alfabetização, usando variáveis educacionais e territoriais existentes nos dados da Fase 2.

### Unidade de análise

- Uma observação representa um município, em um ano, na rede municipal.
- Treino: ciclo de 2023.
- Teste temporal: ciclo de 2024.

### Variável-alvo

A variável **alvo_alfabetizado** é uma proxy municipal:

- **1 - atingiu a meta:** taxa municipal observada maior ou igual à meta municipal de 2025;
- **0 - não atingiu a meta:** taxa observada abaixo da meta;
- municípios sem meta publicada são excluídos da modelagem.

A interpretação correta é **probabilidade de atingimento da meta municipal**. Não se trata de probabilidade individual de alfabetização.

### Saída operacional

Para priorização pública, o risco é calculado por:

    probabilidade de risco = 1 - probabilidade de atingir a meta

O resultado final é um ranking municipal para triagem, disponível em [reports/ranking_risco_2024.csv](reports/ranking_risco_2024.csv).

---

## 3. Descrição da base utilizada

### 3.1 Fontes originais

Os dados têm origem no INEP e foram obtidos na plataforma **Base dos Dados**, conforme a documentação da Fase 2. Os cinco CSVs foram versionados em `dados/` naquele projeto e reaproveitados em `data/` nesta fase. A execução atual lê esses arquivos locais, reconstrói Bronze, Silver e Gold em Python/pandas e usa a Gold como origem da preparação analítica. Não depende de baixar dados ou consultar a AWS durante a execução.

Foram utilizados somente os cinco CSVs herdados da Fase 2:

| Entidade | Registros brutos | Conteúdo |
|---|---:|---|
| Indicador por município | 23.995 | taxa, média de Português e níveis de proficiência |
| Indicador por UF | 145 | indicadores agregados estaduais |
| Metas Brasil | 3 | metas nacionais |
| Metas por UF | 54 | metas estaduais |
| Metas por município | 10.704 | metas e níveis municipais |

As planilhas mantidas em data/external não são lidas por nenhum módulo da pipeline atual.

**Não usamos fontes externas de enriquecimento.** A plataforma que disponibilizou os CSVs originais não representa uma fonte adicional de atributos. UF e região são derivadas do código municipal com um mapeamento no código. Não foram adicionados renda, investimento, infraestrutura, formação docente ou registros de políticas públicas.

### Contexto socioeconômico e político

A análise considera o território em que as crianças estudam para investigar diferenças de desempenho entre municípios. Condições socioeconômicas e políticas educacionais são hipóteses relevantes para explicar essas diferenças, mas **não foram medidas diretamente na base utilizada**. As variáveis UF e região podem refletir vários fatores não observados, sem identificar qual deles explica o resultado.

Portanto, não concluímos que uma política específica ou a renda determina se um aluno será alfabetizado. Usamos os indicadores educacionais e territoriais disponíveis para classificar a situação municipal e indicar onde aprofundar o diagnóstico. Atribuir o resultado agregado a uma criança seria uma **falácia ecológica**.

### 3.2 Reimplementação da arquitetura medalhão

| Aspecto | Fase 2 | Reimplementação local |
|---|---|---|
| Processamento | AWS Glue e Spark | Python e pandas |
| Armazenamento | Amazon S3 | Parquet local |
| Consulta | Athena | pandas |
| Particionamento | ano=YYYY | ano=YYYY |
| Camadas | Bronze, Silver e Gold | Bronze, Silver e Gold |
| Fontes | cinco CSVs | os mesmos cinco CSVs |

A pipeline gera:

| Visão Gold | Registros | Finalidade |
|---|---:|---|
| alfabetizacao_por_municipio | 10.896 | análise municipal e construção do alvo |
| evolucao_temporal | 49 | evolução por UF e ano |
| ranking_municipios | 10.896 | posição do município dentro da UF |
| comparacao_metas_nacionais | 1 | comparação disponível no recorte nacional |

As quatro visões e o dataset de modelagem foram comparados programaticamente com o projeto de referência e apresentaram conteúdo idêntico, desconsiderando metadados de execução e a ordem física da coluna de partição.

### 3.3 Dataset final de modelagem

A seleção ocorre somente depois do diagnóstico de leakage realizado na EDA. O arquivo [data/features/dataset_modelagem.parquet](data/features/dataset_modelagem.parquet) possui **10.654 registros e 9 colunas**:

| Papel | Colunas |
|---|---|
| Identificação | id_municipio, ano |
| Numéricas | media_portugues, meta_2025, meta_uf_2025, taxa_media_uf_loo |
| Categóricas | sigla_uf, regiao |
| Alvo | alvo_alfabetizado |

---

## 4. Etapas de modelagem

### 4.1 Análise exploratória

A EDA investiga distribuição, ausência de dados, evolução temporal, padrões regionais, correlações e risco de leakage.

![Distribuição da taxa de alfabetização e situação frente à meta](images/02_distribuicao_indicador_proxy.png)

A taxa de alfabetização ocupa praticamente toda a escala de 0% a 100%. A maior parte das observações município-ano não atingiu a meta; 242 registros não possuíam uma meta municipal válida e, portanto, não receberam alvo.

### 4.2 Tratamento de data leakage

A pipeline Bronze -> Silver -> Gold preserva todas as variáveis. A exclusão acontece apenas na etapa analítica, depois de justificada pela EDA.

Não entram como features:

| Coluna | Motivo |
|---|---|
| taxa_alfabetizacao | participa diretamente da definição do alvo |
| gap_meta_2025 | função da taxa e da meta |
| status_meta_2025 | origem direta do rótulo |
| nivel_alfabetizacao | discretização da taxa |
| proporcao_aluno_nivel_0 a 8 | decomposição da mesma avaliação |
| id_municipio | identificador, sem significado ordinal |
| ano | usado para o split temporal |

A variável **taxa_media_uf_loo** usa média leave-one-out: a taxa do próprio município é removida do cálculo estadual.

Existe uma limitação residual importante. **media_portugues** e **taxa_media_uf_loo** são informações contemporâneas ao ciclo avaliado. Elas não revelam diretamente o rótulo da linha, mas impedem interpretar o resultado como previsão ex ante. O uso metodologicamente defensável é classificação ou nowcast após a avaliação.

### 4.3 Engenharia e transformação das variáveis

O pré-processamento é integrado ao Pipeline do Scikit-learn:

- variáveis numéricas: imputação pela mediana e padronização;
- variáveis categóricas: imputação pela moda e one-hot encoding;
- categorias desconhecidas no teste: ignoradas pelo encoder;
- todas as transformações são ajustadas somente nos dados de treino de cada fold.

![Padrões territoriais e correlações das variáveis finais](images/02_padroes_territoriais_correlacoes.png)

A matriz mostra dois pontos relevantes:

- media_portugues possui a maior correlação linear com o alvo: 0,61;
- meta_uf_2025 e taxa_media_uf_loo têm correlação de 0,90, indicando redundância territorial.

### 4.4 Treino, validação e teste

- Treino temporal: 5.302 registros de 2023.
- Holdout temporal: 5.352 registros de 2024.
- Validação interna: StratifiedKFold com cinco folds, somente em 2023.
- Otimização: GridSearchCV sobre o parâmetro C da Regressão Logística.
- Semente aleatória: 42.
- O holdout de 2024 não participa da escolha de hiperparâmetros.

A comparação entre os três algoritmos, contudo, também usa 2024. Por isso, esse ciclo informa a justificativa da escolha do algoritmo e não constitui uma avaliação completamente independente de todas as decisões. Uma nova avaliação em outro ciclo é necessária para confirmar a generalização.

---

## 5. Escolha do algoritmo

Foram comparados três algoritmos sob o mesmo pré-processamento:

| Modelo | Acurácia | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|
| **Regressão Logística** | **0,717** | **0,584** | **0,819** | **0,788** |
| Gradient Boosting | 0,650 | 0,441 | 0,815 | 0,734 |
| Random Forest | 0,586 | 0,322 | 0,710 | 0,613 |

A Regressão Logística foi escolhida porque:

- obteve o maior ROC-AUC e PR-AUC;
- apresentou o melhor F1;
- permite interpretar direção e magnitude relativa dos coeficientes;
- é compatível com o objetivo acadêmico de demonstrar todo o fluxo;
- possui menor complexidade operacional.

O melhor hiperparâmetro encontrado foi **C = 1,0**.

---

## 6. Métricas de avaliação

### 6.1 Resultado no holdout de 2024

| Métrica | Valor | Interpretação |
|---|---:|---|
| Acurácia | 0,7173 | 71,7% das classes foram previstas corretamente |
| Precisão da classe atingiu | 0,8081 | quando prevê atingimento, acerta 80,8% |
| Recall da classe atingiu | 0,4569 | identifica somente 45,7% dos que atingiram |
| F1 da classe atingiu | 0,5838 | equilíbrio entre precisão e recall da classe 1 |
| ROC-AUC | 0,8191 | boa capacidade de ordenação entre as classes |
| PR-AUC | 0,7878 | qualidade da classe positiva considerando sua prevalência |

![Matriz de confusão, curva ROC e curva Precisão-Recall](images/03_avaliacao_modelo.png)

### 6.2 Leitura da matriz de confusão

| Resultado | Municípios |
|---|---:|
| Não atingiu e foi previsto como não atingiu | 2.778 |
| Não atingiu, mas foi previsto como atingiu | 252 |
| Atingiu, mas foi previsto como não atingiu | 1.261 |
| Atingiu e foi previsto como atingiu | 1.061 |

Embora as métricas publicadas usem **atingiu a meta** como classe positiva, a política pública se interessa principalmente pela classe de risco.

No limiar de 0,5:

- o modelo encontra **91,7% dos municípios que efetivamente não atingiram a meta**;
- sinaliza 4.039 municípios;
- aproximadamente 68,8% dos sinalizados realmente não atingiram a meta.

Isso favorece triagem com alta cobertura, mas produz muitos falsos alertas.

---

## 7. Interpretação dos resultados

### 7.1 Influência das variáveis

![Coeficientes da Regressão Logística](images/03_interpretabilidade_coeficientes.png)

A variável dominante é **media_portugues**, com coeficiente padronizado de 5,42. Em seguida aparecem diversas dummies de UF, mostrando que o território tem forte poder preditivo.

Essa leitura exige cautela:

- coeficiente não representa efeito causal;
- uma dummy de UF agrega fatores não observados, como políticas locais, estrutura da rede e composição social;
- região e UF são relacionadas, introduzindo redundância;
- a nota de Português é contemporânea à avaliação e não constitui instrumento de antecipação.

### 7.2 Mudança temporal

A proporção de municípios que atingiu a meta mudou fortemente:

| Ano | Municípios | Atingiu a meta |
|---|---:|---:|
| 2023 | 5.302 | 18,1% |
| 2024 | 5.352 | 43,4% |

Essa mudança de 25,3 pontos percentuais altera a distribuição do alvo entre treino e teste. Parte dos erros do modelo decorre desse drift temporal.

### 7.3 Risco regional previsto

![Risco médio previsto por região](images/04_risco_por_regiao.png)

| Região | Municípios no holdout | Risco médio previsto |
|---|---:|---:|
| Norte | 425 | 94,1% |
| Nordeste | 1.756 | 79,3% |
| Sudeste | 1.614 | 79,3% |
| Centro-Oeste | 464 | 63,7% |
| Sul | 1.093 | 54,6% |

O Sudeste aparece com risco previsto praticamente igual ao Nordeste, embora 57,1% dos municípios do Sudeste tenham atingido a meta em 2024, contra 38,2% no Nordeste. Essa divergência é evidência de **calibração temporal insuficiente**, e não um resultado a ser convertido diretamente em política.

---

## 8. Insights encontrados

### Insight 1 - A melhora nacional não foi homogênea

O atingimento da meta subiu de 18,1% para 43,4%, mas o ganho variou fortemente por região:

| Região | 2023 | 2024 |
|---|---:|---:|
| Centro-Oeste | 19,3% | 64,2% |
| Sudeste | 13,0% | 57,1% |
| Nordeste | 17,1% | 38,2% |
| Sul | 32,4% | 29,7% |
| Norte | 3,5% | 25,4% |

O Sul foi a única região com redução na proporção de municípios que atingiu a meta.

### Insight 2 - O território concentra informação não observada

As dummies estaduais aparecem entre as variáveis mais influentes. Isso sugere que características institucionais e históricas não presentes na base estão sendo capturadas indiretamente pelo território.

### Insight 3 - Português é a variável mais associada ao alvo

media_portugues domina correlações e coeficientes. O resultado é esperado porque a alfabetização é construída a partir da mesma avaliação educacional. Isso torna a variável útil para classificação contemporânea, mas fraca para antecipação.

### Insight 4 - Há redundância entre contexto estadual e meta estadual

meta_uf_2025 e taxa_media_uf_loo apresentam correlação de 0,90. Manter ambas reproduz o projeto de referência, mas uma evolução futura deve testar regularização, remoção de redundância e estabilidade dos coeficientes.

### Insight 5 - O modelo prioriza cobertura, não precisão operacional

Ao usar risco maior ou igual a 50%, 4.039 dos 5.352 municípios são sinalizados. A cobertura dos municípios que não atingiram a meta é alta, mas a lista é extensa. O limiar deve ser escolhido conforme capacidade real de atendimento.

### Insight 6 - O drift impede uma leitura otimista das métricas

ROC-AUC de 0,819 demonstra capacidade de ordenação, mas a divergência regional e o baixo recall da classe de atingimento mostram que o modelo não está bem calibrado para todas as decisões de 2024.

---

## 9. Limitações do projeto

1. **Ausência de microdados:** os arquivos não têm ID, características ou trajetória de alunos. O modelo descreve municípios, não probabilidades individuais.
2. **Proxy do alvo:** atingir a meta de 2025 não equivale à definição individual de alfabetização.
3. **Somente dois ciclos:** há pouco histórico para validar generalização temporal.
4. **Features contemporâneas:** a solução é nowcast, não previsão anterior à avaliação.
5. **Drift temporal:** a prevalência do alvo muda de 18,1% para 43,4%.
6. **Calibração regional:** o risco previsto para algumas regiões diverge do resultado observado.
7. **Sem fontes socioeconômicas externas:** renda, população, infraestrutura e formação docente não estão no modelo.
8. **Território como proxy:** UF pode capturar fatores omitidos sem explicá-los.
9. **Redundância:** UF e região, assim como duas variáveis estaduais, carregam informação relacionada.
10. **Sem inferência causal:** coeficientes e importâncias não comprovam impacto de políticas.
11. **Identificação limitada:** a base contém código IBGE, mas não nome do município.
12. **Meta variável:** usar uma meta municipal no alvo muda o limiar de classificação entre municípios.
13. **Comparação no holdout:** os três algoritmos são comparados em 2024. A escolha do hiperparâmetro usa apenas 2023, mas a avaliação não é independente de toda a seleção do modelo.

---

## 10. Aplicação prática para políticas públicas

O modelo pode apoiar:

- triagem inicial de municípios;
- definição de prioridades para visitas e assistência técnica;
- acompanhamento regional;
- seleção de territórios para análises qualitativas;
- construção de cenários conforme a capacidade de atendimento.

Não deve ser usado para:

- classificar ou punir alunos;
- distribuir recursos automaticamente;
- avaliar causalmente uma política;
- comparar gestores sem contexto;
- publicar risco futuro antes de existirem indicadores defasados.

### Uso recomendado

1. O gestor define quantos municípios consegue acompanhar.
2. O ranking fornece uma ordem inicial de análise.
3. A equipe combina o escore com dados administrativos locais.
4. Casos inconsistentes são revisados manualmente.
5. O resultado real do próximo ciclo é usado para monitorar drift e recalibrar o modelo.

---

## 11. Possíveis evoluções futuras

### Dados

- incorporar um terceiro ciclo comparável;
- usar apenas variáveis disponíveis antes do ciclo previsto;
- adicionar nomes municipais por tabela oficial de códigos;
- testar enriquecimento opcional com IBGE, Censo Escolar, FUNDEB e PNAD;
- criar variáveis de política educacional e capacidade da rede.

### Modelagem

- redefinir **risco** como classe positiva para alinhar métricas ao negócio;
- calibrar probabilidades com CalibratedClassifierCV;
- comparar validação temporal walk-forward;
- otimizar o limiar por orçamento e custo de falsos negativos;
- medir métricas por região e UF;
- testar remoção de variáveis redundantes;
- adicionar intervalos de confiança por bootstrap;
- monitorar drift de features, alvo e calibração.

### Governança

- criar ficha de dados e política de atualização;
- registrar versão das fontes e data de referência;
- estabelecer revisão humana obrigatória;
- publicar critérios de uso, contestação e monitoramento.

---

## 12. Respostas às perguntas norteadoras

### Quais fatores mais impactam a alfabetização?

Dentro do modelo, media_portugues é a variável dominante, seguida por indicadores territoriais. O resultado mede associação preditiva. A base não permite concluir que elevar isoladamente uma dessas variáveis causará melhoria na alfabetização.

### Quais municípios apresentam maior risco educacional?

O ranking completo está em [reports/ranking_risco_2024.csv](reports/ranking_risco_2024.csv). Os primeiros códigos são predominantemente de Tocantins, Piauí, Sergipe e Bahia. Como as probabilidades extremas refletem drift e falta de calibração, a lista deve iniciar uma investigação, não encerrá-la.

### Quais regiões possuem padrões semelhantes?

Nordeste e Sudeste apresentam risco médio previsto praticamente igual, próximo de 79,3%. Entretanto, seus resultados observados são diferentes. A semelhança do escore não significa equivalência educacional; neste caso, ela expõe uma limitação do modelo.

### Como prever municípios que podem não atingir metas futuras?

A implementação atual não responde rigorosamente a essa pergunta porque utiliza indicadores do mesmo ciclo. Para previsão futura, as features precisam ser defasadas e avaliadas em um ciclo posterior nunca visto pelo modelo.

### Quais variáveis possuem maior influência nos modelos?

Na Regressão Logística, media_portugues tem maior magnitude, seguida por dummies estaduais. taxa_media_uf_loo também aparece entre as quinze maiores. A tabela completa pode ser reproduzida no notebook 04.

---

## 13. Como reproduzir

### Instalação

No PowerShell:

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt

### Pipeline de dados

    python -m src.preprocessing.run_pipeline

### Dataset de modelagem

    python -m src.preprocessing.build_features

### Treino reproduzível

    python -m src.modeling.model_card

### Notebooks

Execute na ordem:

1. [01_pipeline_medalhao.ipynb](notebooks/01_pipeline_medalhao.ipynb)
2. [02_analise_exploratoria.ipynb](notebooks/02_analise_exploratoria.ipynb)
3. [03_modelagem.ipynb](notebooks/03_modelagem.ipynb)
4. [04_aplicacao_estrategica.ipynb](notebooks/04_aplicacao_estrategica.ipynb)

---

## Estrutura principal

    data/
      br_inep_*.csv
      features/dataset_modelagem.parquet
      lake/
        bronze/
        silver/
        gold/
    images/
      02_distribuicao_indicador_proxy.png
      02_padroes_territoriais_correlacoes.png
      03_avaliacao_modelo.png
      03_interpretabilidade_coeficientes.png
      04_risco_por_regiao.png
    notebooks/
    reports/
      comparacao_modelos.csv
      model_card.json
      modelo_final.joblib
      ranking_risco_2024.csv
      relatorio_tecnico.md
    src/
      preprocessing/
      modeling/
      evaluation/
      visualization/

## Artefatos técnicos

- [Relatório técnico](reports/relatorio_tecnico.md)
- [Model card](reports/model_card.json)
- [Comparação dos modelos](reports/comparacao_modelos.csv)
- [Ranking de risco](reports/ranking_risco_2024.csv)