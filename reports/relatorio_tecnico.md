# Relatório técnico - classificação municipal de alfabetização

Documento de decisões e metodologia. O README traz a narrativa e a execução; os
notebooks trazem os resultados célula a célula; aqui ficam registrados **por que**
as escolhas foram feitas e o que a verificação mostrou.

## 1. Delimitação do problema

O Tech Challenge solicita um modelo supervisionado para prever alfabetização. Os
dados da Fase 2 são agregados por município, ano e rede, sem observações
individuais. O projeto, portanto, não treina um classificador de alunos.

### 1.1 A ausência de microdado é da fonte, não da escolha

| Fonte | Grão mais fino disponível | Identifica aluno? |
|---|---|---|
| Indicador município | município × ano × rede | não |
| Indicador UF | UF × ano × rede | não |
| Meta Brasil | ano × rede | não |
| Meta UF | UF × ano × rede | não |
| Meta município | município × ano × rede | não |

Nenhuma das cinco fontes desce abaixo do município. O INEP não publica microdados
por aluno para esta avaliação (sigilo estatístico e LGPD). A unidade de análise
adotada — município-ano da rede municipal — é a menor tecnicamente possível.

O alvo binário é uma proxy:

- 1: taxa de alfabetização observada maior ou igual à meta municipal de 2025;
- 0: taxa observada abaixo da meta;
- registros sem meta conhecida são excluídos do dataset de modelagem (242 linhas).

Essa formulação reproduz o trabalho de referência, mas a interpretação correta é
atingimento de meta municipal. Converter o resultado agregado em probabilidade
individual seria falácia ecológica.

## 2. Reimplementação da Fase 2

### 2.1 Fontes

| Fonte | Registros |
|---|---:|
| Indicador município | 23.995 |
| Indicador UF | 145 |
| Meta Brasil | 3 |
| Meta UF | 54 |
| Meta município | 10.704 |

Fontes externas foram removidas do fluxo porque o enunciado as apresenta como
enriquecimento opcional. A consequência é declarada na seção 9: o modelo não tem
nenhuma variável acionável de política pública.

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
**Silver:** padronização de rede, arredondamento, deduplicação e qualidade.
**Gold:** quatro visões analíticas da Fase 2. Nenhuma feature é excluída por leakage.

Essa fronteira é proposital: a engenharia de dados disponibiliza dados auditáveis;
a EDA justifica a seleção de variáveis. Um dado descartado na ingestão não pode ser
reexaminado depois.

## 3. Saídas da Gold

| Visão | Registros | Colunas de negócio |
|---|---:|---:|
| alfabetizacao_por_municipio | 10.896 | 11 |
| evolucao_temporal | 49 | 8 |
| ranking_municipios | 10.896 | 8 |
| comparacao_metas_nacionais | 1 | 9 |

## 4. Análise exploratória e leakage

### 4.1 Exclusões por construção

| Coluna | Decisão |
|---|---|
| taxa_alfabetizacao | excluída: define o alvo |
| gap_meta_2025 | excluída: função direta da taxa e da meta |
| status_meta_2025 | excluída: origem do rótulo |
| nivel_alfabetizacao | excluída: faixa discretizada da taxa |
| proporcao_aluno_nivel_0 a 8 | excluídas: decomposição da mesma avaliação (0,986 com a taxa) |
| id_municipio e ano | identificação e split, não entram em X |

`taxa_media_uf_loo` é a média dos pares da mesma UF e ano, **sem** o município da
linha, para não reintroduzir a própria taxa por via indireta.

A seleção acontece em `src/preprocessing/build_features.py`, depois de demonstrada
no notebook 02. O resultado tem 10.654 registros e nove colunas.

### 4.2 O vazamento residual, medido em vez de declarado

Duas variáveis mantidas continuam próximas do alvo:

- `media_portugues` correlaciona **0,9308 em 2023 e 0,9230 em 2024** com a taxa;
- `meta_2025` é literalmente um dos lados da desigualdade que define o rótulo.

Para medir o quanto isso basta, ajustamos um polinômio de grau 3 de
`taxa ~ media_portugues` **em 2023** e aplicamos a regra `taxa estimada >= meta`:

| Ciclo | Acurácia da regra sem modelo | Prevalência real |
|---|---:|---:|
| 2023 | 0,8406 | 0,181 |
| 2024 | **0,8709** | 0,434 |

A regra supera a acurácia do modelo completo (0,7173). Conclusão registrada: o
exercício é **nowcast**, classificação de um ciclo já avaliado, e não previsão
*ex ante*. Manter `media_portugues` segue a referência e é defensável (é uma medida
psicométrica em escala de proficiência, não o percentual do alvo), mas a distância
de 0,93 impede chamar o resultado de antecipação.

## 5. Variáveis finais

| Tipo | Variáveis |
|---|---|
| Numéricas | media_portugues, meta_2025, meta_uf_2025, taxa_media_uf_loo |
| Categóricas | sigla_uf, regiao |
| Identificadores | id_municipio, ano |
| Alvo | alvo_alfabetizado |

Após o one-hot, o estimador vê **33 colunas**: 4 numéricas, 24 dummies de UF e 5 de
região. `meta_uf_2025` e `taxa_media_uf_loo` correlacionam 0,896 entre si —
redundância territorial que justifica manter a regularização L2.

### 5.1 Por que `handle_unknown="ignore"` não é detalhe

São 24 dummies de UF, e não 25, porque **o Acre só aparece em 2024**: o encoder é
ajustado em 2023, que tem 24 UFs. Os 22 municípios acreanos do holdout chegam ao
estimador com o bloco de UF inteiramente zerado, e são classificados apenas pelas
demais variáveis.

Sem `handle_unknown="ignore"` a predição falharia com erro em vez de degradar
graciosamente. O caso é real e mede o custo da opção: nenhum dos 22 municípios do
Acre atingiu a meta em 2024, e o modelo os avalia sem qualquer efeito estadual.

## 6. Pré-processamento e validação

O `ColumnTransformer` executa:

- `SimpleImputer(mediana)` + `StandardScaler` nas numéricas;
- `SimpleImputer(moda)` + `OneHotEncoder(handle_unknown="ignore")` nas categóricas.

Tudo dentro do `Pipeline`, portanto reajustado em cada fold — nenhuma estatística
do conjunto de avaliação vaza para o treino.

Estratégia:

- treino temporal: 2023, 5.302 registros, prevalência 18,1%;
- holdout: 2024, 5.352 registros, prevalência 43,4%;
- tuning: `StratifiedKFold` de 5 folds, semente 42, apenas no treino;
- seleção de hiperparâmetro por F1 dentro de 2023;
- holdout usado para avaliação final.

## 7. Seleção do algoritmo: as duas comparações discordam

Este é o achado metodológico central do projeto e a razão de o relatório dedicar
uma seção inteira a ele.

### 7.1 Comparação por validação cruzada, só em 2023

| Modelo | ROC-AUC val. | PR-AUC val. | F1 val. | Gap treino-val. |
|---|---:|---:|---:|---:|
| Random Forest | 0,9960 | 0,9919 | 0,9794 | 0,0033 |
| Gradient Boosting | 0,9945 | 0,9814 | 0,9801 | 0,0054 |
| Regressão Logística | 0,9740 | 0,9163 | 0,7957 | 0,0028 |
| Baseline | 0,5000 | 0,1813 | 0,0000 | 0,0000 |

O gap treino-validação é baixo em todos os três: **a validação cruzada não acusa
sobreajuste**.

### 7.2 Comparação no ciclo seguinte

| Modelo | CV 2023 | Holdout 2024 | Queda |
|---|---:|---:|---:|
| Random Forest | 0,9960 | 0,7105 | −0,2855 |
| Gradient Boosting | 0,9945 | 0,8155 | −0,1790 |
| Regressão Logística | 0,9740 | **0,8191** | −0,1549 |

**A ordem se inverte.** O primeiro colocado é o último.

### 7.3 Por que isso acontece

`alvo = taxa >= meta_2025`. As features contêm `meta_2025` e um proxy de 0,93 da
taxa. Um modelo com capacidade de representar interações **reconstrói a fronteira
de decisão dentro do ano** — daí o 0,996. Essa fronteira depende da relação
nota↔taxa e da prevalência, e ambas se deslocam entre 2023 e 2024 (18,1% → 43,4%).
A Regressão Logística, aditiva e regularizada, não consegue memorizar a interação e
por isso perde menos.

### 7.4 Consequência metodológica

Validação cruzada dentro de um único ciclo mede consistência entre folds, não
generalização temporal. Com dois ciclos e drift alto, ela é uma medida enganosa. A
escolha do modelo apoia-se no holdout de 2024 — o que, honestamente, significa que
a avaliação **não é independente de todas as decisões**. Um terceiro ciclo é
necessário para fechar esse ponto, e isso está declarado nas limitações.

## 8. Otimização

`GridSearchCV` sobre `C ∈ {0,01; 0,1; 1,0; 10,0}`, métrica F1, com
`return_train_score=True`:

| C | F1 treino | F1 validação | Gap |
|---:|---:|---:|---:|
| 0,01 | 0,7610 | 0,7617 | −0,0007 |
| 0,10 | 0,7921 | 0,7889 | 0,0031 |
| **1,00** | 0,7995 | **0,7957** | 0,0038 |
| 10,00 | 0,7989 | 0,7956 | 0,0033 |

`C = 1,0` vence. O gap nunca passa de 0,004: a regularização é salvaguarda contra
redundância territorial, não remédio para memorização.

## 9. Métricas e por que não a acurácia sozinha

No holdout de 2024: acurácia 0,7173 contra **baseline de 0,5661** (prever sempre
"não atingiu"). O ganho sobre o baseline é de 15,1 pontos — relevante, mas a
acurácia isolada esconde o desequilíbrio, por isso ROC-AUC (0,8191) e PR-AUC
(0,7878) são reportados junto.

| Métrica | Valor |
|---|---:|
| Acurácia | 0,7173 |
| Acurácia do baseline | 0,5661 |
| Precisão (atingiu) | 0,8081 |
| Recall (atingiu) | 0,4569 |
| F1 (atingiu) | 0,5838 |
| ROC-AUC | 0,8191 |
| PR-AUC | 0,7878 |

### 9.1 Calibração

Desvio médio entre probabilidade prevista e frequência observada: **0,1925**. O
modelo subestima o atingimento em 2024, consequência direta do drift de
prevalência. A ordenação é utilizável; o valor absoluto não. Por isso o produto
final é um **ranking**, não um número publicável por município.

## 10. Interpretabilidade

Três lentes sobre o mesmo modelo, conforme recomendado pelo enunciado.

### 10.1 Coeficientes padronizados

`media_portugues` 5,416; `sigla_uf_RN` 2,570; `sigla_uf_PA` 2,161;
`sigla_uf_GO` 1,970; `sigla_uf_MS` −1,795.

### 10.2 Permutation importance (holdout de 2024, 10 repetições)

| Variável | Queda de ROC-AUC |
|---|---:|
| media_portugues | +0,3815 |
| meta_2025 | +0,0059 |
| meta_uf_2025 | −0,0016 |
| taxa_media_uf_loo | −0,0063 |
| sigla_uf | −0,0133 |
| regiao | −0,0276 |

Quatro das seis variáveis têm importância **negativa**: embaralhá-las melhora o
desempenho no ciclo seguinte. O bloco territorial ajusta uma geografia de 2023 que
2024 não reproduz.

### 10.3 SHAP Values (`LinearExplainer`, valores exatos)

| Variável | SHAP médio absoluto |
|---|---:|
| media_portugues | 3,7994 |
| taxa_media_uf_loo | 0,8112 |
| regiao_Sul | 0,5350 |
| meta_uf_2025 | 0,3565 |
| regiao_Nordeste | 0,3129 |
| sigla_uf_RN | 0,2180 |

As três lentes convergem: `media_portugues` é o modelo. As demais ajustam margens.

## 11. Aplicação estratégica

### 11.1 Ranking e limiar

`prob_risco = 1 - prob_atingir_meta`. O arquivo `reports/ranking_risco_2024.csv`
traz o ranking completo do holdout.

| Limiar | Municípios sinalizados | Cobertura do risco | Precisão do alerta |
|---:|---:|---:|---:|
| 0,5 | 4.039 | 0,9168 | 0,6878 |
| 0,7 | 3.680 | 0,8667 | 0,7136 |
| 0,8 | 3.472 | 0,8350 | 0,7287 |
| 0,9 | 3.136 | 0,7825 | 0,7561 |

O corte é decisão de capacidade de atendimento, não do modelo.

### 11.2 Padrões regionais: onde o modelo falha

| Região | Risco previsto | Risco observado | Diferença |
|---|---:|---:|---:|
| Norte | 0,9414 | 0,7459 | +0,20 |
| Nordeste | 0,7931 | 0,6185 | +0,17 |
| Sudeste | 0,7929 | 0,4294 | **+0,36** |
| Centro-Oeste | 0,6369 | 0,3578 | +0,28 |
| Sul | 0,5462 | 0,7027 | **−0,16** |

Nordeste e Sudeste recebem escore idêntico (0,793) e têm realidades opostas. O Sul
é a única região cujo risco o modelo subestima — e é a única que piorou entre os
ciclos. **A pergunta "quais regiões possuem padrões semelhantes" é a que este
trabalho responde pior**, e isso está declarado em vez de maquiado.

### 11.3 Metas futuras

A base atual não sustenta previsão *ex ante*: duas features são do próprio ciclo.
Seriam necessários features defasadas, três ciclos comparáveis e validação em ciclo
posterior nunca usado.

## 12. Limitações

1. A proxy municipal não equivale a um rótulo individual.
2. Vazamento residual assumido: uma regra sem modelo acerta 87,1% em 2024.
3. Apenas dois ciclos; uma única transição temporal.
4. Prevalência muda de 18,1% para 43,4% entre treino e teste.
5. Calibração insuficiente (desvio 0,1925).
6. O bloco territorial prejudica a generalização, medido por permutation importance.
7. Ausência de fontes externas remove qualquer alavanca acionável de política.
8. Importância preditiva não autoriza recomendação causal.
9. A seleção final consultou o holdout; um terceiro ciclo fecharia o ponto.
10. Não há nomes de municípios, somente códigos IBGE.

## 13. Hipóteses da EDA - veredito

| # | Hipótese | Veredito |
|---|---|---|
| H1 | Proficiência em Português associada a maior chance de atingir a meta | **confirmada** — domina as três lentes |
| H2 | O contexto estadual captura desigualdades territoriais relevantes | **parcialmente refutada** — captura a geografia de 2023, mas prejudica 2024 |
| H3 | O salto de prevalência reduz a generalização temporal | **confirmada** — queda de 0,155 a 0,286 de ROC-AUC |
| H4 | As associações não devem ser lidas como efeitos causais | mantida como restrição de interpretação |

H2 é a mais instrutiva: a intuição de que o território ajuda estava correta *dentro*
de um ciclo e errada *entre* ciclos.

## 14. Decisões revistas durante a revisão

| Afirmação anterior | O que a verificação mostrou |
|---|---|
| "A comparação entre algoritmos usa 2024, o que é apenas uma ressalva" | É o que impede escolher o pior modelo. A comparação por CV em 2023 elegeria o Random Forest, que colapsa para 0,711 |
| "media_portugues é feature legítima, apenas correlacionada" | Correto, mas insuficiente: com `meta_2025`, uma regra sem modelo acerta 87,1% |
| "As dummies de UF contribuem de forma secundária, porém consistente" | Consistente dentro de 2023; **negativa** em 2024 na permutation importance |
| "Nordeste e Sudeste possuem padrões semelhantes" | Semelhança de escore, não de realidade: 42,9% contra 61,9% de risco observado |
| Interpretabilidade só por coeficientes | Acrescentadas permutation importance e SHAP, conforme o enunciado recomenda |

## 15. Conclusão

A simplificação atende ao núcleo do desafio: reproduz a Gold da Fase 2, realiza EDA,
trata leakage na etapa correta, integra o pré-processamento ao modelo, compara
algoritmos por dois critérios independentes, valida temporalmente, interpreta por
três lentes e gera um ranking aplicável à triagem.

O principal ganho não foi maximizar métricas, mas tornar o encadeamento auditável e
os limites explícitos:

    cinco CSVs -> Bronze -> Silver -> quatro Golds -> EDA/leakage
    -> dataset de 9 colunas -> Pipeline Scikit-learn -> ranking municipal

E o resultado mais útil para quem for continuar o trabalho não é o ROC-AUC de
0,819, e sim a demonstração de que, neste problema, **a validação cruzada dentro de
um ciclo teria levado à pior escolha possível**.
