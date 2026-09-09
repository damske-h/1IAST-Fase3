# Relatório Técnico — Predição de Risco de Alfabetização nos Municípios Brasileiros

**Tech Challenge – Fase 3 | POSTECH AI Scientist**

Este documento consolida as **decisões analíticas e a metodologia** do projeto.

| Documento | Papel |
|---|---|
| [`README.md`](../README.md) | narrativa, resultados e instruções de execução |
| **este relatório** | por que cada decisão foi tomada, e o que foi descartado |
| [`notebooks/`](../notebooks/) | os resultados reproduzíveis, célula a célula |
| [`reports/model_card.json`](model_card.json) | ficha técnica gerada por código a cada execução |
| [`reports/ranking_risco_municipios.csv`](ranking_risco_municipios.csv) | a saída operacional: risco previsto por município |

**Nenhum número deste relatório foi digitado à mão.** Todos vêm do
`model_card.json`, gerado por `python -m src.modeling.model_card`, ou de células
executadas nos notebooks, com a referência indicada.

---

## 1. Problema e unidade de análise

O **Compromisso Nacional Criança Alfabetizada** pactua metas de alfabetização por município,
aferidas pelo Saeb/INEP ao fim do 2º ano do ensino fundamental, com o corte de 743 pontos na
escala de Língua Portuguesa. O enunciado pede um modelo supervisionado que preveja se **um aluno**
será considerado alfabetizado.

### 1.1 Por que o modelo descreve o município, e não o aluno

**Não existe microdado público de aluno para nenhuma das fontes deste projeto.** Não é uma
escolha de conveniência — é uma restrição da matéria-prima, e vale a pena ser explícito sobre
onde ela aparece:

| Fonte | Grão mais fino disponível | Tem identificador de aluno? |
|---|---|---|
| Indicador Criança Alfabetizada (INEP) | município × ano × série × rede | não |
| Censo Escolar — AFD, ATU, IED | município × ano × localização × dependência | não |
| Saeb — INSE | município × tipo de rede × localização | não |
| IDEB Anos Iniciais | município × rede × ciclo | não |
| Metas do CNCA | município × rede | não |

Nenhuma traz sexo, idade, trajetória, frequência ou qualquer atributo individual — nem uma chave
que permitisse ligá-las entre si no grão do estudante. Um modelo "de aluno" construído sobre
essas bases seria um modelo de município com outro nome.

**A ponte adotada:** a chance de uma criança ser alfabetizada é tratada como **dada pelo contexto
socioeconômico, educacional e político do município** em que ela estuda. Modela-se o contexto —
formação do corpo docente, tamanho das turmas, esforço docente, nível socioeconômico das
famílias, ruralidade, porte, desempenho pregresso da rede e unidade federativa — e a leitura
para o aluno é direta: *uma criança que estuda num município sinalizado em risco tem chance
substancialmente menor de chegar alfabetizada ao fim do 2º ano*.

**A limitação inseparável dessa escolha é a falácia ecológica.** Duas crianças do mesmo município
recebem exatamente a mesma predição. O modelo descreve a condição média, não a trajetória
individual — e a §15 traz essa ressalva junto de todas as outras.

### 1.2 O rótulo: `em risco = taxa < 50%`

O município entra como **em risco** quando **menos da metade** das suas crianças chega
alfabetizada. Duas alternativas foram consideradas e descartadas:

| Alternativa | Por que foi descartada |
|---|---|
| A **meta pactuada** de cada município | é calculada a partir da taxa de 2023 do próprio município — usá-la como rótulo tornaria o alvo função da própria história do município, e a meta vira preditor circular |
| A **média nacional do ano** | é um alvo *relativo*: metade do país está sempre em risco por construção, melhore o país ou piore. Não se conecta a nenhum compromisso absoluto |

O corte em 50% é **absoluto e interpretável sem contexto estatístico**: um prefeito entende
imediatamente o que significa. E cai numa região densa da distribuição (`notebooks/02`, §1), o
que evita isolar uma cauda residual.

Resultado no ciclo modelado: **1.486 dos 5.448 municípios em risco — 27,3%.** Classe minoritária,
o que define as métricas da §9.

### 1.3 Um único ciclo

A modelagem usa apenas **2024**, uma linha por município. O motivo está na EDA: o Rio Grande do
Sul caiu **20,2 p.p.** entre os dois ciclos (§5, A3), um choque exógeno que nenhuma variável da
base descreve. Isso quebra a comparabilidade entre 2023 e 2024 e inviabiliza tratar os dois
ciclos como amostras do mesmo processo.

**A contrapartida é declarada, não escondida:** o ciclo escolhido é justamente o atingido pelo
choque. Os municípios gaúchos entram no modelo com taxa deprimida por um evento conjuntural, e o
classificador lê isso como estrutura. **O risco previsto para o RS está superestimado** — e o
coeficiente da UF (OR 11,5) precisa ser lido com essa ressalva.

---

## 2. Origem dos dados

### 2.1 Pipeline medalhão reconstruída localmente

A Fase 2 entregou a arquitetura Bronze → Silver → Gold em AWS Glue + S3 + Athena. A Fase 3 a
**reproduz em Python/pandas, sem nuvem**, preservando a semântica: schema explícito, hash de
deduplicação, DQ com quarentena, particionamento Hive-style por ano e escrita idempotente.

**Por que reconstruir em vez de só consumir o Parquet da Gold?** Três motivos:

1. a Gold da Fase 2 foi desenhada para **análise descritiva**, e suas colunas mais informativas
   são justamente as que vazam o alvo (§4);
2. as fontes externas precisavam entrar **na Silver e na Gold**, sob as mesmas regras de
   qualidade — sem elas restariam ~4 variáveis legítimas;
3. reprodutibilidade sem credenciais: qualquer avaliador roda um comando e reconstrói tudo.

| Camada | Resultado (`notebooks/01`) |
|---|---|
| Bronze | 10 entidades, **719.757 registros**, score de qualidade 100% |
| Silver | **54.165 duplicatas removidas**, **890 registros em quarentena** |
| Gold | 5 visões — as 4 da Fase 2 + `base_ml_alfabetizacao` |
| Idempotência | 29/29, 30/30 e 10/10 partições idênticas em duas execuções |

### 2.2 Fontes e enriquecimento

| Origem | Conteúdo | Papel |
|---|---|---|
| INEP / Base dos Dados (5 CSVs) | indicador por município e UF; metas Brasil/UF/município | alvo e metas |
| Censo Escolar — AFD (2023, 2024) | adequação da formação docente | enriquecimento |
| Censo Escolar — ATU (2023, 2024) | média de alunos por turma | enriquecimento |
| Censo Escolar — IED (2023, 2024) | esforço docente | enriquecimento |
| Saeb — INSE (2023) | nível socioeconômico municipal | enriquecimento |
| IDEB Anos Iniciais (ciclo 2021) | IDEB, notas e taxa de aprovação | preditor defasado |

O enriquecimento **não era opcional**: sem ele, a maior correlação disponível com o alvo seria a
de `ano` (0,06). Com ele, 0,53 (`ideb_2021`).

### 2.3 Grão da base analítica

A base construída tem **10.896 linhas** (município × ano, rede municipal, ciclos 2023 e 2024),
cobrindo 5.500 municípios e **25 das 27 UFs**. A modelagem consome dela o recorte de **2024**:
**5.448 municípios, uma linha cada**.

A rede municipal foi escolhida por ser a única com o mesmo número de municípios nos dois ciclos
(5.448) e por ser a rede sob gestão direta do município — que é o destinatário das recomendações.
A rede *total* só existe em 2024 e para 398 municípios.

---

## 3. Qualidade de dados — achados e tratamento

Todos viraram check de DQ ou nota de tratamento (`notebooks/01`, §2):

| # | Achado | Tratamento |
|---|---|---|
| 1 | **O arquivo do INSE repete a mesma chave até 7 vezes** (125.741 linhas para 5.558 municípios) | ingerido como chega no Bronze; deduplicado na Silver pelo `_record_hash` — **54.165 linhas**. Sem isso, o join multiplicaria a base |
| 2 | 890 estratos do INSE sem `media_inse` (omissão por sigilo estatístico) | quarentena, com motivo registrado |
| 3 | Percentual de nível do INSE **sem alunos vem em branco, não como zero** | verificado que, tratando branco como zero, os 8 níveis somam 100% em todas as linhas (média 100,00; desvio 0,01) → zero estrutural, não ausência |
| 4 | Cabeçalhos multi-nível nas planilhas do INEP | `skiprows` até a linha de nomes técnicos |
| 5 | Nulos como texto (`"--"`, `"-"`) | `na_values` |
| 6 | `CO_MUNICIPIO` como float nos xlsx, texto de 7 dígitos no CSV | normalização com `zfill(7)` antes de qualquer join |
| 7 | Chave de rede divergente entre arquivos (código vs. texto) | normalização via `REDE_MAP` |
| 8 | **Roraima ausente do arquivo do INEP** em qualquer rede, nos dois ciclos; DF não tem rede municipal | declarado como limitação de cobertura |

---

## 4. Tratamento de data leakage

É o núcleo metodológico do projeto. O arquivo do INEP traz colunas que parecem excelentes
preditoras e são, na verdade, **a mesma medição que gerou o alvo**.

### 4.1 O grau de vazamento foi medido, não presumido

| Coluna | Por que vaza | Correlação com o alvo |
|---|---|---:|
| `media_portugues` | mesma escala Saeb da qual a taxa é o % de alunos com 743+ pontos | **0,927** |
| `proporcao_aluno_nivel_*` | distribuição de proficiência da qual a taxa deriva | **0,986** (soma dos níveis 5–8) |
| `meta_alfabetizacao_2025` | a meta é calculada a partir da taxa de 2023 | **0,966** |
| `nivel_alfabetizacao` | a própria taxa discretizada | — |
| IDEB dos ciclos 2023 e 2025 | contemporâneo e posterior ao alvo | — |

### 4.2 O critério de exclusão

Não é a força da correlação. É: **esta informação estaria disponível no momento em que a predição
precisaria ser feita?**

Por isso `media_portugues` (r = 0,927) sai, e o **IDEB de 2021** (r = 0,54) fica — publicado dois
anos antes do ciclo modelado.

> **Nuance registrada.** A exclusão vale para a coluna **contemporânea**. `media_portugues`
> *defasada* (do ciclo anterior) seria um preditor legítimo. Não a usamos porque, com apenas dois
> ciclos, defasar consumiria o único ciclo alternativo e substituiria o modelo estrutural por um
> modelo de inércia — que prevê bem e explica pouco (§11).

### 4.3 Onde o tratamento é aplicado

Em `src/preprocessing/config.py::COLUNAS_VAZAMENTO` — **23 colunas**, cada uma com o motivo
registrado — e removidas programaticamente por `gold.montar_base_ml()`. A decisão fica auditável
no repositório, e um `assert` no `notebooks/01` verifica que nenhuma sobreviveu até a base de
modelagem.

**A meta pactuada também não entra no modelo.** Ela reaparece no `notebooks/04` §4, mas apenas
como **régua de comparação depois da predição** — nunca como preditor.

### 4.4 O segundo vazamento, mais sutil

Estatísticas de pré-processamento também vazam. Mediana da imputação, média e desvio da
padronização e categorias do encoder são calculadas **dentro do `Pipeline`**, portanto só com o
fold de treino. Padronizar antes de separar os folds usaria informação do conjunto de validação —
e não geraria erro nenhum.

---

## 5. Análise exploratória — os achados que decidiram a modelagem

`notebooks/02`. Cada seção existe para responder a uma pergunta cuja resposta muda a modelagem.

| # | Achado | Consequência |
|---|---|---|
| A1 | Alvo aproximadamente simétrico (assimetria ≈ −0,15), sem massa nos extremos (~1% em 100%, ~1,5% abaixo de 20%) | o corte em 50% cai em região densa; sem risco de separação perfeita |
| A2 | Dispersão **dentro** de cada região é enorme; amplitude entre regiões de ~23 p.p. | métricas de ordenação e calibração, não acurácia |
| A3 | **O RS caiu 20,2 p.p.** de 2023 para 2024, com 89,6% dos municípios em queda — enquanto todas as demais regiões melhoraram | modelagem restrita a um ciclo; risco do RS declarado como superestimado |
| A4 | Sem enriquecimento externo, a maior correlação disponível seria a de `ano` (0,06); com ele, 0,53 | o enriquecimento não era opcional |
| A5 | **Paradoxo de Simpson no INSE:** Spearman agregado 0,29; dentro das regiões — Norte 0,31, Centro-Oeste 0,20, Sul 0,04, Sudeste 0,01, **Nordeste −0,14** | `sigla_uf` como controle **obrigatório** |
| A6 | Ceará e Pernambuco: mesmo INSE (4,37), taxas de 90,1% e 63,0% | hipótese de efeito de política estadual (H4) |
| A7 | AFD, IED e níveis do INSE são **composicionais** (somam 100%) | descartar uma categoria de referência por bloco |
| A8 | No bloco IDEB, `ideb`/notas correlacionam 0,953–0,961 e `indicador_rendimento`/`taxa_aprovacao` correlacionam **0,996** | manter só `ideb_2021` e `taxa_aprovacao_2021` |
| A9 | Escalas incomparáveis (de 3 a 80.160 contra 4 a 6) | padronização obrigatória |
| A10 | Mesmo município correlaciona **0,64** entre seus dois ciclos | dois terços do sinal cabem num ciclo só; e com uma linha por município o vazamento entre folds deixa de ser possível por construção |

---

## 6. Seleção de variáveis

`src/modeling/features.py`. **28 colunas de entrada → 53 features após o pré-processamento.**

| Bloco | Variáveis | Referência descartada |
|---|---:|---|
| Formação docente (AFD) | 4 | `afd_ai_grupo_5` (sem curso superior) |
| Esforço docente (IED) | 5 | `ied_ai_nivel_1` (menor esforço) |
| Níveis do INSE | 7 | `inse_pc_nivel_1` (nível mais baixo) |
| Tamanho de turma (ATU) | 5 | — (médias, não composicional) |
| Socioeconômico — outras | 3 | — |
| IDEB 2021 (defasado) | 2 | — (com indicador de ausência) |
| Categóricas | 2 | *dummy encoding* (`drop="first"`) |

**Duas exclusões que não vieram da EDA:**

- **`regiao`** é função determinística de `sigla_uf` — as dummies de UF já a codificam
  integralmente. Incluí-la só somaria colinearidade.
- **`ano`** fica fora porque a modelagem usa um único ciclo: a coluna é constante.

---

## 7. Pipeline de pré-processamento

Três ramos no `ColumnTransformer`, todos **dentro** do `Pipeline`:

| Ramo | Tratamento | Motivo |
|---|---|---|
| Numéricas gerais | mediana → `StandardScaler` | a penalidade L2 encolhe coeficientes proporcionalmente à escala; sem padronizar, puniria arbitrariamente as variáveis de escala pequena |
| Bloco IDEB | mediana **com `add_indicator=True`** → `StandardScaler` | 13,3% de ausência concentrada em municípios pequenos e isolados: **a ausência é sinal** — e o indicador confirma, com OR 1,32 |
| Categóricas | moda → `OneHotEncoder(drop="first")` | *dummy encoding* — remove uma categoria para evitar multicolinearidade |

`handle_unknown="ignore"` cobre UFs eventualmente ausentes de uma partição de treino.

Todo o pré-processamento vive **dentro** do `Pipeline`, que é o objeto passado ao
`cross_validate` e ao `GridSearchCV` — é isso que garante que nenhuma estatística de
pré-processamento atravesse a fronteira entre folds.

---

## 8. Validação

Com o alvo no grão do município, há **uma linha por município** — nada de peso amostral, nada de
grupos a preservar. A validação volta a ser a padrão do Scikit-learn:

| Etapa | Estratégia | Pergunta que responde |
|---|---|---|
| Partição | `train_test_split` estratificado 75/25 | quanto o modelo erra em municípios que ele nunca viu? |
| Seleção e ajuste | `StratifiedKFold` de 5 folds **sobre o treino** | qual algoritmo e qual regularização, sem tocar no teste? |

**Estratificado nas duas etapas** porque a classe de risco é minoritária (27,3%): uma partição
aleatória simples poderia produzir folds com prevalências bem diferentes, e as métricas ficariam
instáveis por acidente de sorteio.

O conjunto de teste (1.362 municípios) é tocado **uma única vez**, no fim do `notebooks/03`.

> **O que a mudança de abordagem resolveu.** Na formulação anterior — alvo binomial ponderado —
> cada município gerava duas linhas idênticas com rótulos opostos, e toda a informação vivia no
> `sample_weight`. Um scorer que ignorasse o peso devolvia AUC exatamente 0,5, e fazer o peso
> chegar ao scorer exigiria declarar *metadata routing* em cada etapa aninhada do
> `ColumnTransformer` (testado: falha em `StandardScaler.fit_transform`). Era preciso um laço de
> validação escrito à mão. Com o alvo municipal, `GridSearchCV` e `cross_validate` funcionam
> direto — e o SVM e o KNN, antes inviáveis, entram na comparação.

---

## 9. Métricas: por que não acurácia

A classe de risco é **minoritária**. Um modelo que responda "fora de risco" para todo mundo
acerta **72,7%** e não serve para nada. A acurácia é reportada sempre ao lado desse baseline,
justamente para deixar claro que sozinha ela não diz nada.

| Métrica | Por que está aqui |
|---|---|
| **Recall da classe de risco** | a métrica prioritária. Um falso alarme custa uma visita técnica; um falso negativo custa uma geração |
| **PR-AUC** | mais informativa que a ROC quando a classe de interesse é minoritária — não se deixa inflar pelos verdadeiros negativos, que aqui são maioria |
| **ROC-AUC** | capacidade de ordenar municípios por risco, comparável entre modelos |
| **Curva de calibração** | permite ler a saída como *probabilidade*, e não só como ordem — é o que autoriza o corte por cobertura da §13 |

### 9.1 Resultado no conjunto de teste

`notebooks/03`, §7 — 1.362 municípios nunca vistos:

| Métrica | Valor |
|---|---:|
| ROC-AUC | **0,880** |
| PR-AUC | **0,693** |
| Recall da classe de risco | 0,577 |
| Precisão da classe de risco | 0,679 |
| Acurácia | 0,811 |
| *Acurácia do baseline (classe majoritária)* | *0,728* |

O recall de 0,577 é do limiar padrão de 0,5, que **não é o limiar de operação** — a §13 mostra
como escolhê-lo a partir da cobertura desejada.

---

## 10. Otimização de hiperparâmetros

`GridSearchCV` sobre `C` ∈ {0,01; 0,1; 1; 10; 100}, otimizando PR-AUC com `StratifiedKFold` de 5
folds (`notebooks/03`, §6):

- adotado **`C = 10,0`**, com PR-AUC de validação **0,786**;
- diferença treino-validação de **0,0070** — praticamente zero.

**Conclusão honesta:** o modelo **não sofria de overfitting**, e a busca não "resolveu" nenhum
problema. Com 4.086 municípios de treino para 53 features num modelo linear, a regularização L2
é salvaguarda contra colinearidade residual, não remédio para sobreajuste. Reportar isso é mais
útil do que apresentar a otimização como se tivesse produzido um ganho.

### 10.1 `class_weight="balanced"` foi testado e rejeitado

Seria o reflexo natural diante de uma classe minoritária. Medimos:

| Configuração | ROC-AUC | Desvio de calibração |
|---|---:|---:|
| Sem `class_weight` | 0,879 | **0,045** |
| `class_weight="balanced"` | 0,879 | 0,111 |

**Não muda a capacidade de ordenar e piora a calibração em 2,5×.** Como o projeto lê a saída como
probabilidade — e é isso que permite escolher o limiar por cobertura —, o desbalanceamento é
tratado **na escolha do limiar**, não no peso das classes.

---

## 11. Comparação de algoritmos

Todos os candidatos pelo **mesmo** pré-processamento, a mesma validação estratificada e a mesma
partição. Só o estimador muda (`notebooks/03`, §5; números em `model_card.json`).

As métricas da comparação são **independentes de limiar** — recall e precisão dependem do corte
de decisão, que só é escolhido depois, e compará-los aqui premiaria o modelo mais alarmista.

| Modelo | ROC-AUC | PR-AUC | Gap treino-validação |
|---|---:|---:|---:|
| Gradient Boosting | 0,8974 | **0,7964** | 0,0623 |
| **Regressão Logística** | **0,9027** | 0,7853 | **0,0070** |
| Random Forest | 0,8859 | 0,7817 | 0,0745 |
| SVM (RBF) | 0,8873 | 0,7772 | 0,0613 |
| KNN | 0,8433 | 0,6938 | 0,1567 |
| Árvore de Decisão | 0,8450 | 0,6889 | 0,0256 |
| Naive Bayes | 0,8575 | 0,6602 | 0,0060 |
| Baseline (classe majoritária) | 0,5000 | 0,2729 | 0,0000 |

**Leitura:**

1. **A Regressão Logística tem o melhor ROC-AUC do conjunto** e fica a 0,011 do melhor PR-AUC —
   dentro de qualquer margem razoável. Não é preciso argumentar "empata e é mais simples": ela
   lidera a métrica de ordenação.
2. **Sobreajusta quase dez vezes menos que os ensembles** (0,0070 contra 0,0623 e 0,0745). O
   KNN é o extremo oposto, com gap de 0,157.
3. **Sete famílias diferentes param na mesma faixa de PR-AUC (0,66–0,80).** Do linear ao boosting,
   do baseado em distância ao probabilístico: **a limitação é a informação disponível, não a
   capacidade de modelo.** É a justificativa quantitativa das evoluções futuras — microdados de
   aluno e variáveis de política moveriam o resultado; um oitavo algoritmo, não.
4. **Nenhum candidato ficou de fora.** Na formulação anterior, SVM e KNN eram inviáveis (o peso
   amostral dobrava a base para 21.792 observações e o `SVC` é O(n²)). Com 5.448 linhas, o
   catálogo completo das aulas coube.

**Decisão:** a Regressão Logística fica — lidera o ROC-AUC, sobreajusta menos, calibra melhor e é
a única que responde "por quê" em unidades que um gestor entende.

---

## 12. Interpretabilidade

Três lentes, porque cada uma responde a uma pergunta diferente (`notebooks/03`, §9). Os valores
são **para a classe de risco**: OR abaixo de 1 significa *reduz* o risco.

| | Coeficientes (OR de risco) | Permutation Importance (queda no PR-AUC) | SHAP (média absoluta) |
|---|---|---|---|
| 1º | `sigla_uf_CE` — **0,006** | `sigla_uf` — **0,345** | `media_inse` — 0,816 |
| 2º | `sigla_uf_BA` — 11,68 | `media_inse` — 0,137 | `ideb_2021` — 0,664 |
| 3º | `media_inse` — 0,391 | `ideb_2021` — 0,103 | `sigla_uf_RS` — 0,468 |

As três apontam o mesmo conjunto: **território, condição socioeconômica e desempenho pregresso.**
A ordem interna difere porque medem coisas distintas — o coeficiente do Ceará é o maior de todos,
mas o SHAP mede contribuição **média por predição** e aquela dummy só se ativa em 184 municípios;
a *permutation* trata `sigla_uf` como bloco único, somando as 24 dummies.

### 12.1 O mesmo efeito em pontos percentuais

Odds ratio é a unidade estatisticamente correta e a errada para uma reunião. `notebooks/04`, §1
traduz: **quantos pontos percentuais de risco** a variável move quando melhora em um
desvio-padrão.

| Variável | Efeito no risco | Natureza |
|---|---:|---|
| `media_inse` | **−9,2 p.p.** | condição estrutural |
| `ideb_2021` | **−7,5 p.p.** | condição estrutural |
| `afd_ai_grupo_1` (formação adequada) | **−3,0 p.p.** | alavanca escolar |
| `afd_ai_grupo_3` | **−2,1 p.p.** | alavanca escolar |

A separação entre **o que o município recebe** e **o que o município faz** é o que torna a tabela
acionável: as alavancas sob gestão municipal valem cerca de um terço do que valem as condições
herdadas — e a UF, que não aparece aqui por não ser contínua, vale mais que todas.

---

## 13. Do escore à decisão: o limiar por cobertura

O corte de 0,5 é o padrão do software, não uma decisão de política. Quem prioriza município
define primeiro **quanta cobertura quer** — *"não quero deixar de fora mais de 20% dos municípios
em risco"* — e o limiar sai daí (`src/evaluation/metricas.py::limiar_por_recall`).

Sobre os 5.448 municípios do ciclo (`notebooks/04`, §2):

| Cobertura desejada | Limiar | Municípios sinalizados | Precisão |
|---:|---:|---:|---:|
| 60% | — | 1.166 | 0,77 |
| 70% | — | 1.460 | 0,71 |
| **80%** | **0,286** | **1.861** | **0,64** |
| 90% | — | 2.415 | 0,55 |

A tabela põe preço na decisão: cobrir mais custa precisão, e precisão menor significa mais
visitas técnicas a quem não precisava. **É uma escolha de política, e o modelo apenas mostra o
câmbio.**

Adotando 80% de cobertura: **1.861 municípios prioritários**, com taxa média observada de 46,6%,
contra 3.587 em acompanhamento, com 71,2%.

---

## 14. Hipóteses e respostas às perguntas de negócio

### 14.1 As cinco hipóteses da EDA

Registradas **antes** de o modelo existir.

| # | Hipótese | Veredito |
|---|---|---|
| H1 | Desempenho pregresso é o melhor preditor disponível | **Parcialmente** — `ideb_2021` (OR 0,42) é o segundo maior efeito contínuo, atrás do INSE |
| H2 | Efeito socioeconômico é muito menor que a correlação bruta sugere | **Refutada em magnitude, confirmada no mecanismo** (ver abaixo) |
| H3 | Qualificação docente tem efeito próprio | **Confirmada** — `afd_ai_grupo_1`, OR 0,76: mais docentes com formação adequada reduzem o risco |
| H4 | Há efeito de gestão estadual não capturado pelas variáveis | **Fortemente confirmada** — `sigla_uf_CE` com OR **0,006**, e `sigla_uf` domina a permutation importance |
| H5 | Municípios rurais e pequenos em desvantagem | **Parcialmente** — `capital_desc_interior` OR 1,23 confirma; `proporcao_rural` (OR 1,02) é praticamente nulo depois de controlar o resto |

**Sobre H2 — e é uma reversão em relação ao relatório da abordagem anterior.** Havíamos
registrado a expectativa de que controlar por UF *encolheria* o efeito do INSE. Ele não só não
encolheu como **passou a ser o maior efeito contínuo do modelo** (OR 0,391, à frente do IDEB).

A previsão errou porque comparava quantidades diferentes: a EDA mediu uma correlação
**bivariada** (0,29), confundida com a região; o coeficiente do modelo é **parcial**, condicionado
a UF e a mais trinta variáveis. Remover a variação *entre* regiões — que incluía o gradiente
negativo do Nordeste — limpa o sinal em vez de reduzi-lo. É efeito de **supressão**.

O mecanismo da hipótese (Paradoxo de Simpson) segue confirmado, e a conclusão de política também:
o nível socioeconômico pesa, mas **a UF pesa mais** — 2,5× o INSE na permutation importance. Não
é a renda das famílias que separa os municípios brasileiros em primeiro lugar.

### 14.2 As cinco perguntas do enunciado

Todas respondidas pelo **mesmo classificador supervisionado** (`notebooks/04`). Não há um segundo
modelo por trás de nenhuma delas — a versão anterior deste projeto usava K-means para a pergunta 3,
e ele foi removido: o enunciado pede aprendizado supervisionado, e o próprio classificador
responde melhor.

| # | Pergunta | Resposta |
|---|---|---|
| 1 e 5 | Quais fatores mais impactam / têm maior influência? | **A unidade federativa** (queda de 0,345 no PR-AUC ao ser embaralhada), seguida da **condição socioeconômica** (0,137), do **desempenho pregresso** (0,103) e, uma ordem de grandeza abaixo, das **alavancas escolares** |
| 2 | Quais municípios apresentam maior risco? | **A saída nativa do modelo.** `P(em risco)` por município, ordenável, em [`ranking_risco_municipios.csv`](ranking_risco_municipios.csv). O corte é a decisão de cobertura da §13 |
| 3 | Quais regiões têm padrões semelhantes? | **Dois mecanismos e três patamares.** Norte (52,8%) e Nordeste (45,8%) compartilham patamar *e* motores (IDEB pregresso + INSE); Sul (22,0%), Sudeste (9,4%) e Centro-Oeste (7,9%) compartilham os motores (composição do INSE), com o Sul num patamar à parte |
| 4 | Como prever quem não atingirá as metas? | **Risco previsto × meta pactuada.** 43,4% já cumpriram em 2024 a meta de 2025; dos restantes, **1.575** estão abaixo *sem* sinal de risco estrutural e **1.455** *com* sinal |

### 14.3 As duas leituras que mudam a recomendação

**Pergunta 3 — os padrões atravessam as fronteiras.** Norte e Nordeste têm o mesmo problema no
modelo, ainda que sejam regiões diferentes no mapa. E, descendo do agregado, há municípios de
perfil "Norte/Nordeste" no interior do Sudeste e municípios de perfil "Sudeste" dentro do
Nordeste. **Um programa desenhado por região erraria o alvo:** a focalização eficiente é por
perfil de risco, e o ranking da pergunta 2 já entrega essa lista pronta, sem fronteira nenhuma.

**Pergunta 4 — "abaixo da meta" não é um diagnóstico.** Separar os municípios pelo sinal do
modelo transforma um número em decisão de alocação:

| Situação | Municípios | Diagnóstico | Resposta |
|---|---:|---|---|
| Meta já atingida | 2.322 | — | monitoramento |
| Abaixo da meta, **sem** sinal de risco estrutural | 1.575 | as condições comportam a meta; falta execução | **apoio técnico**, retorno rápido |
| Abaixo da meta, **com** sinal de risco | 1.455 | as condições não sustentam a meta | **investimento estruturante**, retorno em anos |

Cobrar resultado do terceiro grupo sem mudar as condições é cobrar o impossível.

---

## 15. Limitações

1. **Falácia ecológica** — o modelo descreve o município, não uma criança. Duas crianças do mesmo
   município recebem a mesma predição. É consequência direta da ausência de microdado de aluno
   (§1.1), não uma escolha de modelagem.
2. **Ausência de variáveis de política educacional** — a lacuna mais séria: o maior efeito do
   modelo (a UF) é justamente o que ele **não consegue explicar**. Nada na base descreve
   formação continuada, material estruturado, avaliação diagnóstica ou regime de colaboração.
3. **Um único ciclo modelado** — não distingue tendência de choque, e o ciclo escolhido contém o
   choque do Rio Grande do Sul: o risco previsto para o estado está **superestimado**.
4. **Cobertura incompleta** — 25 das 27 UFs; DF sem rede municipal, Roraima ausente da fonte.
5. **INSE de 2023 replicado para 2024** como atributo estrutural do município.
6. **AFD/ATU/IED usam o agregado `Total`**, que inclui a rede privada — medem o contexto
   educacional do município, não a rede municipal especificamente.
7. **Não projeta ciclos futuros** — o modelo descreve a estrutura de 2024. Projetar 2025 exigiria
   as características de 2025, que ainda não existem.
8. **Associações condicionais, não efeitos causais.** Nenhuma intervenção pode ser justificada
   apenas por estes coeficientes.
9. **O ranking não avalia gestão.** Ele ordena municípios pelo risco que as condições observadas
   produzem. Um município no topo pode ter gestão excelente enfrentando condições muito adversas.

---

## 16. Decisões revistas durante o projeto

Registradas porque o processo importa tanto quanto o resultado — e porque cada uma foi corrigida
por uma verificação, não por opinião.

| Afirmação inicial | O que a verificação mostrou |
|---|---|
| "As proporções por nível somam exatamente para a taxa" | **Não somam** — o corte de 743 pontos cai *dentro* de um nível. A melhor aproximação é a soma dos níveis 5–8 (r = 0,986, erro médio 5,9 p.p.). Continua vazamento severo, por proximidade da medição e não por identidade |
| "Controlada a região, a relação INSE × taxa é crescente em todas" | **Falso.** Com decis intrarregionais: fraca no Sudeste (0,01) e no Sul (0,04), **negativa** no Nordeste (−0,14). A narrativa correta — confundimento geográfico — é mais forte que a errada |
| "Controlar por UF encolherá o efeito do INSE" | **Cresceu** — o INSE passou a ser o maior efeito contínuo do modelo. Efeito de supressão; ver §14.1 |
| "`meta_alfabetizacao_2030` é 100 para todo município" | É **80** para toda a rede municipal |
| "Cobertura idêntica nos dois ciclos" | O *número* de municípios é idêntico (5.448), o *conjunto* não: 5.500 distintos, 5.396 nos dois, 104 entram ou saem |
| "~850 mil registros no Bronze" | **719.757** |
| "O IDEB pregresso é o maior efeito contínuo" | É o **INSE** (−9,2 p.p. contra −7,5). O IDEB liderava na formulação anterior do alvo, não nesta |
| "O Sul tem motores de risco próprios" | Sul, Sudeste e Centro-Oeste compartilham os motores; o que separa o Sul é o **patamar** |
| "`class_weight='balanced'` é necessário com classe minoritária" | Não altera o ROC-AUC e **piora a calibração em 2,5×**; ver §10.1 |

### 16.1 A mudança de abordagem, e o que ela custou

O projeto passou por duas formulações do alvo. A primeira mantinha o grão de aluno via **dado
binário agrupado** (cada município virava duas observações ponderadas); a segunda, adotada,
classifica o **município**. A troca simplificou a interpretação, devolveu o ferramental padrão do
Scikit-learn e fez a saída do modelo responder diretamente à pergunta 2 do enunciado.

**O que se perdeu merece registro.** A formulação anterior permitia um **teste de estabilidade do
ranking** entre os dois ciclos: a lista dos 300 municípios com pior resíduo em 2023 tinha apenas
**17,3% de sobreposição** com a de 2024. Era o achado metodológico mais forte do projeto — a
prova de que uma lista de "pior gestão" mediria ruído — e **não sobrevive ao recorte de um ciclo
só**. A cautela que ele fundamentava permanece como limitação declarada (§15, item 9), agora
apoiada em argumento e não em número.

---

## 17. Reprodutibilidade

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt

python -m src.preprocessing.run_pipeline   # reconstrói o lake (Bronze → Silver → Gold)
python -m src.modeling.model_card           # regenera reports/model_card.json
# notebooks: 01 → 02 → 03 → 04
```

- `random_state = 42` em toda parte; `src/preprocessing/config.py` centraliza caminhos, mapas de
  domínio e a lista de vazamento.
- A pipeline é **idempotente** — verificada por hash de conteúdo por partição.
- Nenhuma etapa exige credenciais, nuvem ou acesso à internet.
- `reports/model_card.json` registra a proveniência e as métricas de cada execução.
