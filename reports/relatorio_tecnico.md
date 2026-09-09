# Relatório Técnico — Predição de Risco de Alfabetização nos Municípios Brasileiros

**Tech Challenge Fase 3 · POSTECH AI Scientist**

Consolida as **decisões analíticas e a metodologia**. O [README](../README.md) traz a narrativa e
a execução; os [notebooks](../notebooks/) trazem os resultados; este documento traz o *porquê* de
cada escolha e o que foi descartado.

**Nenhum número aqui foi digitado à mão** — todos vêm de `model_card.json`
(`python -m src.modeling.model_card`) ou de células executadas, com a referência indicada.

---

## 1. Problema e unidade de análise

O **Compromisso Nacional Criança Alfabetizada** pactua metas por município, aferidas pelo Saeb ao
fim do 2º ano com corte de 743 pontos. O enunciado pede um modelo supervisionado que preveja se
**um aluno** será considerado alfabetizado.

### 1.1 Não existe microdado de aluno

| Fonte | Grão mais fino | Identificador de aluno? |
|---|---|---|
| Indicador Criança Alfabetizada (INEP) | município × ano × série × rede | não |
| Censo Escolar — AFD, ATU, IED | município × ano × localização × dependência | não |
| Saeb — INSE | município × tipo de rede × localização | não |
| IDEB Anos Iniciais | município × rede × ciclo | não |
| Metas do CNCA | município × rede | não |

Nenhuma traz sexo, idade, trajetória ou frequência — nem chave que permitisse ligá-las no grão do
estudante.

### 1.2 A ponte: modelar o contexto que responde pela criança

O município entra como **em risco** quando `taxa_alfabetizacao < 50%` no ciclo de 2024 — menos da
metade das suas crianças chega alfabetizada. A leitura para a pergunta original é direta: *uma
criança que estuda ali tem chance substancialmente menor de ser considerada alfabetizada*.

**Por que este corte.** É absoluto e interpretável sem contexto estatístico — um gestor entende na
hora — e cai numa região densa da distribuição, sem isolar uma cauda residual. Duas alternativas
foram descartadas:

| Alternativa | Por que saiu |
|---|---|
| A **meta pactuada** de cada município | é calculada a partir da taxa de 2023 do próprio município: o alvo viraria função da própria história, e a meta, um preditor circular |
| A **média nacional do ano** | é relativa — metade do país está sempre em risco por construção, melhore o país ou piore |

Resultado: **1.464 dos 5.396 municípios em risco — 27,1%.** Classe minoritária, o que define as
métricas da §6.

**A limitação inseparável é a falácia ecológica:** o modelo descreve o município, não uma criança
específica. Duas crianças do mesmo lugar recebem a mesma predição.

### 1.3 Histórico e atualidade

| Natureza | Variáveis | Por que é legítima |
|---|---|---|
| **Histórico** | `taxa_2023`, `media_portugues_2023`, `ideb_2021`, `taxa_aprovacao_2021` | anteriores ao ciclo previsto |
| **Atualidade** | INSE (média, 7 níveis, ruralidade, porte), AFD, IED, ATU, UF, capital/interior | descrevem o município, não o resultado da prova |

**Painel:** os **5.396** municípios de 2024 que têm histórico de 2023, de 5.448 no ciclo. Os 52
restantes (majoritariamente do Acre) só existem em 2024.

### 1.4 Um único ciclo modelado

O Rio Grande do Sul caiu **20,2 p.p.** entre 2023 e 2024 (§4, A3), um choque exógeno que nenhuma
variável descreve. Empilhar os dois ciclos como amostras do mesmo processo assumiria uma
comparabilidade que não existe.

**A contrapartida é declarada:** para o RS o modelo vê um histórico bom (pré-choque) e um
resultado ruim (pós-choque), e lê a diferença como estrutura. O efeito é absorvido pela dummy da
UF (*odds ratio* 14,98), e **o risco previsto para o estado é pessimista demais**.

---

## 2. Origem dos dados

### 2.1 Pipeline medalhão reconstruída

A Fase 2 entregou Bronze → Silver → Gold em AWS Glue + S3 + Athena. A Fase 3 **reproduz em
Python/pandas, sem nuvem**, preservando schema explícito, hash de deduplicação, DQ com quarentena,
particionamento Hive-style e escrita idempotente.

**Por que reconstruir em vez de consumir o Parquet pronto?** Porque a Gold da Fase 2 foi desenhada
para análise descritiva e suas colunas mais informativas são as que vazam o alvo (§3); porque as
fontes externas precisavam entrar **na Silver e na Gold**, sob as mesmas regras; e porque assim
qualquer avaliador reproduz tudo com um comando, sem credenciais.

| Camada | Resultado (`notebooks/01`) |
|---|---|
| Bronze | 10 entidades, **719.757 registros**, DQ 100% |
| Silver | **54.165 duplicatas removidas**, **890 em quarentena** |
| Gold | 5 visões — as 4 da Fase 2 + `base_ml_alfabetizacao` |
| Idempotência | 29/29, 30/30 e 10/10 partições idênticas em duas execuções |

Sem o enriquecimento externo, a maior correlação disponível com o alvo seria a de `ano` (0,06).
Com ele, 0,53.

### 2.2 Qualidade de dados — achados e tratamento

| # | Achado | Tratamento |
|---|---|---|
| 1 | **O arquivo do INSE repete a mesma chave até 7 vezes** (125.741 linhas, 71.576 hashes) | ingerido como chega na Bronze, deduplicado na Silver. Sem isso o join multiplicaria a base por até 7 |
| 2 | 890 estratos sem `media_inse` (sigilo estatístico) | quarentena com motivo registrado |
| 3 | Percentual de nível do INSE **sem alunos vem em branco, não zero** | verificado que, tratando branco como zero, os 8 níveis somam 100% em todas as linhas → zero estrutural |
| 4 | Cabeçalhos multi-nível, nulos como `"--"`, `CO_MUNICIPIO` como float | `skiprows`, `na_values`, `zfill(7)` antes de qualquer join |
| 5 | Chave de rede divergente (código vs. texto) | normalização via `REDE_MAP` |
| 6 | **Roraima ausente do arquivo do INEP**; DF não tem rede municipal | limitação de cobertura: 25 das 27 UFs |

---

## 3. Tratamento de data leakage

É o núcleo metodológico. O arquivo do INEP traz colunas que parecem excelentes preditoras e são
**a mesma medição que gerou o alvo**.

### 3.1 Medido, não presumido

| Coluna | Por que vaza | Correlação |
|---|---|---:|
| `media_portugues` (2024) | mesma escala Saeb da qual a taxa é o % acima de 743 pontos | **0,927** |
| `proporcao_aluno_nivel_*` | distribuição de proficiência da qual a taxa deriva | **0,986** (níveis 5-8) |
| `meta_alfabetizacao_2025` | calculada a partir da taxa de 2023 | **0,966** |
| `nivel_alfabetizacao` | a própria taxa discretizada | — |
| IDEB dos ciclos 2023 e 2025 | contemporâneo e posterior ao alvo | — |

### 3.2 O critério

Não é a força da correlação. É: **esta informação estaria disponível no momento da predição?**

Daí a decisão que estrutura o projeto: **vazamento é contemporâneo, não histórico.**
`media_portugues` de **2024** é vazamento e sai; a de **2023** é um preditor honesto — e acabou
sendo a variável contínua mais importante do modelo. O IDEB de 2021 (r = 0,54) fica pelo mesmo
motivo.

### 3.3 Onde é aplicado

Em `config.COLUNAS_VAZAMENTO` — **23 colunas**, cada uma com o motivo — removidas por
`gold.montar_base_ml()`. A decisão fica auditável no repositório, e um `assert` no `notebooks/01`
verifica que nenhuma sobreviveu.

A **meta pactuada** também não entra: ela reaparece no `notebooks/04` §5, mas só como régua de
comparação **depois** da predição.

### 3.4 O segundo vazamento, mais sutil

Mediana da imputação, média e desvio da padronização e categorias do encoder são calculadas
**dentro do `Pipeline`**, portanto só com o fold de treino. Padronizar antes de separar os folds
usaria informação da validação — e não geraria erro nenhum.

---

## 4. Análise exploratória — os achados que decidiram a modelagem

`notebooks/02`, em 9 etapas.

| # | Achado | Consequência |
|---|---|---|
| A1 | Alvo aproximadamente simétrico, ~1% em 100% e ~1,5% abaixo de 20% | o corte em 50% cai em região densa; sem separação perfeita |
| A2 | Dispersão dentro de cada região é enorme; amplitude de ~23 p.p. entre regiões | métricas de ordenação e calibração, não acurácia |
| A3 | **O RS caiu 20,2 p.p.**, com 89,6% dos municípios em queda | modelar um ciclo; risco do RS declarado como pessimista |
| A4 | Sem enriquecimento, correlação máxima 0,06; com ele, 0,53 | o enriquecimento não era opcional |
| A5 | **Paradoxo de Simpson no INSE:** agregado 0,29; Norte 0,31, Centro-Oeste 0,20, Sul 0,04, Sudeste 0,01, **Nordeste −0,14** | `sigla_uf` como controle **obrigatório** |
| A6 | Ceará e Pernambuco: mesmo INSE (4,37), taxas de 90,1% e 63,0% | hipótese de efeito de política estadual (H4) |
| A7 | AFD, IED e níveis do INSE são **composicionais** (somam 100%) | descartar uma categoria de referência por bloco |
| A8 | No bloco IDEB, `ideb`/notas correlacionam 0,953-0,961; rendimento e aprovação, **0,996** | manter só `ideb_2021` e `taxa_aprovacao_2021` |
| A9 | Escalas de 3 a 80.160 contra 4 a 6 | padronização obrigatória |
| A10 | Mesmo município correlaciona **0,64** entre os ciclos | justifica o bloco histórico **e** exige uma linha por município |

---

## 5. Seleção de variáveis e pré-processamento

`src/modeling/features.py`. **28 colunas de entrada → 53 features após o pré-processamento.**

| Bloco | Variáveis | Referência descartada |
|---|---:|---|
| Histórico | 4 | — |
| Formação docente (AFD) | 4 | `afd_ai_grupo_5` (sem curso superior) |
| Esforço docente (IED) | 5 | `ied_ai_nivel_1` (menor esforço) |
| Níveis do INSE | 7 | `inse_pc_nivel_1` (nível mais baixo) |
| Tamanho de turma (ATU) | 5 | — (médias, não composicional) |
| Socioeconômico — outras | 3 | — |
| Categóricas | 2 | *dummy encoding* (`drop="first"`) |

Fora do modelo: **`regiao`**, função determinística de `sigla_uf` (as dummies já a codificam), e
**`ano`**, constante no ciclo modelado.

Três ramos no `ColumnTransformer`: numéricas comuns (mediana → `StandardScaler`); **IDEB e
aprovação com `add_indicator=True`**, porque 13,3% e 5,0% de ausência concentrada em municípios
pequenos e isolados significa que **a ausência é sinal**; e categóricas com moda →
`OneHotEncoder(drop="first", handle_unknown="ignore")`.

> **Nota sobre normalização** (aula de classificação): árvores e Naive Bayes não precisam;
> Regressão Logística e SVM precisam. Como todos passam pelo **mesmo** `Pipeline`, padronizar não
> prejudica os primeiros, garante os segundos e mantém a comparação justa.

---

## 6. Validação e métricas

**`train_test_split` estratificado 75/25** — 4.047 municípios de treino e **1.349 de teste,
tocados uma única vez**. Sobre o treino, **`StratifiedKFold` de 5 folds** para comparar algoritmos
e ajustar hiperparâmetros, sempre com `return_train_score=True`.

Estratificado nas duas etapas porque a classe de risco é minoritária: um sorteio simples poderia
produzir partições com prevalências diferentes e as métricas ficariam instáveis por acidente.

| Métrica | Por que está aqui |
|---|---|
| **Recall da classe de risco** | a prioritária. Um falso alarme custa uma visita técnica; um falso negativo custa uma geração |
| **PR-AUC** | mais informativa que a ROC com classe minoritária — não se infla com os verdadeiros negativos |
| **ROC-AUC** | capacidade de ordenar municípios por risco |
| **Curva de calibração** | permite ler a saída como *probabilidade*, e é o que autoriza o corte por cobertura da §9 |
| *Acurácia* | reportada **sempre ao lado do baseline** de 72,9% |

---

## 7. Comparação de algoritmos

Mesmo pré-processamento, mesma validação, mesma partição — só o estimador muda (`notebooks/03`,
Etapa 9; números em `model_card.json`). Os quatro da aula de classificação supervisionada mais os
dois ensembles da aula de otimização.

| Modelo | ROC-AUC | PR-AUC | Recall | Precisão | Gap treino-val. | Tempo (s) |
|---|---:|---:|---:|---:|---:|---:|
| **Regressão Logística** | **0,9146** | 0,8093 | 0,666 | 0,763 | **0,0084** | 1,0 |
| Gradient Boosting | 0,9117 | **0,8129** | 0,664 | 0,765 | 0,0607 | 1,8 |
| Random Forest | 0,9096 | 0,8095 | 0,585 | 0,795 | 0,0587 | 0,8 |
| SVM (RBF) | 0,9010 | 0,7991 | 0,604 | 0,810 | 0,0536 | 2,6 |
| Árvore de Decisão | 0,8791 | 0,7410 | 0,595 | 0,730 | 0,0388 | 0,9 |
| Naive Bayes | 0,8656 | 0,6739 | **0,926** | 0,447 | 0,0062 | 0,1 |
| Baseline (classe majoritária) | 0,5000 | 0,2713 | 0,000 | 0,000 | 0,0000 | 3,5 |

**Leitura:**

1. **A Regressão Logística lidera o ROC-AUC** e fica a 0,0036 do melhor PR-AUC — dentro do desvio
   entre folds (0,026). E **sobreajusta sete vezes menos** que os ensembles. A escolha se sustenta
   em resultado medido, não em preferência por simplicidade.
2. **O Naive Bayes tem recall 0,926 e precisão 0,447**: dispara alarme para quase todo mundo. É o
   efeito esperado da premissa de independência aplicada a três blocos composicionais, e confirma
   empiricamente o descarte que estaria só em prosa.
3. **O baseline acerta 72,9% sem prever ninguém em risco** — a prova de que acurácia sozinha não
   serve como critério.

### 7.1 Contribuição de cada bloco de variáveis

| Bloco | ROC-AUC | PR-AUC |
|---|---:|---:|
| Só histórico | 0,9128 | **0,8109** |
| Só atualidade | 0,8887 | 0,7587 |
| Histórico + atualidade | **0,9146** | 0,8093 |

**O histórico carrega quase todo o sinal preditivo** — sozinho ele empata com o modelo completo (e
até o supera marginalmente no PR-AUC, dentro do ruído). Isso é esperado: desempenho educacional
tem forte inércia (A10), e o passado do próprio município resume boa parte do que as variáveis de
contexto tentam capturar.

**A conclusão prática é a oposta de "remover o bloco de atualidade".** Ele é o único que responde
*o que fazer*: o histórico diz que o município vai mal, mas só o contexto diz que a formação
docente ou o tamanho da turma são as alavancas. Um modelo só com histórico prevê bem e não
recomenda nada.

### 7.2 Otimização de hiperparâmetros

`GridSearchCV` sobre `C` ∈ {0,01; 0,1; 1; 10; 100}, `scoring="average_precision"`:

| C | PR-AUC validação | PR-AUC treino | Gap |
|---:|---:|---:|---:|
| 0,01 | 0,7664 | 0,7743 | 0,0079 |
| 0,1 | 0,8065 | 0,8224 | 0,0159 |
| **1,0** | **0,8093** | 0,8294 | 0,0201 |
| 10 | 0,8074 | 0,8301 | 0,0227 |
| 100 | 0,8067 | 0,8302 | 0,0236 |

**`C = 1,0`**, com gap muito abaixo do limiar de alerta de 0,05 usado na aula. A *validation
curve* mostra treino e validação caminhando coladas em todo o intervalo.

**Conclusão honesta: não havia overfitting para resolver.** Com 4.047 municípios de treino para 53
features num modelo linear, a penalidade L2 é salvaguarda contra colinearidade residual, não
remédio para sobreajuste. Note também que o desvio entre folds (0,026) é **maior** que a diferença
entre `C = 0,1` e `C = 100`.

### 7.3 Avaliação final no teste

| Métrica | Valor |
|---|---:|
| ROC-AUC | **0,8978** |
| PR-AUC | **0,7535** |
| Recall (risco) | 0,6230 |
| Precisão (risco) | 0,7575 |
| Acurácia | 0,8436 |
| *Acurácia do baseline* | *0,7287* |

Matriz de confusão: 910 verdadeiros negativos, 228 verdadeiros positivos, **138 falsos negativos**
(o erro caro — municípios em risco que ficaram fora da lista) e 73 falsos positivos.

Praticamente o mesmo da validação cruzada: não houve sobreajuste à partição.

### 7.4 Calibração

Desvio médio absoluto de **0,032** entre previsto e observado por decil, com leve subestimação do
risco na faixa intermediária. É calibração suficiente para ler a saída como probabilidade — e é
isso que autoriza compará-la diretamente com a meta pactuada.

---

## 8. Interpretabilidade

Três lentes (`notebooks/03`, Etapa 16), porque respondem a perguntas diferentes.

| Lente | Primeiro colocado | Segundo | Terceiro |
|---|---|---|---|
| Coeficientes (log-odds) | `sigla_uf_RS` +2,71 | `sigla_uf_GO` −2,07 | `sigla_uf_BA` +1,94 |
| Permutation importance | `sigla_uf` **0,211** | `media_portugues_2023` 0,078 | `ideb_2021` 0,046 |
| SHAP (média absoluta) | dummies de UF | `media_portugues_2023` | `ideb_2021` |

**As três convergem.** Depois de controlar tudo o que se consegue medir, **a UF é o maior efeito
do modelo** — embaralhá-la derruba o PR-AUC quase três vezes mais que a segunda colocada.

Ceará (*odds ratio* de risco **0,19**) e Bahia (**6,96**) são vizinhos, da mesma região e com
condições socioeconômicas próximas, em polos opostos. O Rio Grande do Sul aparece no extremo de
risco (14,98) por um motivo conhecido: as enchentes de 2024.

### 8.1 Efeitos marginais, em pontos percentuais

| Variável | Efeito no risco | Natureza |
|---|---:|---|
| `media_portugues_2023` | **−6,1 p.p.** | condição herdada |
| `ideb_2021` | **−5,5 p.p.** | condição herdada |
| `media_inse` | **−5,3 p.p.** | condição herdada |
| `taxa_2023` | −4,4 p.p. | condição herdada |
| `afd_ai_grupo_1` (formação adequada) | **−2,4 p.p.** | **alavanca escolar** |
| `afd_ai_grupo_3` | −1,7 p.p. | **alavanca escolar** |

A separação entre o que o município **recebe** e o que ele **faz** é o que torna a tabela
acionável: as alavancas sob gestão municipal valem menos da metade das condições herdadas — e a
UF, que não aparece por não ser contínua, vale mais que todas.

---

## 9. Do escore à decisão: o limiar por cobertura

O corte de 0,5 é convenção do software. Quem prioriza município define primeiro **quanta cobertura
quer** — *"não quero perder mais de 20% dos municípios em risco"* — e o limiar sai daí
(`evaluation.limiar_por_recall`). Sobre o conjunto de teste:

| Cobertura desejada | Limiar | Municípios sinalizados | Precisão |
|---:|---:|---:|---:|
| 60% | 0,542 | 285 | 0,77 |
| 70% | 0,393 | 362 | 0,71 |
| **80%** | **0,258** | **456** | **0,64** |
| 90% | 0,146 | 596 | 0,55 |

Cobrir mais custa precisão, e precisão menor significa mais visitas técnicas a quem não precisava.
**É uma escolha de política, e o modelo apenas mostra o câmbio.**

Aplicado aos 5.396 municípios: **1.711 prioritários**, com taxa média observada de 44,7%, contra
3.685 em acompanhamento, com 71,3%.

---

## 10. Hipóteses — veredito

Registradas na EDA **antes** de o modelo existir.

| # | Hipótese | Veredito |
|---|---|---|
| H1 | O desempenho pregresso é o melhor preditor disponível | **Confirmada** — o bloco histórico sozinho atinge PR-AUC 0,811, contra 0,809 do modelo completo |
| H2 | O efeito socioeconômico é menor do que a correlação bruta sugere | **Confirmada** — `media_inse` vale −5,3 p.p., atrás de duas variáveis de histórico |
| H3 | A qualificação docente tem efeito próprio | **Confirmada, fraca** — `afd_ai_grupo_1`, −2,4 p.p. |
| H4 | Há efeito de gestão estadual que as variáveis não capturam | **Fortemente confirmada** — `sigla_uf` domina a permutation importance com 0,211 |
| H5 | Municípios rurais e pequenos em desvantagem | **Parcialmente** — `capital_desc_interior` confirma; `proporcao_rural` é quase nulo depois dos controles |

---

## 11. Respostas às perguntas de negócio

Todas pelo **mesmo classificador supervisionado** (`notebooks/04`).

| # | Pergunta | Resposta |
|---|---|---|
| 1 e 5 | Fatores de maior impacto / influência | **UF** (0,211 na permutation importance), **desempenho pregresso** (−6,1 e −5,5 p.p.), **condição socioeconômica** (−5,3) e, por último, as **alavancas escolares** (−2,4) |
| 2 | Municípios de maior risco | A saída nativa do modelo, em `ranking_risco_municipios.csv`. Com 80% de cobertura: **1.711 prioritários** |
| 3 | Regiões com padrões semelhantes | **Norte (52,7%) e Nordeste (45,5%) formam um par** — mesmo patamar e mesmos motores. Sul (21,7%), Sudeste (9,5%) e Centro-Oeste (8,5%) ficam no patamar baixo, sem gargalo comum |
| 4 | Quem não atingirá as metas | **2.314** já atingiram; **1.692** estão abaixo sem sinal de risco; **1.296** abaixo com sinal |

### 11.1 As duas leituras que mudam a recomendação

**Pergunta 3.** Onde o resultado é baixo (Norte e Nordeste), o gargalo é o mesmo: desempenho
pregresso e nível socioeconômico. Onde já é alto, **não há obstáculo comum** — as restrições viram
locais. E os perfis atravessam as fronteiras: há municípios de perfil nordestino no interior do
Sudeste. **Focalizar por perfil, não por território.**

**Pergunta 4.** "Abaixo da meta" não é diagnóstico. Separado pelo que o modelo prevê, vira decisão
de alocação: quem está abaixo **sem** sinal de risco precisa de **apoio à execução**, com retorno
rápido; quem está abaixo **com** sinal precisa de **investimento estruturante**. Cobrar resultado
do segundo grupo sem mudar as condições é cobrar o impossível.

---

## 12. Limitações

1. **Falácia ecológica** — o modelo descreve o município, não uma criança. Consequência direta da
   ausência de microdado (§1.1), não escolha de modelagem.
2. **Sem variáveis de política educacional** — o maior efeito do modelo (a UF) é o que ele não
   consegue explicar.
3. **Um único ciclo modelado**, e ele contém o choque do RS: o risco previsto para o estado é
   pessimista demais.
4. **Cobertura incompleta** — 25 das 27 UFs; 5.396 dos 5.448 municípios do ciclo.
5. **INSE de 2023 replicado para 2024**; AFD/ATU/IED usam o agregado `Total`, que inclui a rede
   privada.
6. **Não projeta ciclos futuros** — descreve a estrutura de 2024.
7. **Associação, não causalidade.**
8. **A lista não avalia gestão** — ordena municípios pelas condições observadas.

---

## 13. Decisões revistas durante o projeto

Cada uma foi corrigida por uma verificação, não por opinião.

| Afirmação inicial | O que a verificação mostrou |
|---|---|
| "As proporções por nível somam exatamente para a taxa" | **Não somam** — o corte de 743 cai *dentro* de um nível; erro médio de 5,9 p.p. Continua vazamento severo, por proximidade da medição e não por identidade |
| "Controlada a região, a relação INSE × taxa é crescente em todas" | **Falso.** Fraca no Sudeste (0,01) e no Sul (0,04), **negativa** no Nordeste (−0,14) |
| "`meta_alfabetizacao_2030` é 100 para todo município" | É **80** |
| "Cobertura idêntica nos dois ciclos" | O *número* é idêntico (5.448), o *conjunto* não: 5.500 distintos, 5.396 nos dois |
| "~850 mil registros no Bronze" | **719.757** |
| "`media_portugues` deve ser excluída em qualquer forma" | Só a **contemporânea**. Defasada, virou a variável contínua mais importante do modelo — a distinção reorganizou o projeto |
| "A primeira divisão da árvore é a taxa de 2023" | É a **nota de Português de 2023**; e o segundo nível já usa dummies de UF |
| "`GridSearchCV` não funciona com alvo ponderado" | **Funciona** — bastava declarar `set_fit_request` em cada etapa aninhada. O obstáculo era falta de configuração, não limitação da biblioteca |

### 13.1 As formulações que ficaram pelo caminho

O projeto testou três formulações do alvo antes de fixar esta:

| Versão | Alvo | Resultado | Por que saiu |
|---|---|---|---|
| v1 | criança via dado binário agrupado, 2 ciclos, sem histórico | AUC 0,661 (67,9% do teto do oráculo) | o split temporal impedia usar o histórico defasado, que é o preditor mais forte |
| v2 | município, sem histórico | ROC-AUC 0,880 | descartava informação legítima |
| v3 | criança via dado binário agrupado, com histórico | AUC 0,687 (79,1% do teto) | exigia metadata routing e excluía SVM e KNN, que não aceitam `sample_weight` |
| **v4 (atual)** | **município, com histórico** | **ROC-AUC 0,898 no teste** | — |

O que se perdeu na v3 → v4 foi o **teto do oráculo** (0,737), que dava uma referência absoluta ao
AUC. Em troca, todos os algoritmos das aulas couberam na comparação e o notebook passou a seguir a
estrutura da aula de classificação supervisionada sem adaptações.

---

## 14. Reprodutibilidade

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt

python -m src.preprocessing.run_pipeline   # reconstrói o lake
python -m src.modeling.model_card          # regenera reports/model_card.json
# notebooks: 01 → 02 → 03 → 04
```

`random_state = 42` em toda parte; `src/preprocessing/config.py` centraliza caminhos, mapas de
domínio e a lista de vazamento. A pipeline é **idempotente**, verificada por hash de conteúdo por
partição, e nenhuma etapa exige credenciais, nuvem ou internet.
