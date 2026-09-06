# Relatório Técnico — Predição de Alfabetização nos Municípios Brasileiros

**Tech Challenge – Fase 3 | POSTECH AI Scientist**

Este documento consolida as **decisões analíticas e a metodologia** do projeto.

| Documento | Papel |
|---|---|
| [`README.md`](../README.md) | narrativa, resultados e instruções de execução |
| **este relatório** | por que cada decisão foi tomada, e o que foi descartado |
| [`notebooks/`](../notebooks/) | os resultados reproduzíveis, célula a célula |
| [`reports/model_card.json`](model_card.json) | ficha técnica gerada por código a cada execução |

**Nenhum número deste relatório foi digitado à mão.** Todos vêm do
`model_card.json`, gerado por `python -m src.modeling.model_card`, ou de células
executadas nos notebooks, com a referência indicada.

---

## 1. Problema e unidade de análise

O **Compromisso Nacional Criança Alfabetizada** pactua metas de alfabetização por município,
aferidas pelo Saeb/INEP ao fim do 2º ano do ensino fundamental, com o corte de 743 pontos na
escala de Língua Portuguesa. O enunciado pede um modelo supervisionado que preveja se **um aluno**
será considerado alfabetizado.

### 1.1 A ponte entre o grão do dado e a pergunta

Não existem microdados de aluno publicados para o Indicador Criança Alfabetizada — a base é
agregada por município. Havia duas saídas:

| Alternativa | Consequência |
|---|---|
| **Reformular o alvo** para um indicador municipal (ex.: "município abaixo da média nacional") | mais simples, mas responde outra pergunta e cria um alvo **relativo** — metade dos municípios está sempre em risco por construção, e o alvo não se conecta à meta absoluta do CNCA |
| **Manter o grão de aluno** via dado binário agrupado | fiel ao enunciado, alvo absoluto, ao custo de uma construção menos trivial |

**Decisão: manter o grão de aluno.** Cada linha município × ano vira **duas observações** com o
mesmo vetor de características:

```
y = 1  com peso = taxa_alfabetizacao / 100          (fração alfabetizada)
y = 0  com peso = 1 - taxa_alfabetizacao / 100      (fração restante)
```

A verossimilhança resultante é **idêntica** à de uma logística ajustada sobre os alunos
individuais daquele município — é o formato que a regressão logística binomial estima
nativamente. A probabilidade prevista lê-se como *"probabilidade de um aluno daquele município
estar alfabetizado"*.

Conferência da expansão (`notebooks/03`, §1): 10.896 linhas → **21.792 observações**, soma dos
pesos = 10.896, taxa média implícita **61,54%** — idêntica à observada.

**Limitação inseparável desta escolha:** *falácia ecológica*. O modelo estima a probabilidade do
aluno **típico** de um município, não de uma criança específica. Nenhuma característica individual
entra.

### 1.2 Peso por município, não por aluno

Cada município contribui com peso total 1, independentemente do porte. Coerente com o uso
pretendido (priorizar municípios) e com o dado disponível — a base não traz o número de alunos
avaliados. Ponderar pelo porte deslocaria as estimativas para as capitais.

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

### 2.3 Grão da base analítica

**Município × ano, rede municipal, ciclos 2023 e 2024** — 10.896 linhas, 5.500 municípios,
**25 das 27 UFs**.

A rede municipal foi escolhida por ser a única com o mesmo número de municípios nos dois ciclos
(5.448) e por ser a rede sob gestão direta do município. A rede *total* só existe em 2024 e para
398 municípios.

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
anos antes do primeiro ciclo do alvo.

> **Nuance registrada.** A exclusão vale para a coluna **contemporânea**. `media_portugues`
> *defasada* (do ciclo anterior) seria um preditor legítimo. Não a usamos porque, com apenas dois
> ciclos, defasar custaria 2023 inteiro como treino e inviabilizaria a validação temporal — ver
> §11.

### 4.3 Onde o tratamento é aplicado

Em `src/preprocessing/config.py::COLUNAS_VAZAMENTO` — **23 colunas**, cada uma com o motivo
registrado — e removidas programaticamente por `gold.montar_base_ml()`. A decisão fica auditável
no repositório, e um `assert` no `notebooks/01` verifica que nenhuma sobreviveu até a base de
modelagem.

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
| A1 | Alvo aproximadamente simétrico (assimetria ≈ −0,15), sem massa nos extremos (~1% em 100%, ~1,5% abaixo de 20%) | alvo binomial viável; sem risco de separação perfeita |
| A2 | Dispersão **dentro** de cada região é enorme; amplitude entre regiões de ~23 p.p. | métricas de ordenação e calibração, não acurácia |
| A3 | **O RS caiu 20,2 p.p.** de 2023 para 2024, com 89,6% dos municípios em queda — enquanto todas as demais regiões melhoraram | split temporal mantido, mas reportado **com e sem o RS** |
| A4 | Sem enriquecimento externo, a maior correlação disponível seria a de `ano` (0,06); com ele, 0,53 | o enriquecimento não era opcional |
| A5 | **Paradoxo de Simpson no INSE:** Spearman agregado 0,29; dentro das regiões — Norte 0,31, Centro-Oeste 0,20, Sul 0,04, Sudeste 0,01, **Nordeste −0,14** | `sigla_uf` como controle **obrigatório** |
| A6 | Ceará e Pernambuco: mesmo INSE (4,37), taxas de 90,1% e 63,0% | hipótese de efeito de política estadual (H4) |
| A7 | AFD, IED e níveis do INSE são **composicionais** (somam 100%) | descartar uma categoria de referência por bloco |
| A8 | No bloco IDEB, `ideb`/notas correlacionam 0,953–0,961 e `indicador_rendimento`/`taxa_aprovacao` correlacionam **0,996** | manter só `ideb_2021` e `taxa_aprovacao_2021` |
| A9 | Escalas incomparáveis (de 3 a 80.160 contra 4 a 6) | padronização obrigatória |
| A10 | Mesmo município correlaciona **0,64** entre seus dois ciclos | `GroupKFold` por município |

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
- **`ano`** fica fora porque o teste temporal treina apenas em 2023: um coeficiente de ano
  estimado só com 2023 não se aplicaria a 2024. O modelo descreve a **estrutura municipal**, não
  a tendência nacional.

---

## 7. Pipeline de pré-processamento

Três ramos no `ColumnTransformer`, todos **dentro** do `Pipeline`:

| Ramo | Tratamento | Motivo |
|---|---|---|
| Numéricas gerais | mediana → `StandardScaler` | a penalidade L2 encolhe coeficientes proporcionalmente à escala; sem padronizar, puniria arbitrariamente as variáveis de escala pequena |
| Bloco IDEB | mediana **com `add_indicator=True`** → `StandardScaler` | 13,3% de ausência concentrada em municípios pequenos e isolados: **a ausência é sinal** |
| Categóricas | moda → `OneHotEncoder(drop="first")` | *dummy encoding* — remove uma categoria para evitar multicolinearidade |

`handle_unknown="ignore"` cobre um caso real: os municípios do Acre só existem em 2024, então a
UF `AC` é desconhecida para um modelo treinado em 2023.

---

## 8. Validação

### 8.1 Por que a validação é um laço explícito, e não `GridSearchCV`

É a decisão técnica mais sutil do projeto.

Com alvo binomial ponderado, cada município gera **duas linhas idênticas em `X`** com rótulos
opostos. Toda a informação está no `sample_weight`. Um scorer que ignore o peso enxerga, para
cada município, um par indistinguível com `y=1` e `y=0` — e devolve **AUC exatamente 0,5**,
qualquer que seja o modelo.

Fazer o peso chegar ao scorer exigiria ativar o *metadata routing* do Scikit-learn e declarar o
roteamento em **cada** etapa aninhada do `ColumnTransformer`, inclusive nas que apenas ignoram o
peso. Testado: falha em `StandardScaler.fit_transform`. O laço em
`src/evaluation/metricas.py::validar_cruzado` é mais curto e não deixa dúvida sobre onde o peso
entra — no ajuste **e** na avaliação. A interface espelha a do `GridSearchCV`, inclusive
devolvendo o score de treino ao lado do de validação.

### 8.2 As duas estratégias

| Estratégia | Pergunta que responde |
|---|---|
| `GroupKFold` por `id_municipio` (5 folds) | generaliza para **municípios não vistos**? |
| Split temporal 2023 → 2024 | generaliza para o **próximo ciclo**? |

O `GroupKFold` é necessário porque o mesmo município correlaciona 0,64 entre seus ciclos: deixá-lo
atravessar a fronteira treino/validação permitiria memorização.

---

## 9. Referências de desempenho: piso, modelo e teto

Dados agregados impõem um **teto**: dentro de um município, todos os alunos compartilham o mesmo
vetor de características, e nenhum modelo distingue a criança alfabetizada da não alfabetizada da
mesma escola. Medimos esse teto com um **oráculo** — um "modelo" que já conhece a taxa observada.

| Referência | AUC | Brier | KS |
|---|---:|---:|---:|
| Piso — prever a média nacional | 0,500 | 0,237 | 0,084 |
| **Modelo — validação cruzada** | **0,661** | **0,218** | **0,229** |
| Teto — oráculo com a taxa observada | 0,737 | 0,198 | 0,343 |

**O modelo percorre 67,9% do intervalo entre o piso e o teto.**

Generalização temporal (treino só em 2023):

| Recorte | AUC | Brier |
|---|---:|---:|
| 2024, todas as UFs | 0,642 | 0,220 |
| **2024, sem o Rio Grande do Sul** | **0,672** | 0,214 |

O RS custa 3 pontos de AUC — confirmação quantitativa do diagnóstico da EDA. E o número sem o RS
é **superior ao da própria validação cruzada**, sinal forte de que o modelo captura estrutura
estável, e não particularidades de um ciclo.

---

## 10. Otimização de hiperparâmetros

Busca em grade em `C` ∈ {0,01; 0,1; 1; 10; 100}, com `GroupKFold` (`notebooks/03`, §5):

- `C` de 0,1 a 100 entrega o **mesmo** AUC de validação (0,661); só `C = 0,01` piora;
- adotado **`C = 1,0`**, o mais regularizado entre os empatados;
- diferença treino-validação de **0,0015** — praticamente zero.

**Conclusão honesta:** o modelo **não sofria de overfitting**, e a busca não "resolveu" nenhum
problema. Com 21.792 observações para 53 features num modelo linear, a regularização L2 é
salvaguarda contra colinearidade residual, não remédio para sobreajuste. Reportar isso é mais
útil do que apresentar a otimização como se tivesse produzido um ganho.

---

## 11. Comparação de algoritmos

Todos os candidatos pelo **mesmo** pré-processamento, a mesma validação ponderada e a mesma
partição. Só o estimador muda (`notebooks/03`, §10; números em `model_card.json`):

| Modelo | ROC-AUC | Brier | Gap treino-validação | Tempo (s) | % do teto |
|---|---:|---:|---:|---:|---:|
| Gradient Boosting | 0,6631 | 0,2178 | 0,0140 | 32,2 | 68,8 |
| **Regressão Logística** | **0,6611** | **0,2182** | **0,0015** | **0,5** | 67,9 |
| Random Forest | 0,6575 | 0,2194 | 0,0276 | 2,6 | 66,4 |
| Naive Bayes | 0,6414 | 0,3314 | 0,0009 | 0,4 | 59,6 |
| Árvore de Decisão | 0,6400 | 0,2217 | 0,0055 | 0,8 | 59,0 |
| Baseline (taxa média) | 0,5000 | 0,2367 | 0,0000 | 0,3 | 0,0 |

**Leitura:**

1. **O Gradient Boosting vence por 0,002 — dentro do desvio-padrão (0,003).** Empate estatístico,
   ao custo de 64× mais tempo, gap treino-validação 9× maior e toda a interpretabilidade.
2. **A Regressão Logística sobreajusta menos que qualquer modelo competitivo** (0,0015 contra
   0,014 e 0,028 dos ensembles).
3. **O Naive Bayes confirma o descarte a priori de forma mais grave que o previsto:** Brier de
   0,331, **pior que o baseline (0,237)** — probabilidades completamente descalibradas, como
   esperado de uma premissa de independência aplicada a três blocos composicionais.
4. **Nenhum algoritmo passa de 68,8% do intervalo piso→teto.** Cinco famílias diferentes, do
   linear ao boosting, param praticamente no mesmo lugar: **a limitação é a informação
   disponível, não a capacidade de modelo.** É a justificativa quantitativa das evoluções
   futuras — microdados de aluno e variáveis de política moveriam o resultado; um sexto algoritmo,
   não.

**Ficam de fora por impossibilidade técnica, não por preferência:** o **SVM**, porque o `SVC` não
produz probabilidade calibrada nativamente e é O(n²) em 21.792 observações; e qualquer estimador
que não aceite `sample_weight`, já que o alvo depende inteiramente do peso.

**Decisão:** a Regressão Logística fica — não por ser a melhor no AUC, mas porque empata com a
melhor dentro do desvio, sobreajusta menos, calibra melhor e é a única que responde "por quê".

---

## 12. Interpretabilidade

Três lentes, porque cada uma responde a uma pergunta diferente (`notebooks/03`, §8):

| | Coeficientes | Permutation Importance | SHAP |
|---|---|---|---|
| 1º | `sigla_uf_CE` (1,62) | `sigla_uf` (0,080) | `ideb_2021` (0,217) |
| 2º | `ideb_2021` (0,285) | `ideb_2021` (0,042) | `sigla_uf_CE` (0,131) |
| 3º | `media_inse` (0,118) | `media_inse` (0,007) | `media_inse` (0,105) |

As três apontam o mesmo trio. A ordem interna difere porque medem coisas distintas: o coeficiente
do Ceará é o maior de todos, mas o SHAP mede contribuição **média por predição** e aquela dummy só
se ativa em 184 dos 5.500 municípios; a *permutation* trata `sigla_uf` como bloco único.

Em pontos percentuais da taxa esperada, por desvio-padrão (`notebooks/04`, §1): IDEB 2021
**+5,5**, INSE **+2,5**, tamanho de turma **−1,5**, formação docente **+1,3**.

---

## 13. Hipóteses — registro e veredito

As cinco foram registradas na EDA **antes** de o modelo existir.

| # | Hipótese | Veredito |
|---|---|---|
| H1 | Desempenho pregresso é o melhor preditor disponível | **Confirmada** — `ideb_2021`, OR 1,33 |
| H2 | Efeito socioeconômico é muito menor que a correlação bruta sugere | **Confirmada em magnitude, refutada no mecanismo** (ver abaixo) |
| H3 | Qualificação docente tem efeito próprio | **Confirmada, fraca** — `afd_ai_grupo_1`, OR 1,06 |
| H4 | Há efeito de gestão estadual não capturado pelas variáveis | **Fortemente confirmada** — `sigla_uf_CE`, OR **5,06** |
| H5 | Municípios rurais e pequenos em desvantagem | **Parcialmente** — `capital_desc_interior` (OR 1,09) vai na direção oposta |

**Sobre H2.** Registramos, antes de treinar, a expectativa de que controlar por UF *encolheria* o
coeficiente de `media_inse`. Ele **dobrou** (0,073 → 0,118). A previsão estava errada porque
comparava quantidades diferentes: a EDA mediu uma correlação **bivariada**, confundida com a
região; o coeficiente do modelo é **parcial**, condicionado a UF e a mais trinta variáveis.
Remover a variação entre regiões — que incluía o gradiente negativo do Nordeste — limpa o sinal
em vez de reduzi-lo. É efeito de supressão, não contradição. A conclusão prática permanece: o
nível socioeconômico **não determina** o resultado.

---

## 14. Respostas às perguntas de negócio

`notebooks/04`.

| # | Pergunta | Resposta |
|---|---|---|
| 1 e 5 | Quais fatores mais impactam / têm maior influência? | **Respondido.** UF (dominante) > desempenho pregresso (+5,5 p.p.) > socioeconômico (+2,5) > alavancas escolares (1–1,5) |
| 2 | Quais municípios apresentam maior risco? | **Respondido, com ressalva metodológica.** Priorizar pela taxa observada; o resíduo distingue "contexto adverso" de "aquém do contexto", mas **só é confiável no agregado** — uma lista individual mudaria 83% entre ciclos |
| 3 | Quais regiões têm padrões semelhantes? | **Respondido, contrariando a formulação.** Não há grupos naturais (silhueta ≤ 0,20 para todo `k`); os quatro segmentos adotados **atravessam** as regiões — o de maior vulnerabilidade aparece em 24 das 25 UFs |
| 4 | Como prever quem não atingirá as metas? | **Respondido.** 43,4% já cumpriram em 2024 a meta de 2025; dos 3.030 restantes, **678** têm estrutura que a sustenta e **2.352** não |

### 14.1 O teste que reprovou uma entrega tentadora

Seria natural publicar um ranking dos municípios que "pior gerem sua rede", usando o resíduo do
modelo. Testamos a estabilidade antes:

| Critério | Sobreposição da lista dos 300 piores entre 2023 e 2024 | Correlação entre ciclos |
|---|---:|---:|
| Resíduo do modelo | **17,3%** | 0,28 |
| Taxa observada | 35,7% | 0,64 |

Quatro em cada cinco nomes mudariam no ciclo seguinte. A lista mediria ruído, e seria usada para
punir prefeituras. **Não foi produzida** — e a decisão está documentada, com o número que a
sustenta.

O resíduo **é** útil no agregado: ele detectou o choque do RS sozinho, indo de +10,1 (2023) para
−10,3 (2024), isolado de qualquer outra UF.

---

## 15. Limitações

1. **Falácia ecológica** — estima o aluno típico do município, não uma criança específica.
2. **Ausência de variáveis de política educacional** — a lacuna mais séria: o maior efeito do
   modelo (a dummy do Ceará) é justamente o que ele **não consegue explicar**.
3. **Peso por município, não por aluno.**
4. **Cobertura incompleta** — 25 das 27 UFs; DF sem rede municipal, Roraima ausente da fonte.
5. **INSE de 2023 replicado para 2024.**
6. **AFD/ATU/IED usam o agregado `Total`**, que inclui a rede privada.
7. **Apenas dois ciclos** — tendência e choque indistinguíveis; inviabiliza série temporal.
8. **Associações condicionais, não efeitos causais.**

---

## 16. Decisões revistas durante o projeto

Registradas porque o processo importa tanto quanto o resultado — e porque cada uma foi corrigida
por uma verificação, não por opinião.

| Afirmação inicial | O que a verificação mostrou |
|---|---|
| "As proporções por nível somam exatamente para a taxa" | **Não somam** — o corte de 743 pontos cai *dentro* de um nível. A melhor aproximação é a soma dos níveis 5–8 (r = 0,986, erro médio 5,9 p.p.). Continua vazamento severo, por proximidade da medição e não por identidade |
| "Controlada a região, a relação INSE × taxa é crescente em todas" | **Falso.** Com decis intrarregionais: fraca no Sudeste (0,01) e no Sul (0,04), **negativa** no Nordeste (−0,14). A narrativa correta — confundimento geográfico — é mais forte que a errada |
| "Controlar por UF encolherá o coeficiente do INSE" | **Dobrou** (0,073 → 0,118). Efeito de supressão; ver §13 |
| "`meta_alfabetizacao_2030` é 100 para todo município" | É **80** para toda a rede municipal |
| "Cobertura idêntica nos dois ciclos" | O *número* de municípios é idêntico (5.448), o *conjunto* não: 5.500 distintos, 5.396 nos dois, 104 entram ou saem |
| "~850 mil registros no Bronze" | **719.757** |

**Regra adotada a partir daí:** todo número afirmado em texto precisa sair de uma célula
executada, do `model_card.json` ou ser calculado no próprio título do gráfico.

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
