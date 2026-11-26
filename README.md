# Construtor de Redes Neurais

Programa de terminal que monta, treina, avalia, salva e carrega redes
neurais MLP a partir de perguntas feitas ao usuário. A rede é implementada à
mão, só com NumPy — sem Keras, TensorFlow, PyTorch ou scikit-learn.

Trabalho Prático 02 da disciplina de Inteligência Artificial ("Sistema
Configurável de Redes Neurais Artificiais (MLP)").

## Instalar

1. Instale o [Python 3](https://www.python.org/downloads/) (3.10 ou mais novo).
   No instalador do Windows, marque **"Add python.exe to PATH"**.
2. Abra um terminal nesta pasta e instale as bibliotecas:

   ```
   pip install -r requirements.txt
   ```

   Isso instala o NumPy (cálculos da rede), o matplotlib (só para o gráfico
   do erro, opcional) e o pygame-ce, o pygame_gui e o Pillow (interface
   gráfica).

## Rodar

**No Windows, com dois cliques:** abra `executar.bat`.

**Pelo terminal**, nesta pasta:

```
python main.py
```

O programa pergunta a arquitetura da rede, a base de dados, a divisão em
treino/validação/teste, se deve normalizar as entradas, a taxa de
aprendizado e o número de épocas. Depois treina, mostra o erro de treino e de
validação a cada época e avalia o resultado no conjunto de teste. Valor
digitado errado não quebra o programa: ele explica o problema e pergunta de
novo.

### Interface gráfica (em construção)

Com dois cliques em `executar_interface.bat`, ou `python interface_grafica.py`.
O hub monta a rede (base, camadas, ativações, divisão, normalização) e
"Gerar" treina gravando os passos. O player toca o treino passo a passo:
espaço tocar/pausar, ←/→ passo, Shift+←/→ neurônio, Home/End fase (com Ctrl,
exemplo; com Ctrl+Shift, época), −/+ velocidade, V visão; clique ou arraste
na barra de tempo para pular. Salvar/carregar e prever entrada ainda virão
(ver `PLANO_INTERFACE.md`).

Respondendo `n` em "Quer criar uma rede neural?", o programa oferece
**carregar uma rede salva** (`.json`) para fazer previsões.

### Exemplo: aprendendo o XOR

Respostas para reproduzir (o que vem depois de cada pergunta):

```
=== Construtor de Redes Neurais ===
Quer criar uma rede neural? (s/n) s
Quantos neurônios na camada de entrada? 2
Quantas camadas ocultas? 1
  Camada oculta 1 — quantos neurônios? 4
  Camada oculta 1 — ativação (sigmoid/relu/tanh/linear)? sigmoid
Quantos neurônios na camada de saída? 1
  Camada de saída — ativação (sigmoid/relu/tanh/linear) (Enter = sigmoid)?
Usar bias? (s/n) s
Semente aleatória (Enter = aleatória): 42

Rede montada:
  Entrada: 2 neurônios (sem ativação)
  Oculta 1: 4 neurônios, sigmoid — 2 × 4 = 8 pesos (+4 bias)
  Saída: 1 neurônio, sigmoid no treino / degrau na classificação — 4 × 1 = 4 pesos (+1 bias)
  Total de pesos: 12

Bases disponíveis:
  1) and.csv  (bases/)
  2) xor.csv  (bases/)
  3) bezdekIris.data  (Datasets/)
  4) dataset.csv  (Datasets/)
  5) iris.data  (Datasets/)
Base de dados — número da lista ou caminho do arquivo (Enter = cancelar e montar outra rede): 2
  Base carregada: 4 amostras.
  Base pequena (menos de 10 amostras): não foi separada. O treino e o teste usam a base inteira.
Normalizar as entradas (min-max, escala do treino)? (s/n) n
Taxa de aprendizado (η) (Enter = 0.1): 0.5
Número de épocas (Enter = 1000): 3000

Treinando...
  Época  300/3000 — erro treino (MSE): 0.0958
  ...
  Época 3000/3000 — erro treino (MSE): 0.0004

Avaliação (base inteira):
  MSE treino: 0.0004 | teste: 0.0004
  Acurácia: 100,00% (4/4)

  entrada | esperado | ŷ      | classe | ok?
  0 0     | 0        | 0.0211 | 0      | sim
  0 1     | 1        | 0.9763 | 1      | sim
  1 0     | 1        | 0.9847 | 1      | sim
  1 1     | 0        | 0.0201 | 0      | sim
```

Com a mesma semente o resultado se repete (a semente define os pesos
iniciais, a divisão da base e a ordem dos exemplos no treino).

Depois da avaliação aparece o menu:

| Opção | O que faz |
|---|---|
| 1 | Testar uma entrada digitada (previsão): mostra o `ŷ` e, na classificação, a classe |
| 2 | Treinar mais épocas, continuando dos pesos atuais |
| 3 | Criar outra rede |
| 4 | Ver o gráfico do erro por época (treino e validação) |
| 5 | Salvar a rede em um arquivo `.json` (padrão: `redes/<base>.json`) |
| 0 | Sair |

### Exemplo: base do professor (regressão)

`Datasets/dataset.csv` tem 179 linhas: `sementes_m2`, `nitrogenio_kg_ha` →
`produtividade_kg_ha` (entre 1700 e 3160). A saída é um número, não uma
classe, e as saídas esperadas **não são normalizadas** (exigência do
enunciado). Por isso:

| Pergunta | Resposta | Por quê |
|---|---|---|
| Entrada / ocultas / saída | `2`, `1` oculta com `8` neurônios `relu`, `1` | |
| Ativação da saída | `linear` | sigmoid só vai de 0 a 1; a linear alcança 2500 |
| Base | `4` (dataset.csv) | |
| Divisão | Enter (70/15/15 → 125/27/27) | |
| Normalizar as entradas | `s` | 450 sementes e 120 kg de N ficam entre 0 e 1 |
| Taxa de aprendizado | `0.00001` | com Y na casa de 2500, o erro é grande; η pequeno evita saltos |
| Épocas | `1000` | |

Com semente `1`, o MSE de teste fica perto de 25 mil (erro típico de ~160
kg/ha), contra ~60 mil se a rede previsse sempre a média. Com `tanh` ou
`sigmoid` na oculta a rede satura e fica prevendo só a média: use `relu`.
Na regressão a tabela mostra `entrada | esperado | ŷ | erro` e não há
acurácia.

### Gráfico do erro

A opção 4 abre uma janela com as curvas do erro (MSE) do treino e da
validação em cada época e salva a mesma figura em `graficos/`, com o nome da
base e a data (ex.: `graficos/erro_xor_20261125_153000.png`). Feche a janela
para voltar ao menu.

A curva descendo mostra a rede aprendendo. Se a validação começa a subir
enquanto o treino continua caindo, a rede está decorando o treino. Sem o
`matplotlib` instalado, o resto do programa funciona igual e a opção 4 só
avisa como instalar.

### Experimentos para tentar

- **XOR sem camada oculta** (responda `0` em "Quantas camadas ocultas?"): a
  rede não passa de 75% de acerto, com qualquer semente. O XOR não é
  linearmente separável; é a camada oculta que resolve.
- **AND sem camada oculta** (`bases/and.csv`, com bias): chega a 100%. Um
  neurônio só já separa as classes com uma reta (o Perceptron da Aula 12).
- **AND sem bias:** não chega a 100%. Para a entrada `(0, 0)` a soma
  ponderada é sempre 0, a sigmoid devolve 0,5 e o degrau classifica como 1.

## Escolher a base de dados

Depois de montar a rede, o programa lista os arquivos `.csv`, `.data` e
`.txt` das pastas `bases/` e `Datasets/`. Digite o número, ou o caminho de
qualquer outro arquivo. Para uma base nova aparecer na lista, basta copiar o
arquivo para uma dessas pastas.

## Formato da base de dados

- Arquivo CSV separado por **vírgula**, com **números** (a última coluna
  pode ser o nome da classe — veja "Classe em texto" abaixo).
- Cabeçalho opcional: a primeira linha é ignorada se tiver texto onde a
  linha seguinte tem número.
- As **últimas colunas** são a saída esperada, uma para cada neurônio de
  saída. As outras são as entradas.
- Número de colunas = neurônios de entrada + neurônios de saída. Se não
  bater, o programa avisa quantas colunas esperava.
- Saída **0 ou 1** (ou classe em texto) = classificação: mostra classe e
  acurácia. Qualquer outro número = regressão: use `linear` na saída.

O que o programa faz com a base:

- **Divide em treino, validação e teste** quando a base tem 10 linhas ou
  mais. Pergunta os percentuais (Enter = `70/15/15`; aceita `70/15/15`,
  `70 15 15` ou `70,15,15`; com `/` a vírgula vale como decimal, como em
  `70,5/14,5/15`; precisam somar 100) e embaralha com a semente
  escolhida. O treino ajusta os pesos; a validação só mede o erro a cada
  época; o teste fica reservado para a avaliação final. Com menos de 10
  linhas (XOR, AND), treina e avalia na base inteira, sem validação, e avisa.
- **Normaliza as entradas, se você quiser** (min-max: `(x − mín) / (máx − mín)`).
  É feita depois da divisão, com o mínimo e o máximo **só do treino**,
  aplicados também à validação, ao teste, às entradas digitadas e salvos
  junto com a rede. As saídas esperadas nunca são normalizadas.
- **Normalização por lotes (opcional):** se normalizou e a base foi
  dividida, o programa pergunta o tamanho do lote. Com Enter, usa o treino
  inteiro; com um número menor, cada lote de registros do treino é
  normalizado com o mínimo e o máximo dele mesmo. É uma opção extra deste
  projeto e foge do padrão do enunciado (uma escala só, do treino, para os
  três conjuntos); o programa avisa. Lotes muito pequenos pioram o resultado.

### Classe em texto (Iris)

A última coluna também pode ser o **nome da classe**, como em
`Datasets/bezdekIris.data`:

```
5.1,3.5,1.4,0.2,Iris-setosa
7.0,3.2,4.7,1.4,Iris-versicolor
6.3,3.3,6.0,2.5,Iris-virginica
```

Nesse caso o programa converte sozinho: cada classe vira uma coluna 0/1
(setosa `1 0 0`, versicolor `0 1 0`, virginica `0 0 1`). A rede precisa de
**um neurônio de saída por classe** — 3 para a Iris — e a tabela mostra o
nome da classe em vez dos números. A classe escolhida é a do neurônio de
**maior ŷ**, para a rede sempre responder exatamente uma classe.

### Exemplo: Iris

| Pergunta | Resposta |
|---|---|
| Neurônios na camada de entrada | `4` |
| Camadas ocultas | `1`, com `5` neurônios, `sigmoid` |
| Neurônios na camada de saída | `3` (ativação: Enter = sigmoid) |
| Bias / semente | `s` / `1` |
| Base de dados | `3` (bezdekIris.data) |
| Divisão / normalizar / lote | Enter / `s` / Enter |
| Taxa de aprendizado / épocas | `0.1` / `200` |

Resultado: 106 amostras para treino, 22 para validação e 22 para teste,
acurácia de **95,45% (21/22)** no teste. Em outras sementes fica entre 95% e
100%. Não chega sempre a 100% porque versicolor e virginica têm medidas
parecidas.

`iris.data` e `bezdekIris.data` são a mesma base; a segunda corrige dois
erros de digitação da primeira (linhas 35 e 38). Prefira a `bezdekIris.data`.

## Salvar e carregar uma rede

A opção 5 do menu grava a rede num arquivo JSON. Para usar depois: rode o
programa, responda `n` em "Quer criar uma rede neural?", `s` em "Quer
carregar uma rede salva?" e informe o arquivo (Enter volta à primeira
pergunta). A rede carregada dá
exatamente as mesmas saídas da original e aplica sozinha a normalização
salva às entradas digitadas.

Formato do arquivo (versão 1):

```json
{
  "formato": "rede-mlp",
  "versao": 1,
  "n_entradas": 2,
  "usar_bias": true,
  "camadas": [
    {
      "neuronios": 8,
      "ativacao": "relu",
      "pesos": [[0.41, -1.07], [1.32, 0.05], "... uma lista por neurônio"],
      "vieses": [0.12, -0.03, "... um por neurônio"]
    },
    {
      "neuronios": 1,
      "ativacao": "linear",
      "pesos": [[-35.2, 41.7, "... um peso por neurônio da camada anterior"]],
      "vieses": [2410.8]
    }
  ],
  "normalizacao": {"funcao": "min-max", "minimos": [150.0, 0.0], "maximos": [450.0, 120.0]},
  "classes": null
}
```

| Campo | Conteúdo |
|---|---|
| `n_entradas` | Neurônios da camada de entrada (colunas de entrada da base) |
| `usar_bias` | Se os vieses são usados/treinados |
| `camadas` | Camadas ocultas em ordem; **a última é a de saída**. `pesos[i][j]` é o peso da entrada `j` no neurônio `i`; `vieses[i]` é o viés do neurônio `i` |
| `normalizacao` | `null` se as entradas não foram normalizadas; senão a função e os mínimos/máximos de cada entrada, calculados no treino |
| `classes` | Nomes das classes (base com classe em texto) ou `null` |

## Como a rede funciona

| Tema | Decisão |
|---|---|
| Tipo | MLP: todo neurônio liga a todos os da camada seguinte |
| Neurônio | Guarda pesos, viés, entradas, `z = Σ wⱼxⱼ + b`, saída `a = f(z)` e delta `δ = ∂E/∂z` (Aula 12, slides 4–6) |
| Camada de entrada | Só repassa os valores, sem pesos e sem ativação |
| Ativações | `linear` (identidade), `sigmoid`, `tanh` e `relu`, com derivadas; cada camada escolhe a sua, inclusive a de saída |
| Saída na classificação | Com 1 saída, degrau `1 se ŷ ≥ 0,5`; com várias, a de maior ŷ. O degrau só gera a classe: não tem derivada útil para treinar |
| Pesos iniciais | Xavier/Glorot em camadas sigmoid, tanh e linear: uniforme em `±√(6/(entradas + neurônios))`. He em camadas ReLU: `±√(6/entradas)`. É a recomendação da seção 1.5 do enunciado: evita somas `z` grandes que saturam a sigmoid/tanh. A semente torna os pesos reproduzíveis |
| Bias | Opcional. Começa em 0 e é salvo junto com os pesos; se desligado, fica em 0 |
| Erro | MSE: `E = (1/N) · Σ (y − ŷ)²`, de treino, validação e teste |
| Retropropagação | Na saída `δ = 2(ŷ − y) · f'(z)`; nas ocultas `δⱼ = f'(zⱼ) · Σᵢ δᵢ wᵢⱼ`, com os pesos de antes do ajuste (Aula 13, slides 8–14). Todos os deltas são calculados antes de qualquer atualização |
| Treino | SGD: depois de **cada exemplo**, `w = w − η · δ · x` e `b = b − η · δ`. A ordem dos exemplos é sorteada a cada época |

Diferença para os slides da Aula 13: lá a atualização é feita uma vez com a
média dos gradientes das N amostras; o enunciado do trabalho pede SGD, que
atualiza a cada exemplo. As fórmulas são as mesmas, com N = 1. O teste
`TesteExemploAula13` refaz o exemplo XOR dos slides com os pesos deles e
confere ŷ, MSE, deltas e gradientes.

## Estrutura do código

```
RedesNeuraisApp/
├── main.py              # menu e perguntas (único arquivo com input/print)
├── experimento.py       # fluxo sem input/print: dividir, normalizar, treinar, avaliar, prever, salvar
├── dados.py             # carregar CSV, normalizar, dividir treino/validação/teste, acurácia
├── persistencia.py      # salvar e carregar a rede em JSON
├── graficos.py          # gráfico do erro por época (matplotlib, opcional)
├── interface_grafica.py # abre a interface gráfica (pygame)
├── interface/           # interface gráfica (em construção; ver PLANO_INTERFACE.md)
│   ├── app.py           # janela, telas e eventos
│   ├── tela_hub.py      # hub: escolher a base e montar a rede
│   ├── config_hub.py    # validação do formulário do hub
│   ├── tela_gravando.py # barra de progresso enquanto grava o treino
│   ├── tela_player.py   # player passo a passo
│   ├── desenho_rede.py  # desenha a rede com pygame
│   ├── passos.py        # Passo da animação e estado_em()
│   ├── formulas.py      # textos das fórmulas com números
│   ├── gravador.py      # grava o treino SGD em passos
│   └── layout_rede.py   # posições, cores e espessuras da rede desenhada
├── rede/
│   ├── ativacoes.py     # linear, sigmoid, relu, tanh, degrau e derivadas
│   ├── inicializacao.py # pesos iniciais Xavier/He
│   ├── neuronio.py      # classe Neuronio: pesos, bias, z, a, delta
│   ├── camada.py        # classe Camada: lista de neurônios
│   ├── rede_neural.py   # classe RedeNeural: forward, retropropagar, prever, classificar
│   ├── treinamento.py   # treino SGD com MSE de treino e validação
│   └── perda.py         # MSE e derivada
├── bases/               # xor.csv, and.csv
├── Datasets/            # dataset.csv (professor), bezdekIris.data, iris.data
├── testes/
│   ├── test_rede.py         # ativações, Xavier/He, neurônio, slides da aula, gradiente, SGD
│   ├── test_dados.py        # CSV, classes em texto, normalização, divisão, acurácia
│   ├── test_persistencia.py # salvar/carregar JSON
│   ├── test_experimento.py  # fluxo completo sem terminal
│   ├── test_gravador.py     # passos do player batem com a rede real
│   ├── test_layout_rede.py  # posições, cores e espessuras
│   ├── test_config_hub.py   # validação do hub
│   ├── test_interface.py    # fumaça da interface (sem janela)
│   ├── test_main.py         # programa inteiro com respostas simuladas
│   └── test_graficos.py     # gráfico do erro
├── executar.bat         # abre o programa com dois cliques (Windows)
├── executar_interface.bat # abre a interface gráfica com dois cliques
├── testar.bat           # roda os testes com dois cliques (Windows)
├── requirements.txt
├── PLANO.md             # especificação original do projeto
└── README.md
```

## Testes

Com dois cliques em `testar.bat`, ou pelo terminal nesta pasta:

```
python -m unittest discover testes
```

São 263 testes e levam cerca de 40 segundos. Os mais importantes:

- `TesteGradiente` (`testes/test_rede.py`): para cada peso de cada
  neurônio, compara o gradiente da retropropagação (`δ · x`) com
  `(E(w + ε) − E(w − ε)) / (2ε)`. Como os dois batem, o backpropagation
  está correto.
- `TesteExemploAula13`: reproduz os números do exemplo dos slides.
- `TestePersistencia`: a rede salva e carregada dá saídas idênticas.

## Problemas comuns

| Problema | Solução |
|---|---|
| `'python' não é reconhecido como um comando` | Reinstale o Python marcando "Add python.exe to PATH", ou use `py main.py` |
| `ModuleNotFoundError: No module named 'numpy'` | Rode `pip install -r requirements.txt` |
| `ModuleNotFoundError: No module named 'rede'` | Rode os comandos de dentro da pasta do projeto |
| `Arquivo não encontrado` ao informar a base | Escolha pelo número da lista, ou use o caminho completo do arquivo |
| `a rede precisa de 3 neurônio(s) de saída` | A base tem a classe em texto: monte a rede com um neurônio de saída por classe |
| Aviso "Use a ativação linear na saída" | A saída da base passa do que a ativação da saída produz (sigmoid: 0 a 1; tanh: −1 a 1; relu: ≥ 0): monte a rede com `linear` na saída |
| Regressão prevendo sempre o mesmo valor | Use `relu` na oculta, normalize as entradas e diminua a taxa (ex.: `0.00001`) |
| A rede não aprende (acurácia baixa) | Tente mais épocas (opção 2 do menu), outra semente, mais neurônios na camada oculta ou ligar o bias |
