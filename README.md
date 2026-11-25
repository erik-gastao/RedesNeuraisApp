# Construtor de Redes Neurais

Programa de terminal que monta, treina e avalia uma rede neural MLP a partir
de perguntas feitas ao usuário. A rede é implementada à mão, só com NumPy —
sem Keras, TensorFlow, PyTorch ou scikit-learn.

Trabalho da disciplina de Inteligência Artificial.

## Instalar

1. Instale o [Python 3](https://www.python.org/downloads/) (3.10 ou mais novo).
   No instalador do Windows, marque **"Add python.exe to PATH"**.
2. Abra um terminal nesta pasta e instale as bibliotecas:

   ```
   pip install -r requirements.txt
   ```

   Isso instala o NumPy (cálculos da rede) e o matplotlib (só para o gráfico
   do erro, opcional).

## Rodar

**No Windows, com dois cliques:** abra `executar.bat`.

**Pelo terminal**, nesta pasta:

```
python main.py
```

O programa pergunta a arquitetura da rede, a base de dados, a taxa de
aprendizado e o número de épocas. Depois treina, mostra o erro caindo e
avalia o resultado. Valor digitado errado não quebra o programa: ele explica
o problema e pergunta de novo.

### Exemplo: aprendendo o XOR

Respostas para reproduzir (o que vem depois de cada pergunta):

```
=== Construtor de Redes Neurais ===
Quer criar uma rede neural? (s/n) s
Quantos neurônios na camada de entrada? 2
Quantas camadas ocultas? 1
  Camada oculta 1 — quantos neurônios? 4
  Camada oculta 1 — ativação (sigmoid/relu/tanh)? sigmoid
Quantos neurônios na camada de saída? 1
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
  3) bezdekIris.data  (Dataset/)
  4) iris.data  (Dataset/)
Base de dados — número da lista ou caminho do arquivo (Enter = cancelar e montar outra rede): 2
  Base carregada: 4 amostras.
  Base pequena (menos de 10 amostras): não foi separada. O treino e o teste usam a base inteira.
Taxa de aprendizado (η) (Enter = 0.5): 0.5
Número de épocas (Enter = 5000): 10000

Treinando...
  Época  1000/10000 — erro (MSE): 0.0735
  ...
  Época 10000/10000 — erro (MSE): 0.0007

Avaliação (base inteira):
  Erro final (MSE): 0.0007
  Acurácia: 100,00% (4/4)

  entrada | esperado | ŷ      | classe | ok?
  0 0     | 0        | 0.0118 | 0      | sim
  0 1     | 1        | 0.9726 | 1      | sim
  1 0     | 1        | 0.9735 | 1      | sim
  1 1     | 0        | 0.0324 | 0      | sim
```

Com a mesma semente o resultado se repete. Com Enter (semente aleatória) os
pesos iniciais mudam a cada execução, e a rede pode precisar de mais épocas.

Depois da avaliação aparece o menu:

| Opção | O que faz |
|---|---|
| 1 | Testar uma entrada digitada: mostra o `ŷ` e a classe |
| 2 | Treinar mais épocas, continuando dos pesos atuais |
| 3 | Criar outra rede |
| 4 | Ver o gráfico do erro por época |
| 0 | Sair |

### Gráfico do erro

A opção 4 abre uma janela com a curva do erro (MSE) em cada época e salva a
mesma figura em `graficos/`, com o nome da base e a data (ex.:
`graficos/erro_xor_20261112_153000.png`). Feche a janela para voltar ao menu.

A curva descendo mostra a rede aprendendo. Quando ela fica reta, treinar
mais com a mesma configuração quase não muda o resultado. Depois de "Treinar
mais épocas", o gráfico mostra todas as épocas desde o início.

O gráfico usa o `matplotlib`. Sem ele instalado, o resto do programa
funciona igual e a opção 4 só avisa como instalar.

### Experimentos para tentar

- **XOR sem camada oculta** (responda `0` em "Quantas camadas ocultas?"): a
  rede não passa de 75% de acerto, com qualquer semente. O XOR não é
  linearmente separável; é a camada oculta que resolve.
- **AND sem camada oculta** (`bases/and.csv`, com bias): chega a 100%. Um
  neurônio só já separa as classes com uma reta.
- **AND sem bias:** não chega a 100%. Para a entrada `(0, 0)` a soma
  ponderada é sempre 0, a sigmoid devolve 0,5 e o degrau classifica como 1.

## Escolher a base de dados

Depois de montar a rede, o programa lista os arquivos `.csv`, `.data` e
`.txt` das pastas `bases/` e `Dataset/`:

```
Bases disponíveis:
  1) and.csv  (bases/)
  2) xor.csv  (bases/)
  3) bezdekIris.data  (Dataset/)
  4) iris.data  (Dataset/)
Base de dados — número da lista ou caminho do arquivo (Enter = cancelar e montar outra rede):
```

Digite o número, ou o caminho de qualquer outro arquivo. Para uma base nova
aparecer na lista, basta copiar o arquivo para `bases/`.

## Formato da base de dados

- Arquivo CSV separado por **vírgula**, com **números** (a última coluna
  pode ser o nome da classe — veja "Classe em texto" abaixo).
- Cabeçalho opcional: a primeira linha é ignorada se tiver texto onde a
  linha seguinte tem número.
- As **últimas colunas** são a saída esperada, uma para cada neurônio de
  saída. As outras são as entradas.
- Número de colunas = neurônios de entrada + neurônios de saída. Se não
  bater, o programa avisa quantas colunas esperava.
- A saída esperada deve ser **0 ou 1**, porque a rede classifica em 0 ou 1.

Exemplo (`bases/xor.csv`), 2 entradas e 1 saída:

```
x1,x2,y
0,0,0
0,1,1
1,0,1
1,1,0
```

O que o programa faz com a base, sem perguntar:

- **Normaliza as entradas e as saídas** para o intervalo de 0 a 1 (min-max
  por coluna, com a escala da parte de treino). Entradas digitadas no menu
  passam pela mesma conversão, então digite os valores na escala original
  da base; o ŷ mostrado volta para a escala original. O erro (MSE) é
  calculado na escala normalizada.
- **Normalização por lotes (opcional):** em bases com 10 linhas ou mais, o
  programa pergunta o tamanho do lote. Com Enter, usa a base inteira; com
  um número menor, cada lote de registros do treino é normalizado com o
  mínimo e o máximo dele mesmo. Lotes muito pequenos pioram o resultado
  (na Iris: 96,67% com a base inteira ou lotes de 30; 76,67% com lotes de 4).
- **Separa treino e teste** quando a base tem 10 linhas ou mais: 80% para
  treinar e 20% para avaliar, embaralhadas com a semente escolhida. Com menos
  de 10 linhas, treina e avalia na base inteira e avisa.

### Classe em texto (Iris)

A última coluna também pode ser o **nome da classe**, como em
`Dataset/bezdekIris.data`:

```
5.1,3.5,1.4,0.2,Iris-setosa
7.0,3.2,4.7,1.4,Iris-versicolor
6.3,3.3,6.0,2.5,Iris-virginica
```

Nesse caso o programa converte sozinho: cada classe vira uma coluna 0/1
(setosa `1 0 0`, versicolor `0 1 0`, virginica `0 0 1`). A rede precisa de
**um neurônio de saída por classe** — 3 para a Iris — e a tabela mostra o
nome da classe em vez dos números.

Com mais de um neurônio de saída, a classe escolhida é a do neurônio de
**maior ŷ**, e não o degrau em cada saída: assim a rede sempre responde
exatamente uma classe. Com um neurônio só (XOR, AND) continua valendo o
degrau.

### Exemplo: Iris

| Pergunta | Resposta |
|---|---|
| Neurônios na camada de entrada | `4` |
| Camadas ocultas | `1`, com `5` neurônios, `sigmoid` |
| Neurônios na camada de saída | `3` |
| Bias | `s` |
| Semente | `1` |
| Base de dados | `3` (bezdekIris.data) |
| Taxa de aprendizado | `0.5` |
| Épocas | `5000` |

Resultado: 120 amostras para treino, 30 para teste, acurácia de **96,67%
(29/30)** no teste. Em outras sementes fica entre 93% e 100%. Não chega
sempre a 100% porque versicolor e virginica têm medidas parecidas.

`iris.data` e `bezdekIris.data` são a mesma base; a segunda corrige dois
erros de digitação da primeira (linhas 35 e 38). Prefira a `bezdekIris.data`.

## Como a rede funciona

| Tema | Decisão |
|---|---|
| Tipo | MLP: todo neurônio liga a todos os da camada seguinte |
| Camada de entrada | Só repassa os valores, sem pesos e sem ativação |
| Camadas ocultas | Quantidade, neurônios e ativação (`sigmoid`, `relu`, `tanh`) escolhidos pelo usuário, camada por camada |
| Camada de saída | Sigmoid no treino; o degrau (`1 se ŷ ≥ 0,5, senão 0`) só gera a classe final. O degrau não tem derivada útil, então não dá para treinar com ele. Com várias saídas, a classe é a do maior ŷ |
| Pesos iniciais | Aleatórios, uniformes entre −1 e 1 |
| Bias | Opcional. Começa em 0; se desligado, fica em 0 |
| Erro | MSE: `E = (1/N) · Σ (y − ŷ)²` |
| Treino | Gradiente descendente com a base inteira por atualização: `W = W − η · ∂E/∂W` |

## Estrutura do código

```
RedesNeuraisApp/
├── main.py              # menu e perguntas (único arquivo com input/print)
├── dados.py             # carregar CSV, normalizar, separar treino/teste, acurácia
├── graficos.py          # gráfico do erro por época (matplotlib, opcional)
├── rede/
│   ├── ativacoes.py     # sigmoid, relu, tanh, degrau e derivadas
│   ├── camada.py        # classe Camada: pesos, forward, backward, atualizar
│   ├── rede_neural.py   # classe RedeNeural: forward, treinar, prever, classificar
│   └── perda.py         # MSE e derivada
├── bases/
│   ├── xor.csv
│   └── and.csv
├── Dataset/
│   ├── bezdekIris.data  # Iris corrigida (usar esta)
│   └── iris.data        # Iris original
├── testes/
│   ├── test_rede.py     # ativações, formas, exemplo da aula, gradiente, aprendizado
│   ├── test_dados.py    # CSV, classes em texto, normalização, separação, acurácia
│   ├── test_main.py     # programa inteiro com respostas simuladas
│   └── test_graficos.py # gráfico do erro
├── executar.bat         # abre o programa com dois cliques (Windows)
├── testar.bat           # roda os testes com dois cliques (Windows)
├── requirements.txt
├── PLANO.md             # especificação do projeto
└── README.md
```

## Testes

Com dois cliques em `testar.bat`, ou pelo terminal nesta pasta:

```
python -m unittest discover testes
```

São 93 testes e levam cerca de 9 segundos. O mais importante é a verificação
numérica do gradiente (`TesteGradiente` em `testes/test_rede.py`): para cada
peso, compara o gradiente calculado pelo backpropagation com
`(E(w + ε) − E(w − ε)) / (2ε)`. Como os dois batem, o backpropagation está
correto.

## Problemas comuns

| Problema | Solução |
|---|---|
| `'python' não é reconhecido como um comando` | Reinstale o Python marcando "Add python.exe to PATH", ou use `py main.py` |
| `ModuleNotFoundError: No module named 'numpy'` | Rode `pip install -r requirements.txt` |
| `ModuleNotFoundError: No module named 'rede'` | Rode os comandos de dentro da pasta do projeto |
| `Arquivo não encontrado` ao informar a base | Escolha pelo número da lista, ou use o caminho completo do arquivo |
| `a rede precisa de 3 neurônio(s) de saída` | A base tem a classe em texto: monte a rede com um neurônio de saída por classe |
| A rede não aprende (acurácia baixa) | Tente mais épocas (opção 2 do menu), outra semente, mais neurônios na camada oculta ou ligar o bias |
