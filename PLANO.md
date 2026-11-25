# Plano — Construtor de Redes Neurais (versão inicial)

> **Atualização (2026-11-25):** o enunciado do Trabalho Prático 02 substitui
> partes deste plano. Valem agora: entidade Neurônio com delta, ativação
> Linear, pesos iniciais Xavier/He, treino **SGD** (atualiza a cada exemplo),
> divisão **treino/validação/teste** com percentuais configuráveis,
> normalização **opcional e só das entradas** (as saídas não são
> normalizadas), MSE de validação por época e salvar/carregar a rede em JSON.
> O README descreve o comportamento atual; este documento fica como histórico
> das decisões iniciais.

Documento de especificação para implementar o projeto. Quem for executar
(pessoa ou outro modelo de IA) deve seguir este plano na ordem das etapas.

## 1. Contexto

Trabalho em grupo (até 4 pessoas) da disciplina de Inteligência Artificial.
Entrega: código-fonte completo + vídeo explicando código, implementação e
funcionamento.

O programa é um **construtor de redes neurais MLP pelo terminal**. O usuário
responde perguntas, a rede é montada, treinada com uma base de dados (o
professor vai fornecer uma) e avaliada pela capacidade de identificar o padrão.

## 2. Regras gerais para quem implementar

- **Linguagem:** Python 3 (máquina tem Python 3.14 e NumPy 2.5 instalados).
- **Bibliotecas permitidas:** apenas `numpy` (cálculo) e, na fase opcional,
  `matplotlib` (gráficos). **Proibido** Keras, TensorFlow, PyTorch,
  scikit-learn ou qualquer biblioteca que já implemente rede neural — o
  objetivo do trabalho é implementar a rede à mão.
- **Sem front-end web e sem backend.** É um programa de terminal.
- Código, nomes de variáveis, comentários e mensagens ao usuário em
  **português**.
- Código simples e didático: o grupo precisa explicar cada parte no vídeo.
  Prefira clareza a otimização. Comente as fórmulas.
- Lógica da rede separada da interface (pasta `rede/` não faz `input()` nem
  `print()`).
- Não inventar requisitos além deste plano. Em dúvida, perguntar ao usuário.

## 3. Decisões já tomadas (vieram das aulas)

| Tema | Decisão |
|---|---|
| Tipo de rede | MLP, camadas totalmente conectadas (todo neurônio liga a todos da próxima) |
| Camada de entrada | Só repassa os valores. **Sem** função de ativação |
| Camadas ocultas | Usuário escolhe quantidade, neurônios e ativação **de cada uma** |
| Ativações disponíveis nas ocultas | `sigmoid`, `relu`, `tanh` |
| Camada de saída | Ativação fixa no código (ver item 4.1) |
| Pesos iniciais | Aleatórios, uniforme entre −1 e 1 (`seed` configurável para repetir experimentos) |
| Matriz de pesos | Cada camada tem `W` com forma `(n_atual, n_anterior)` |
| Perda | MSE: `E = (1/N) · Σ (y − ŷ)²` |
| Derivada da perda | `∂E/∂ŷ = (2/N) · (ŷ − y)` (forma do professor) |
| Treinamento | Gradiente descendente com **base inteira por atualização** (gradientes somados sobre as N amostras, o 1/N já está na derivada) |
| Atualização | `W = W − η · ∂E/∂W` |

## 4. Pontos em aberto (usar o padrão indicado até o professor confirmar)

1. **Degrau na saída.** O professor desenhou a saída com degrau, mas o degrau
   não tem derivada útil e impede o backpropagation.
   **Padrão:** a camada de saída usa **sigmoid** no treino; o **degrau**
   (`1 se ŷ ≥ 0,5, senão 0`) é aplicado só para gerar a classe final na
   avaliação e na previsão. Deixar isso explícito em comentário no código.
2. **Bias.** Os slides não usam bias.
   **Padrão:** perguntar no prompt `Usar bias? (s/n)`. Bias começa em 0.
   Se não usar, `b` fica zerado e não é atualizado.
3. **NumPy permitido?** **Padrão:** sim. Se o professor proibir, as funções
   de `rede/` precisam ser reescritas com listas e laços; por isso manter as
   operações de matriz concentradas em poucas funções.
4. **Formato da base.** **Padrão:** CSV, separador vírgula, com ou sem
   cabeçalho (detectar: se a primeira linha não for numérica, é cabeçalho).
   As **últimas `k` colunas** são a saída esperada, onde `k` = número de
   neurônios da camada de saída. As demais são as entradas.
5. **Ajuste da camada oculta.** O professor ainda não deu esse slide.
   Implementar o backpropagation padrão (item 6.3) e conferir com a aula
   quando sair.

## 5. Estrutura de arquivos

```
RedesNeuraisApp/
├── main.py              # ponto de entrada: menu e perguntas (única parte com input/print)
├── rede/
│   ├── __init__.py
│   ├── ativacoes.py     # funções de ativação e derivadas
│   ├── camada.py        # classe Camada
│   ├── rede_neural.py   # classe RedeNeural (forward, backward, treino, previsão)
│   └── perda.py         # MSE e derivada
├── dados.py             # carregar CSV, normalizar, separar treino/teste
├── bases/
│   ├── xor.csv          # 4 linhas, para validar
│   └── and.csv          # 4 linhas, para validar
├── testes/
│   └── test_rede.py     # testes com unittest (sem pytest, para não exigir instalação)
├── requirements.txt     # numpy (matplotlib só na fase opcional)
└── README.md            # como instalar e rodar
```

## 6. Especificação por arquivo

### 6.1 `rede/ativacoes.py`

Cada função recebe e devolve `numpy.ndarray`.

| Função | Fórmula | Derivada (em função de `z` ou `a`) |
|---|---|---|
| `sigmoid(z)` | `1 / (1 + e^(−z))` — fazer `np.clip(z, -500, 500)` antes, evita overflow | `a · (1 − a)` |
| `relu(z)` | `max(0, z)` | `1 se z > 0, senão 0` |
| `tanh(z)` | `np.tanh(z)` | `1 − a²` |
| `degrau(z)` | `1 se z ≥ 0, senão 0` (para a saída: aplicar em `ŷ − 0,5`, ou função `classificar(ŷ)` com limiar 0,5) | não usada no treino |

Expor um dicionário para o menu escolher pelo nome:

```python
ATIVACOES = {
    "sigmoid": (sigmoid, derivada_sigmoid),
    "relu": (relu, derivada_relu),
    "tanh": (tanh, derivada_tanh),
}
```

Padronizar a assinatura das derivadas como `derivada(z, a)` para todas, mesmo
que cada uma use só um dos dois.

### 6.2 `rede/camada.py`

```python
class Camada:
    def __init__(self, n_entradas: int, n_neuronios: int, nome_ativacao: str,
                 usar_bias: bool, gerador: np.random.Generator): ...
    def forward(self, A_anterior: np.ndarray) -> np.ndarray: ...
    def backward(self, dA: np.ndarray) -> np.ndarray: ...
    def atualizar(self, taxa: float) -> None: ...
```

- `W`: forma `(n_neuronios, n_entradas)`, `gerador.uniform(-1, 1, ...)`.
- `b`: forma `(n_neuronios,)`, zeros.
- `forward` recebe `A_anterior` com forma `(N, n_entradas)` (N = amostras) e:
  - `Z = A_anterior @ W.T + b`
  - `A = f(Z)`
  - **guarda** `A_anterior`, `Z` e `A` (o backward precisa deles).
- `backward(dA)` recebe `∂E/∂A` desta camada, forma `(N, n_neuronios)`:
  - `dZ = dA * f'(Z, A)`
  - `self.dW = dZ.T @ A_anterior` (soma sobre as amostras)
  - `self.db = dZ.sum(axis=0)` (só se `usar_bias`)
  - devolve `dA_anterior = dZ @ W` — calcular **antes** de atualizar `W`.
- `atualizar(taxa)`: `W -= taxa * dW`; `b -= taxa * db` se usar bias.
- Método `numero_de_pesos()` devolve `n_neuronios × n_entradas` (para mostrar
  ao usuário; é a regra "quantidade de pesos verificável pela camada anterior").

### 6.3 `rede/rede_neural.py`

```python
class RedeNeural:
    def __init__(self, n_entradas: int, ocultas: list[tuple[int, str]],
                 n_saidas: int, usar_bias: bool, seed: int | None): ...
    def forward(self, X: np.ndarray) -> np.ndarray: ...
    def treinar(self, X, Y, epocas: int, taxa: float,
                ao_fim_da_epoca=None) -> list[float]: ...
    def prever(self, X) -> np.ndarray: ...       # ŷ contínuo (sigmoid)
    def classificar(self, X) -> np.ndarray: ...  # degrau(ŷ ≥ 0,5) → 0/1
    def resumo(self) -> list[str]: ...           # linhas descrevendo camadas e nº de pesos
```

- `ocultas`: lista de `(n_neuronios, nome_ativacao)`, uma por camada oculta.
- Monta as camadas encadeando `n_entradas` de cada uma com os neurônios da
  anterior. Camada de saída: `Camada(..., "sigmoid", ...)` fixa.
- Um único `np.random.default_rng(seed)` compartilhado por todas as camadas.
- `treinar`, por época:
  1. `Ŷ = forward(X)`
  2. `erro = mse(Y, Ŷ)` → guardar na lista de histórico
  3. `dA = derivada_mse(Y, Ŷ)` → `(2/N)(Ŷ − Y)`
  4. percorrer camadas **de trás para frente** chamando `backward`
  5. chamar `atualizar(taxa)` em todas
  6. se `ao_fim_da_epoca` foi passado, chamar com `(epoca, erro)` — o
     `main.py` usa isso para imprimir progresso (a pasta `rede/` não imprime)
- Devolve o histórico de erros.
- Permitir 0 camadas ocultas (vira um perceptron simples) — útil para mostrar
  no vídeo que ele **não** aprende XOR e a rede com oculta aprende.

### 6.4 `rede/perda.py`

- `mse(Y, Y_hat)`: `np.mean(np.sum((Y - Y_hat) ** 2, axis=1))`
  (média sobre amostras, soma sobre neurônios de saída; com 1 saída é
  exatamente a fórmula do professor).
- `derivada_mse(Y, Y_hat)`: `(2 / N) * (Y_hat - Y)`, `N = Y.shape[0]`.

### 6.5 `dados.py`

- `carregar_csv(caminho, n_saidas) -> (X, Y)`: lê com `np.genfromtxt` ou
  `csv`; detecta cabeçalho; últimas `n_saidas` colunas viram `Y`.
  Erros claros se o arquivo não existir ou as colunas não baterem.
- `normalizar(X) -> (X_norm, minimos, maximos)`: min-max para [0, 1] por
  coluna. Coluna constante → evitar divisão por zero (deixar 0).
  Guardar `minimos`/`maximos` para normalizar entradas digitadas depois.
  Vale para **X e Y**: as saídas também são normalizadas (escala do treino),
  e `desnormalizar(Y_norm, minimos, maximos)` devolve ŷ na escala original
  para mostrar ao usuário. Classes em one-hot (0/1) não mudam.
- `normalizar_por_lotes(X, tamanho_lote)`: divide as linhas em lotes e
  normaliza cada lote com o min/máx dele mesmo. Opcional, só para as
  entradas do treino e só em bases com 10+ linhas; o padrão (Enter) é a
  base inteira. Sobra de 1 linha entra no lote anterior. Teste e entradas
  digitadas usam a escala do treino inteiro.
- `separar_treino_teste(X, Y, proporcao_teste=0.2, seed)`: embaralha e separa.
  Se a base tiver menos de 10 linhas (XOR, AND), não separar: treina e testa
  na base inteira, e avisar o usuário disso.
- `acuracia(Y, Y_classificado)`: fração de amostras em que **todas** as
  saídas batem.

### 6.6 `main.py` — fluxo do prompt

Toda entrada validada: valor inválido → mensagem e pergunta de novo (nunca
quebrar com traceback). Criar funções auxiliares `perguntar_inteiro(msg,
minimo)`, `perguntar_float(msg, minimo)`, `perguntar_sn(msg)`,
`perguntar_opcao(msg, opcoes)`.

```
=== Construtor de Redes Neurais ===
Quer criar uma rede neural? (s/n)          → n: encerra
Quantos neurônios na camada de entrada?    → inteiro ≥ 1
Quantas camadas ocultas?                   → inteiro ≥ 0
  Camada oculta 1 — quantos neurônios?     → inteiro ≥ 1
  Camada oculta 1 — ativação (sigmoid/relu/tanh)?
  ... (repete para cada oculta)
Quantos neurônios na camada de saída?      → inteiro ≥ 1
Usar bias? (s/n)
Semente aleatória (Enter = aleatória):

[mostrar resumo]
  Entrada: 2 neurônios (sem ativação)
  Oculta 1: 3 neurônios, sigmoid — 2 × 3 = 6 pesos (+3 bias)
  Saída: 1 neurônio, sigmoid no treino / degrau na classificação — 3 × 1 = 3 pesos
  Total de pesos: 9

Caminho da base de dados (CSV):            → validar colunas = entradas + saídas
Taxa de aprendizado (η):                   → float > 0, sugerir 0.5
Número de épocas:                          → inteiro ≥ 1, sugerir 5000

[treino: imprimir erro a cada ~10% das épocas]
  Época  500/5000 — erro (MSE): 0.2481
  ...

[avaliação]
  Erro final: 0.0021
  Acurácia: 100,00% (4/4)
  Tabela: entrada | esperado | ŷ | classe | ok?

Menu final:
  1) Testar uma entrada digitada
  2) Treinar mais épocas
  3) Criar outra rede
  0) Sair
```

### 6.7 Bases de validação (`bases/`)

`xor.csv`:
```
x1,x2,y
0,0,0
0,1,1
1,0,1
1,1,0
```
`and.csv`: mesma forma, `y` = 1 só em `1,1`.

## 7. Testes (`testes/test_rede.py`, com `unittest`)

Rodar com `python -m unittest discover testes`.

1. **Ativações:** `sigmoid(0) == 0.5`; `derivada_sigmoid` em 0 vale 0,25;
   `relu(-2) == 0`, `relu(3) == 3`; `tanh(0) == 0`.
2. **Formas:** rede `2 → [3 sigmoid] → 1` tem `W` de formas `(3, 2)` e `(1, 3)`;
   `forward` de `X` com forma `(4, 2)` devolve `(4, 1)`.
3. **Exemplo da aula:** sem bias, entrada `(0, 0)`: ocultos dão `a = 0,5`
   qualquer que seja o peso; com pesos de saída `[-0.015, -0.902, 0.136]`
   (atribuídos manualmente), `ŷ ≈ 0.4036` (tolerância 1e-4).
4. **Verificação numérica do gradiente (o teste mais importante):** para uma
   rede pequena com bias e cada ativação, comparar `dW` do backprop com
   `(E(w + ε) − E(w − ε)) / (2ε)`, `ε = 1e-5`. Diferença relativa < 1e-5.
   Isso prova que o backpropagation está correto sem depender de slide.
5. **Aprendizado:** `AND` com 0 ocultas chega a 100%; `XOR` com
   `2 → [4 sigmoid] → 1`, bias, `seed=42`, η = 0,5, 10000 épocas chega a 100%.
   Se não convergir com esses valores, ajustar η/épocas/seed no teste e
   registrar no README — não alterar as fórmulas.

## 8. Ordem de implementação (cada etapa só termina com o critério cumprido)

| # | Etapa | Pronto quando |
|---|---|---|
| 1 | `ativacoes.py` + testes 1 | testes passam |
| 2 | `camada.py` (forward) + `rede_neural.py` (forward) + testes 2 e 3 | testes passam |
| 3 | `perda.py`, `backward`, `atualizar`, `treinar` + teste 4 | gradiente numérico bate |
| 4 | bases XOR/AND + teste 5 | XOR aprende 100% |
| 5 | `dados.py` | carrega `xor.csv`, normaliza, separa |
| 6 | `main.py` com o fluxo completo | rodar `python main.py` ponta a ponta com XOR sem erro; entradas inválidas não quebram |
| 7 | `README.md` + `requirements.txt` | outra pessoa do grupo consegue rodar só lendo o README |

Após cada etapa, mostrar ao usuário o que mudou e explicar as decisões
(o projeto também é de aprendizado).

## 9. Fora da versão inicial (só depois que a etapa 7 estiver pronta)

- Gráfico do erro por época com `matplotlib` (bom para o vídeo).
- Desenho da rede com espessura/cor das ligações proporcional aos pesos.
- Salvar e carregar a rede treinada (pesos em `.json` ou `.npz`).
- Ajustes conforme o professor confirmar os pontos em aberto (seção 4).

## 10. Roteiro sugerido do vídeo (para guiar a organização do código)

1. **Código-fonte:** estrutura de pastas; `ativacoes.py` (fórmulas e
   derivadas); `camada.py` (matriz de pesos, forward, o que é guardado);
   `rede_neural.py` (backpropagation de trás para frente); `perda.py` (MSE).
2. **Implementação:** por que Python + NumPy; por que sem bibliotecas de IA;
   decisões: sigmoid no treino e degrau na classificação, pesos aleatórios
   entre −1 e 1, seed, base inteira por atualização.
3. **Funcionamento:** rodar `main.py`, montar a rede pelo prompt, treinar
   XOR mostrando o erro caindo; mostrar que sem camada oculta o XOR não é
   aprendido; rodar com a base do professor.

Sugestão de divisão entre 4 pessoas: (A) ativações + perda, (B) camada +
rede/backprop, (C) dados + bases + testes, (D) main.py + README + vídeo.
Todos devem entender o backpropagation, porque o vídeo exige domínio do código.
