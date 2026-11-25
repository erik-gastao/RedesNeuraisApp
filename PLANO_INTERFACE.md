# Plano — Interface gráfica com a rede animada (player passo a passo)

Documento de especificação da parte gráfica. Complementa o `PLANO.md`:
tudo o que ele define para a rede continua valendo. Quem for executar
(pessoa ou outro modelo de IA) deve seguir as etapas na ordem.

Versão 2 (2026-11-19): funde o plano anterior (tkinter, treino ao vivo)
com o novo pedido — a rede é mostrada **como um vídeo**, com pausar,
voltar, avançar e parar num passo escolhido.

## 1. Objetivo

Um **hub** onde o usuário escolhe a base e monta a estrutura da rede, e
depois um **player** onde vê a rede sendo executada passo a passo:

- neurônios acendendo conforme a ativação de cada um;
- cada ligação acendendo quando seu `w · a` é calculado;
- soma, bias e ativação de cada neurônio, com a fórmula e os números reais;
- o erro sendo calculado na saída;
- o erro voltando da saída para a entrada no backpropagation;
- os pesos sendo ajustados (`W = W − η · dW`);
- a curva do erro ao longo das épocas.

Tudo controlado como um vídeo: ▶/⏸, passo para frente e para trás, barra
de tempo, velocidade, pontos de parada.

O objetivo é didático: a interface deve servir para explicar no vídeo o
que o código faz.

## 2. Decisões tomadas

| Tema | Decisão |
|---|---|
| Papel na entrega | **Interface é a entrega principal** (professor aceitou). O `main.py` no terminal continua funcionando como alternativa. |
| Tecnologia | **pygame-ce** (o `pygame` original pode não ter wheel para Python 3.14; o `pygame-ce` tem). |
| Campos e botões do hub | **pygame_gui** (caixa de texto, lista suspensa, slider, botões, tema). |
| Nível de detalhe | **Até cada ligação**: cada `w · a`, depois soma, bias, ativação. Pode ser reduzido para "neurônio" ou "camada" em redes grandes. |
| Amostras | **Uma amostra em foco**: o usuário escolhe a amostra e vê os números dela. O gradiente mostrado é o da base inteira (batch), como no código real. |
| Extras | Pontos de parada, painel de fórmulas com números, exportar GIF/MP4, comparar épocas lado a lado. |
| Exportação | GIF com **Pillow** (já instalado) e MP4 com **imageio + imageio-ffmpeg** (baixa o ffmpeg pelo pip). |
| Janela | **1280 × 720, redimensionável** (16:9, bom para gravar/exportar). |

Dependências novas no `requirements.txt`: `pygame-ce`, `pygame_gui`,
`imageio`, `imageio-ffmpeg`. `numpy`, `matplotlib` e `Pillow` já existem.

## 3. Regras

- **A rede não muda.** A interface só lê o que `rede/` já expõe:
  `rede.camadas[i].W`, `.b`, `.Z`, `.A`, `.A_anterior`, `.dW`, `.db`, e
  chama `forward`, `treinar`, `prever` e `classificar`. Nenhum cálculo de
  rede dentro da pasta `interface/` — os valores mostrados vêm da rede real.
- **O programa de terminal continua existindo e funcionando.** Os testes
  atuais continuam passando sem mudança.
- Português em código, comentários e textos da janela.
- Código simples e didático, como no `PLANO.md`.
- A janela nunca pode travar nem fechar com erro: valor inválido mostra
  mensagem na própria janela.
- Lógica separada do desenho: tudo que é cálculo (passos, posições, cores)
  fica em módulos **sem pygame**, testáveis com `unittest`.

## 4. Pontos em aberto (com padrão)

1. **Neurônios de entrada e saída.** **Padrão:** escolher a base primeiro e
   preencher entrada e saída sozinho, sem deixar editar (entrada = colunas
   de entrada; saída = 1 coluna numérica ou nº de classes). Elimina o erro
   de base incompatível.
2. **Redes grandes.** **Padrão:** desenhar no máximo 12 neurônios por
   camada; acima disso, os 11 primeiros, um "⋮" e o último, com o total
   escrito embaixo. Passos de ligações escondidas são agrupados.
3. **Quais épocas gravar em detalhe.** **Padrão:** 1, 2, 10, 100 e a última,
   editável no hub (campo "Épocas detalhadas: 1, 2, 10, 100, fim").

## 5. Arquitetura: gravar primeiro, tocar depois

Animar enquanto treina não permite voltar. Por isso a interface tem duas
partes:

1. **Gravador** (`gravador.py`, sem pygame): roda o treino real usando
   `Experimento` e, a cada época, lê os valores que as camadas já guardam.
   Gera:
   - **Resumo de toda época:** `W`, `b`, erro, acurácia (cópias numpy).
     Iris com 5000 épocas ≈ poucos MB.
   - **Passos detalhados** só das épocas escolhidas (ver 4.3).
2. **Player** (`player.py` + desenho): mantém só um índice `i` na lista de
   passos e desenha o passo `i`. Voltar = desenhar `i − 1`. Nada é
   recalculado, então barra de tempo, voltar e pontos de parada são simples.

### 5.1 O que é um passo

```python
@dataclass
class Passo:
    epoca: int
    fase: str           # "normalizacao" | "forward" | "erro" | "backward" | "atualizacao"
    tipo: str           # "entrada", "ligacao", "soma", "bias", "ativacao",
                        # "erro_saida", "dZ", "dW", "dA", "ajuste", ...
    camada: int | None
    neuronio: int | None
    origem: int | None  # neurônio da camada anterior (passos de ligação)
    valores: dict       # números do passo (w, a, produto, z, a, dW...)
    formula: str        # genérica:   "z = Σ w·a + b"
    formula_numeros: str  # com números: "z₂ = 0.40·0.80 + (−1.20)·0.10 + 0.30 = 0.50"
    texto: str          # explicação curta em português
```

Cada passo carrega tudo que precisa para ser desenhado sozinho. Para isso,
o estado da rede (pesos e ativações já calculadas até ali) é obtido pela
época do passo (resumo) mais o que veio antes dentro da mesma fase; o
`player` monta esse estado com uma função pura `estado_em(passos, i)`.

### 5.2 Hierarquia dos passos

```
Época
 └ Fase: Forward → Erro → Backward → Atualização
    └ Camada
       └ Neurônio
          └ Ligação (cada w·a)
```

Antes da época 1 há uma fase **"Normalização"** (só passos, sem treino):
valor original → `(x − mín) / (máx − mín)` → valor normalizado, para a
amostra em foco, em X e em Y, indicando o lote dela quando a normalização
for por lotes. No fim do forward, a saída mostra ŷ normalizado e ŷ
desnormalizado (escala original).

Sequência de um neurônio no forward (nível "ligação"):
`ligacao j=1` → … → `ligacao j=n` → `soma` → `bias` → `ativacao`.

Backward por camada (da saída para a entrada): `erro_saida` (ŷ − y) →
`dZ` de cada neurônio → `dW` de cada ligação → `dA` enviado para a camada
anterior. Atualização: `ajuste` por camada (espessuras mudam de uma vez).

Níveis menores ("neurônio", "camada") apenas juntam passos.

### 5.3 Amostra em foco

Treino é batch (`rede.treinar` usa a base inteira). Os passos de forward
mostram os números **da amostra em foco** (linha `k` de `Z`/`A` da
camada). Os passos de backward mostram `dZ` da amostra em foco nos
neurônios e o `dW` **da base inteira** nas ligações, com o texto
deixando isso claro ("dW soma a contribuição das 120 amostras").
Trocar a amostra em foco regrava só os passos detalhados (rápido, usa os
resumos já guardados).

## 6. Telas

### 6.1 Hub (configuração)

```
┌───────────────────────────── Construtor de Redes Neurais ─────────────────────┐
│ BASE              [bezdekIris.data ▼]   150 amostras · 4 entradas · 3 classes │
│                                                                               │
│ CAMADAS OCULTAS   1: [5 ] [sigmoid ▼] [−]                                     │
│                   [+ adicionar camada]                                        │
│ [x] Usar bias     Semente [1   ]   η [0.5  ]   Épocas [5000 ]                 │
│ NORMALIZAÇÃO      X e Y min-max · Lote [base inteira ▼] (só bases com 10+)    │
│                                                                               │
│ DETALHE           (•) ligação  ( ) neurônio  ( ) camada                       │
│ Épocas detalhadas [1, 2, 10, 100, fim]     Amostra em foco [12 ▼]             │
│                                                                               │
│ Prévia:  4 → [5 sigmoid] → 3      (35 pesos + 8 bias)                         │
│                                                     [ Gerar e abrir player ]  │
└───────────────────────────────────────────────────────────────────────────────┘
```

"Gerar" roda o gravador com barra de progresso (sem travar: grava em
pedaços entre quadros) e abre o player.

### 6.2 Player

```
┌──────────────────────────────────────────────────────┬────────────────────────┐
│                                                      │ ÉPOCA 1 · FORWARD      │
│   x1 ●──────╲                                        │ Camada 1 · neurônio 2  │
│              ●────╲                                  │                        │
│   x2 ●──────╱      ●  setosa      0.97               │ z = Σ w·a + b          │
│              ●────                                   │ z₂ = 0.40·0.80         │
│   x3 ●──────╲      ●  versicolor  0.03               │    + (−1.20)·0.10      │
│              ●────                                   │    + 0.30 = 0.50       │
│   x4 ●──────╱      ●  virginica   0.00               │ a = σ(0.50) = 0.62     │
│              ●                                       │                        │
│  Entrada   Oculta 1 (sigmoid)   Saída                │ "O neurônio soma as    │
│                                                      │ entradas pesadas..."   │
│                                                      ├────────────────────────┤
│                                                      │ erro  ╲__              │
│                                                      │        ‾‾‾‾  (época)   │
├──────────────────────────────────────────────────────┴────────────────────────┤
│ ⏮ ◀◀ ◀ ▶/⏸ ▶ ▶▶ ⏭   0.5x 1x 2x 4x        passo 1.234 / 8.910        ● parada │
│ [■■■■■forward■■■■|▓▓erro|□□□□backward□□□□|▒▒ajuste▒▒|■■■■forward ...]         │
│ [Hub] [Comparar épocas] [Exportar GIF/MP4] [Pontos de parada] [Amostra: 12 ▼] │
└───────────────────────────────────────────────────────────────────────────────┘
```

- **Centro:** rede desenhada.
- **Direita:** painel de fórmulas (genérica + com números + texto) e
  curva do erro (feita à mão em pygame, com marcador da época atual).
- **Embaixo:** controles e barra de tempo com faixas coloridas por fase;
  clicar/arrastar na barra pula para o passo.

## 7. Como a rede é desenhada

### 7.1 Posições

- Camadas em colunas igualmente espaçadas, da esquerda (entrada) para a
  direita (saída). Neurônios centralizados na vertical.
- Raio do neurônio entre 10 e 24 px conforme a maior camada.
- Uma linha de cada neurônio para todos da camada seguinte (MLP completa).
- Rótulos: nome da entrada (cabeçalho ou `x1..xn`) à esquerda; classe e ŷ
  à direita da saída; nome da camada e ativação embaixo de cada coluna.
- Recalculado ao redimensionar a janela.

### 7.2 Cores — visão "ativação" (forward)

| Elemento | Representa | Como |
|---|---|---|
| Preenchimento do neurônio | `a` da amostra em foco | sigmoid: 0 = branco, 1 = azul forte. tanh: −1 = laranja, 0 = branco, +1 = azul. relu: 0 = branco, maior valor da camada = azul forte |
| Neurônio de entrada | valor normalizado (0 a 1) | escala da sigmoid |
| Cor da ligação | sinal do peso `w` | positivo = azul, negativo = laranja |
| Espessura da ligação | `\|w\|` | 0,5 a 6 px, proporcional ao maior `\|w\|` da rede |
| Contorno do neurônio | bias (se ligado) | mesma regra das ligações |
| Saída | classe escolhida | contorno verde se acertou, vermelho se errou |
| Neurônios ainda não calculados | — | cinza claro |

Azul/laranja e não verde/vermelho para os pesos: legível para daltônicos.
Legenda fixa no canto.

### 7.3 Cores — visão "gradiente" (backward)

Ligação mostra `dW` (sinal = cor, tamanho = espessura, escala própria).
Neurônio mostra `dZ`. Mostra que o erro nasce na saída e volta
enfraquecendo. A visão troca sozinha conforme a fase do passo, e pode ser
forçada por um botão.

### 7.4 Destaque do passo atual

- Passo de **ligação**: a ligação brilha e um pulso anda da origem ao
  destino (da direita para a esquerda no backward); brilho ∝ `|w · a|`.
- Passo de **soma/bias/ativação**: o neurônio pulsa e "enche" até a cor
  da ativação; `z` e `a` aparecem ao lado.
- Passo de **ajuste**: flash nas ligações da camada, espessura interpolada
  do peso velho para o novo.
- Animações dentro de um passo são interpoladas pelo tempo (tween), então
  voltar/pular mostra o passo já no estado final.

### 7.5 Tooltip

Mouse sobre neurônio: camada, índice, `z`, `a`, bias, `dZ`.
Mouse sobre ligação: de onde para onde, `w`, `dW`.

## 8. Controles do player

| Controle | Tecla | Ação |
|---|---|---|
| ▶/⏸ | espaço | tocar/pausar |
| ◀ / ▶ | ← / → | passo anterior / próximo |
| ◀◀ / ▶▶ | Shift + ← / → | neurônio/camada anterior / próxima (conforme nível) |
| ⏮ / ⏭ | Home / End ou PgUp / PgDn | fase anterior / próxima; com Ctrl, época |
| Velocidade | − / + | 0,25x, 0,5x, 1x, 2x, 4x (passos por segundo) |
| Barra de tempo | clique/arraste | pula para o passo |
| Amostra em foco | lista | regrava os passos detalhados |
| Visão | V | ativação / gradiente |

Entre épocas detalhadas, a barra mostra "saltos" (ex.: época 10 → 100):
tocar ali anima só o resumo (pesos mudando época a época, curva crescendo),
sem passos internos.

## 9. Extras

### 9.1 Pontos de parada

Janela "Pontos de parada" com lista de regras (ligar/desligar cada uma):

- início de uma fase (forward, erro, backward, atualização);
- chegar na época N;
- chegar na camada C (opcional: neurônio K);
- erro menor que um valor;
- clicar com botão direito num neurônio/ligação: "parar aqui".

Ao tocar, o player verifica as regras a cada passo e pausa no primeiro que
casar. Marcas vermelhas na barra de tempo. Lógica em `pontos_parada.py`
(sem pygame): `def deve_parar(passo, regras) -> bool`.

### 9.2 Painel de fórmulas

Mostra `formula`, `formula_numeros` e `texto` do passo atual. Números com
2 casas; sinal negativo entre parênteses. Fonte monoespaçada.

### 9.3 Comparar épocas

Tela dividida: mesma rede desenhada duas vezes com os pesos de duas épocas
escolhidas (padrão: 1 e a última), mesma amostra em foco, ŷ de cada uma
embaixo. Usa só os resumos.

### 9.4 Exportar GIF/MP4

Escolher trecho (passo inicial e final, ou "fase atual", "época atual"),
formato, FPS (padrão 30) e resolução (padrão 1280 × 720). Renderiza fora
da tela numa `Surface`, quadro a quadro, sem depender da velocidade real:

- GIF: Pillow (`Image.save(..., save_all=True)`).
- MP4: `imageio.get_writer(..., fps=30)` com imageio-ffmpeg.

Barra de progresso e botão cancelar. Arquivo salvo em `exportacoes/`.

## 10. Estrutura de arquivos

```
RedesNeuraisApp/
├── interface_grafica.py     # ponto de entrada: python interface_grafica.py
├── executar_interface.bat   # dois cliques (Windows)
├── experimento.py           # NOVO, compartilhado com main.py (ver 10.1)
├── interface/
│   ├── __init__.py
│   │   # --- sem pygame (testáveis) ---
│   ├── gravador.py          # gera resumos por época e lista de Passo
│   ├── passos.py            # dataclass Passo, agrupar por nível, estado_em()
│   ├── layout_rede.py       # posições, cores, espessuras
│   ├── pontos_parada.py     # regras e deve_parar()
│   ├── formulas.py          # monta formula/formula_numeros/texto
│   │   # --- com pygame ---
│   ├── app.py               # laço principal, troca entre telas
│   ├── tela_hub.py          # hub de configuração (pygame_gui)
│   ├── tela_player.py       # player: junta desenho, controles, painéis
│   ├── tela_comparar.py     # comparar épocas
│   ├── desenho_rede.py      # desenha a rede e o destaque do passo
│   ├── barra_tempo.py       # controles + barra com faixas de fase
│   ├── painel_formulas.py
│   ├── grafico_erro.py      # curva do erro desenhada em pygame
│   ├── exportar.py          # GIF/MP4
│   └── tema.json            # tema do pygame_gui
└── testes/
    ├── test_experimento.py
    ├── test_gravador.py
    ├── test_layout_rede.py
    ├── test_pontos_parada.py
    └── test_interface.py    # fumaça com SDL_VIDEODRIVER=dummy
```

### 10.1 `experimento.py` — evitar duplicar o `main.py`

Extrair do `main.py` a sequência carregar → separar → normalizar → montar
→ treinar → avaliar para uma classe sem `input`/`print`:

```python
class Experimento:
    def __init__(self, caminho_base, ocultas, usar_bias, seed): ...
    def treinar(self, epocas, taxa, ao_fim_da_epoca=None) -> list[float]: ...
    def avaliar(self) -> dict: ...       # erro, acurácia treino/teste, linhas da tabela
    def prever_entrada(self, valores) -> tuple: ...   # normaliza e devolve (ŷ, classe)
    def reiniciar(self): ...
```

O `main.py` passa a usar `Experimento`. `test_main.py` deve continuar
passando sem mudança.

### 10.2 Gravador

```python
def gravar(experimento, epocas, taxa, epocas_detalhadas, nivel, amostra) -> Gravacao
```

- Usa `ao_fim_da_epoca` do `treinar` para copiar `W`, `b`, erro. Para
  as épocas detalhadas, precisa de `Z`, `A`, `dZ`/`dW` **antes** da
  atualização: o gravador treina uma época por vez (`treinar(X, Y, 1, η)`)
  e copia o que as camadas guardaram (`A_anterior`, `Z`, `A`, `dW`, `db`)
  junto com o `W` de antes do ajuste. Mesmo resultado que treinar de uma
  vez (testado).
- `dZ` da amostra em foco é recalculado no gravador a partir de `Z`, `A`
  e `dA` com as mesmas funções de `rede/ativacoes.py`, ou guardado pela
  camada se for exposto (decidir na etapa 2, sem mudar o comportamento).

### 10.3 Desempenho

- Desenho com 60 FPS alvo; a rede é desenhada numa `Surface` de fundo
  que só é refeita quando os pesos ou o tamanho mudam; o destaque do
  passo vai por cima.
- Limite de 12 neurônios por camada segura redes grandes.
- Gravação: Iris, nível ligação, 5 épocas detalhadas ≈ dezenas de
  milhares de passos — ok em memória. Se passar de ~200 mil passos, o
  hub avisa e sugere nível "neurônio".

## 11. Ordem de implementação

| # | Etapa | Pronto quando |
|---|---|---|
| 1 | `experimento.py` + `main.py` usando ele + `test_experimento.py` | testes antigos passam sem mudança, novos passam |
| 2 | `passos.py`, `formulas.py`, `gravador.py` + `test_gravador.py` | valores gravados batem com `rede.forward` e com `dW` real; treinar época a época = treinar de uma vez |
| 3 | `layout_rede.py` + `test_layout_rede.py` | posições, cores, espessuras, "⋮" testados |
| 4 | `app.py` + hub (pygame_gui) + rede estática | escolher Iris, montar 4 → 5 → 3, ver a rede com pesos iniciais |
| 5 | Player: barra de tempo, ◀ ▶, ▶/⏸, velocidade, atalhos | navegar para frente e para trás em XOR sem erro |
| 6 | Forward animado + painel de fórmulas | ver cada `w·a`, soma, bias, ativação com números certos |
| 7 | Erro, backward (visão gradiente) e ajuste animados | ver `dW` nascendo na saída e voltando; ajuste muda espessuras |
| 8 | Saltos entre épocas + curva do erro | tocar da época 10 à 100 mostra pesos e curva mudando |
| 9 | Pontos de parada + `test_pontos_parada.py` | "parar no backward da época 100" pausa no passo certo |
| 10 | Comparar épocas | época 1 × última lado a lado |
| 11 | Exportar GIF/MP4 | exportar a fase forward da época 1 do XOR em GIF e MP4 |
| 12 | Tooltip, legenda, tema, `executar_interface.bat`, README | outra pessoa do grupo usa só lendo o README |

Após cada etapa, mostrar ao usuário o que mudou e explicar as decisões.

## 12. Testes

- `test_experimento.py`: XOR 100%, Iris ≥ 90%, entrada/saída vindas da
  base, `prever_entrada` normaliza igual ao treino.
- `test_gravador.py`: `z` e `a` dos passos = `Z`/`A` reais da amostra em
  foco; soma dos produtos `w·a` dos passos de ligação + bias = `z`; `dW`
  dos passos = `dW` real; pesos do resumo da época N = pesos ao treinar N
  épocas de uma vez; níveis menores têm menos passos e mesma ordem de fases.
- `test_layout_rede.py`: posições dentro da área; espaçamento igual;
  `neuronios_visiveis(30)` = 11 índices + `None` + último; sigmoid 0 =
  branco, 1 = azul; peso negativo = laranja; maior peso = espessura máxima.
- `test_pontos_parada.py`: cada tipo de regra casa no passo certo e só nele.
- `test_interface.py`: com `SDL_VIDEODRIVER=dummy`, abre o app, gera XOR,
  avança e volta 50 passos chamando o laço direto, exporta 10 quadros em
  GIF numa pasta temporária, fecha. Pulado se pygame não estiver instalado.
- Teste visual (cores, animação) é manual: roteiro no fim de cada etapa.

## 13. Fora deste plano

- Desenhar redes com centenas de neurônios por camada.
- Editar pesos clicando nas ligações.
- Salvar/carregar rede treinada (seção 9 do `PLANO.md`; quando existir,
  ganha botões no hub).
- Treino por amostra (SGD) — mudaria `rede/` e os resultados do terminal.

## 14. Uso no vídeo

1. Abrir o hub, escolher XOR, montar 2 → [4 sigmoid] → 1, gerar.
2. Época 1, forward: passo a passo uma amostra, mostrando cada `w·a` e a
   fórmula; saída perto de 0,5, errada.
3. Fase erro e backward: o erro voltando pela rede (visão gradiente).
4. Ajuste dos pesos: espessuras mudando.
5. Tocar os saltos até a última época: curva do erro caindo, platô e queda.
6. Forward na última época: agora acerta.
7. Comparar épocas 1 × última.
8. Montar sem camada oculta e mostrar que o XOR não é aprendido.
9. Iris (4 → 5 → 3), ponto de parada "erro < 0,05", amostra setosa.
10. Trechos exportados em MP4 para editar o vídeo final.
