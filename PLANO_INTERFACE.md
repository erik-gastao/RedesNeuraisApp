# Plano — Interface gráfica com a rede animada (player passo a passo)

Documento de especificação da parte gráfica. Complementa o `README.md` (que
descreve a rede e o terminal) e o enunciado do Trabalho Prático 02. Quem for
executar (pessoa ou outro modelo de IA) deve seguir as etapas na ordem.

Versão 3 (2026-11-25): adaptada ao núcleo novo, que já atende o enunciado —
entidade `Neuronio` com delta, treino **SGD** (pesos ajustados a cada
exemplo), divisão **treino/validação/teste**, normalização **opcional só das
entradas**, ativação da saída configurável (inclusive `linear`, para
regressão) e salvar/carregar a rede em JSON. A versão 2 (player tipo vídeo
em pygame) continua valendo no que não conflita com isso.

## 1. Objetivo

Um **hub** onde o usuário escolhe a base e monta a rede (ou carrega uma
rede salva), e um **player** onde vê a rede sendo treinada passo a passo:

- neurônios acendendo conforme a ativação de cada um;
- cada ligação acendendo quando seu `w · a` é calculado;
- soma, bias e ativação de cada neurônio, com a fórmula e os números reais;
- o erro sendo calculado na saída;
- o delta (δ) voltando da saída para a entrada na retropropagação;
- os pesos sendo ajustados logo depois de cada exemplo (`w = w − η · δ · a`);
- as curvas do erro de **treino e validação** ao longo das épocas;
- a previsão de uma entrada digitada, com a mesma animação do forward.

Tudo controlado como um vídeo: ▶/⏸, passo para frente e para trás, barra
de tempo, velocidade, pontos de parada.

O objetivo é didático: a interface deve servir para explicar no vídeo o
que o código faz, e permitir executar **todas** as funcionalidades do
enunciado sem mexer no código (restrição 5).

## 2. Decisões tomadas

| Tema | Decisão |
|---|---|
| Papel na entrega | **Interface é a entrega principal** (professor aceitou). O `main.py` no terminal continua funcionando como alternativa. |
| Tecnologia | **pygame-ce** (o `pygame` original pode não ter wheel para Python 3.14; o `pygame-ce` tem). Streamlit foi descartado: recarrega a página a cada clique e não sustenta a animação tipo vídeo. |
| Campos e botões do hub | **pygame_gui** (caixa de texto, lista suspensa, slider, botões, tema). |
| Treino mostrado | **SGD, um exemplo por vez**, na mesma ordem embaralhada que `rede/treinamento.treinar` usa com a mesma semente. |
| Nível de detalhe | **Até cada ligação** para o exemplo em foco. Pode ser reduzido para "neurônio" ou "camada" em redes grandes. |
| Exemplo em foco | Nas épocas detalhadas, o exemplo escolhido é mostrado em detalhe (forward, erro, retropropagação, ajuste); os outros exemplos da época aparecem como **um passo compacto cada** ("exemplo 7: ŷ, erro, pesos ajustados"), para deixar claro que os pesos mudam a cada exemplo. |
| Divisão da base | Percentuais treino/validação/teste no hub (padrão 70/15/15, somando 100), com a semente. Bases com menos de 10 linhas não são divididas (como no terminal). |
| Normalização | Caixa "Normalizar entradas (min-max do treino)". As saídas nunca são normalizadas. Lote opcional, como no terminal. |
| Ativação da saída | Lista no hub. Padrão sugerido pela base: `linear` se a saída é numérica (regressão), `sigmoid` se é 0/1 ou classe em texto. |
| Salvar/carregar | Botões "Salvar rede" (player) e "Carregar rede" (hub), usando `persistencia.py` (mesmo JSON do terminal). |
| Extras | Pontos de parada, painel de fórmulas com números, exportar GIF/MP4, comparar épocas lado a lado. |
| Exportação | GIF com **Pillow** e MP4 com **imageio + imageio-ffmpeg** (baixa o ffmpeg pelo pip). |
| Janela | **1280 × 720, redimensionável** (16:9, bom para gravar/exportar). |

Dependências novas no `requirements.txt`: `pygame-ce`, `pygame_gui`,
`Pillow`, `imageio`, `imageio-ffmpeg`. `numpy` e `matplotlib` já existem.
Conferir na etapa 4 se o `pygame-ce` instala no Python 3.14.

## 3. Regras

- **A rede não muda.** A interface só lê o que o núcleo já expõe e chama
  seus métodos. Nenhum cálculo de rede dentro da pasta `interface/` — os
  valores mostrados vêm da rede real:
  - leitura: `rede.camadas[i].neuronios[j]` → `.pesos`, `.bias`,
    `.entradas`, `.z`, `.a`, `.delta`; `camada.W`, `camada.b`,
    `camada.deltas`, `camada.nome_ativacao`; `rede.ativacao_saida`;
  - chamadas: `rede.forward`, `rede.retropropagar`, `rede.atualizar`,
    `rede.treinar_exemplo`, `rede.prever`, `rede.classificar`,
    `rede.treinamento.treinar`, `dados.*`, `persistencia.salvar_rede` /
    `carregar_rede`, `rede.perda.mse`.
- **O programa de terminal continua existindo e funcionando.** Os testes
  atuais continuam passando sem mudança.
- Português em código, comentários e textos da janela.
- Código simples e didático, como o resto do projeto.
- A janela nunca pode travar nem fechar com erro: valor inválido mostra
  mensagem na própria janela (percentuais que não somam 100, arquivo de rede
  inválido, base incompatível…).
- Lógica separada do desenho: tudo que é cálculo (passos, posições, cores)
  fica em módulos **sem pygame**, testáveis com `unittest`.

## 4. Pontos em aberto (com padrão)

1. **Neurônios de entrada e saída.** **Padrão:** escolher a base primeiro e
   preencher a entrada sozinho (colunas de entrada). Saída: classe em texto
   → nº de classes, sem editar; saída numérica → campo "Colunas de saída"
   (padrão 1), porque o CSV não diz quantas das últimas colunas são saída.
2. **Redes grandes.** **Padrão:** desenhar no máximo 12 neurônios por
   camada; acima disso, os 11 primeiros, um "⋮" e o último, com o total
   escrito embaixo. Passos de ligações escondidas são agrupados.
3. **Quais épocas gravar em detalhe.** **Padrão:** 1, 2, 10, 100 e a última,
   editável no hub (campo "Épocas detalhadas: 1, 2, 10, 100, fim").
4. **Exemplo em foco.** **Padrão:** o primeiro exemplo do treino; lista no
   hub e no player. Trocar regrava só os passos detalhados.

## 5. Arquitetura: gravar primeiro, tocar depois

Animar enquanto treina não permite voltar. Por isso a interface tem duas
partes:

1. **Gravador** (`gravador.py`, sem pygame): roda o treino SGD real, um
   exemplo por vez, e copia o que os neurônios guardam. Gera:
   - **Resumo de toda época:** `W` e `b` de cada camada, MSE de treino e de
     validação, acurácia (classificação). Iris com 1000 épocas ≈ poucos MB.
   - **Passos detalhados** só das épocas escolhidas (ver 4.3): completos
     para o exemplo em foco, compactos para os demais.
2. **Player** (`player.py` + desenho): mantém só um índice `i` na lista de
   passos e desenha o passo `i`. Voltar = desenhar `i − 1`. Nada é
   recalculado, então barra de tempo, voltar e pontos de parada são simples.

### 5.1 O que é um passo

```python
@dataclass
class Passo:
    epoca: int
    exemplo: int | None   # posição do exemplo na ordem SGD da época (None fora do treino)
    fase: str             # "normalizacao" | "forward" | "erro" | "retropropagacao" | "atualizacao" | "exemplo"
    tipo: str             # "entrada", "ligacao", "soma", "bias", "ativacao",
                          # "erro_saida", "delta", "gradiente", "ajuste", "exemplo_compacto", ...
    camada: int | None
    neuronio: int | None
    origem: int | None    # neurônio da camada anterior (passos de ligação)
    valores: dict         # números do passo (w, a, produto, z, a, δ, gradiente...)
    formula: str          # genérica:   "z = Σ w·a + b"
    formula_numeros: str  # com números: "z₂ = 0.40·0.80 + (−1.20)·0.10 + 0.30 = 0.50"
    texto: str            # explicação curta em português
```

Cada passo carrega tudo que precisa para ser desenhado sozinho. O estado da
rede (pesos antes do exemplo e ativações já calculadas) vem do próprio
gravador; o `player` monta esse estado com uma função pura
`estado_em(passos, i)`.

### 5.2 Hierarquia dos passos

```
Época
 └ Exemplo (ordem SGD)          ← compacto, exceto o exemplo em foco
    └ Fase: Forward → Erro → Retropropagação → Atualização
       └ Camada
          └ Neurônio
             └ Ligação (cada w·a)
```

Antes da época 1 há uma fase **"Normalização"** (só se a normalização
estiver ligada): valor original → `(x − mín) / (máx − mín)` → valor
normalizado, para o exemplo em foco, só nas entradas, indicando o lote dele
quando a normalização for por lotes. A saída esperada aparece sempre na
escala original.

Sequência de um neurônio no forward (nível "ligação"):
`ligacao j=1` → … → `ligacao j=n` → `soma` → `bias` → `ativacao`.

Retropropagação (da saída para a entrada), como na Aula 13:
`erro_saida` (`∂E/∂ŷ = 2(ŷ − y)`) → `delta` de cada neurônio de saída
(`δ = ∂E/∂ŷ · f'(z)`) → `delta` das ocultas (`δⱼ = f'(zⱼ) · Σᵢ δᵢ wᵢⱼ`, com
os pesos de **antes** do ajuste) → `gradiente` de cada ligação (`δ · a`).
Atualização: `ajuste` por camada (`w = w − η · δ · a`, `b = b − η · δ`;
espessuras mudam de uma vez), só depois de todos os deltas.

Níveis menores ("neurônio", "camada") apenas juntam passos.

### 5.3 Exemplo em foco

Com SGD, cada exemplo tem seu próprio forward, erro, deltas e ajuste: os
números mostrados são **do próprio exemplo**, e o gradiente de uma ligação
é `δ · a` desse exemplo (não há soma sobre a base). O exemplo em foco é
mostrado em detalhe; os outros viram um passo compacto com ŷ, erro e a
mudança dos pesos (espessuras se ajustam a cada um). Trocar o exemplo em
foco regrava só os passos detalhados (rápido: o gravador repete o treino
com a mesma semente).

## 6. Telas

### 6.1 Hub (configuração)

```
┌───────────────────────────── Construtor de Redes Neurais ─────────────────────┐
│ BASE              [dataset.csv ▼]   179 amostras · 2 entradas · saída numérica│
│                   Colunas de saída [1 ]                  [ Carregar rede… ]   │
│ CAMADAS OCULTAS   1: [8 ] [relu ▼] [−]                                        │
│                   [+ adicionar camada]                                        │
│ SAÍDA             1 neurônio · ativação [linear ▼]                            │
│ [x] Usar bias     Semente [1   ]   η [0.00001]   Épocas [1000 ]               │
│ DIVISÃO           Treino [70] Validação [15] Teste [15] %  → 125 / 27 / 27    │
│ NORMALIZAÇÃO      [x] Entradas min-max (escala do treino) · Lote [treino ▼]   │
│                                                                               │
│ DETALHE           (•) ligação  ( ) neurônio  ( ) camada                       │
│ Épocas detalhadas [1, 2, 10, 100, fim]     Exemplo em foco [1 ▼]              │
│                                                                               │
│ Prévia:  2 → [8 relu] → 1 linear      (24 pesos + 9 bias)                     │
│                                                     [ Gerar e abrir player ]  │
└───────────────────────────────────────────────────────────────────────────────┘
```

- Percentuais que não somam 100, lote inválido ou base incompatível mostram
  a mensagem embaixo do campo e desabilitam "Gerar".
- Saída numérica com ativação `sigmoid`: aviso "a saída não será
  normalizada; use linear".
- "Gerar" roda o gravador com barra de progresso (sem travar: grava em
  pedaços entre quadros) e abre o player.
- "Carregar rede…" abre um arquivo `.json` e vai direto para o player em
  **modo previsão** (sem linha do tempo de treino; ver 9.5).

### 6.2 Player

```
┌──────────────────────────────────────────────────────┬────────────────────────┐
│                                                      │ ÉPOCA 1 · EXEMPLO 12   │
│   x1 ●──────╲                                        │ FORWARD                │
│              ●────╲                                  │ Camada 1 · neurônio 2  │
│   x2 ●──────╱      ●  setosa      0.97               │                        │
│              ●────                                   │ z = Σ w·a + b          │
│   x3 ●──────╲      ●  versicolor  0.03               │ z₂ = 0.40·0.80         │
│              ●────                                   │    + (−1.20)·0.10      │
│   x4 ●──────╱      ●  virginica   0.00               │    + 0.30 = 0.50       │
│              ●                                       │ a = σ(0.50) = 0.62     │
│  Entrada   Oculta 1 (sigmoid)   Saída (sigmoid)      ├────────────────────────┤
│                                                      │ erro ╲__  treino       │
│                                                      │       ‾‾‾ validação    │
├──────────────────────────────────────────────────────┴────────────────────────┤
│ ⏮ ◀◀ ◀ ▶/⏸ ▶ ▶▶ ⏭   0.5x 1x 2x 4x        passo 1.234 / 8.910        ● parada │
│ [■forward■|▓erro|□retro□|▒ajuste|·ex·ex·ex·ex·|■forward ...]                   │
│ [Hub] [Prever entrada] [Salvar rede] [Comparar épocas] [Exportar] [Paradas]   │
└───────────────────────────────────────────────────────────────────────────────┘
```

- **Centro:** rede desenhada.
- **Direita:** painel de fórmulas (genérica + com números + texto) e
  curvas do erro de treino e validação (feitas à mão em pygame, com
  marcador da época atual).
- **Embaixo:** controles e barra de tempo com faixas coloridas por fase
  (os exemplos compactos aparecem como pontinhos); clicar/arrastar na barra
  pula para o passo.
- Na regressão, a saída mostra ŷ e o esperado na escala original (ex.:
  `ŷ 2558,97 · esperado 2566,30 kg/ha`), sem classe.

## 7. Como a rede é desenhada

### 7.1 Posições

- Camadas em colunas igualmente espaçadas, da esquerda (entrada) para a
  direita (saída). Neurônios centralizados na vertical.
- Raio do neurônio entre 10 e 24 px conforme a maior camada.
- Uma linha de cada neurônio para todos da camada seguinte (MLP completa).
- Rótulos: nome da entrada (cabeçalho ou `x1..xn`) à esquerda; classe e ŷ
  (ou ŷ e esperado, na regressão) à direita da saída; nome da camada e
  ativação embaixo de cada coluna.
- Recalculado ao redimensionar a janela.

### 7.2 Cores — visão "ativação" (forward)

| Elemento | Representa | Como |
|---|---|---|
| Preenchimento do neurônio | `a` do exemplo | sigmoid: 0 = branco, 1 = azul forte. tanh: −1 = laranja, 0 = branco, +1 = azul. relu: 0 = branco, maior valor da camada = azul forte. linear: escala divergente pelo maior `\|a\|` da camada (negativo = laranja, positivo = azul) |
| Neurônio de entrada | valor que entra na rede (normalizado se a normalização estiver ligada) | escala da relu (0 = branco, maior valor = azul) |
| Cor da ligação | sinal do peso `w` | positivo = azul, negativo = laranja |
| Espessura da ligação | `\|w\|` | 0,5 a 6 px, proporcional ao maior `\|w\|` da rede |
| Contorno do neurônio | bias (se ligado) | mesma regra das ligações |
| Saída | classe escolhida | contorno verde se acertou, vermelho se errou (só classificação) |
| Neurônios ainda não calculados | — | cinza claro |

Azul/laranja e não verde/vermelho para os pesos: legível para daltônicos.
Legenda fixa no canto.

### 7.3 Cores — visão "gradiente" (retropropagação)

Neurônio mostra o delta `δ` (sinal = cor, tamanho = intensidade, escala
própria). Ligação mostra o gradiente `δ · a` do exemplo. Mostra que o erro
nasce na saída e volta enfraquecendo. A visão troca sozinha conforme a fase
do passo, e pode ser forçada por um botão.

### 7.4 Destaque do passo atual

- Passo de **ligação**: a ligação brilha e um pulso anda da origem ao
  destino (da direita para a esquerda na retropropagação); brilho ∝ `|w · a|`.
- Passo de **soma/bias/ativação**: o neurônio pulsa e "enche" até a cor
  da ativação; `z` e `a` aparecem ao lado.
- Passo de **delta**: o neurônio pulsa na visão gradiente e `δ` aparece ao lado.
- Passo de **ajuste**: flash nas ligações da camada, espessura interpolada
  do peso velho para o novo.
- Passo **compacto** (outro exemplo): flash rápido na rede inteira e
  espessuras atualizadas.
- Animações dentro de um passo são interpoladas pelo tempo (tween), então
  voltar/pular mostra o passo já no estado final.

### 7.5 Tooltip

Mouse sobre neurônio: camada, índice, `z`, `a`, bias, `δ`.
Mouse sobre ligação: de onde para onde, `w`, gradiente `δ · a`.

## 8. Controles do player

| Controle | Tecla | Ação |
|---|---|---|
| ▶/⏸ | espaço | tocar/pausar |
| ◀ / ▶ | ← / → | passo anterior / próximo |
| ◀◀ / ▶▶ | Shift + ← / → | neurônio/camada anterior / próxima (conforme nível) |
| ⏮ / ⏭ | Home / End ou PgUp / PgDn | fase anterior / próxima; com Ctrl, exemplo; com Ctrl+Shift, época |
| Velocidade | − / + | 0,25x, 0,5x, 1x, 2x, 4x (passos por segundo) |
| Barra de tempo | clique/arraste | pula para o passo |
| Exemplo em foco | lista | regrava os passos detalhados |
| Visão | V | ativação / gradiente |

Entre épocas detalhadas, a barra mostra "saltos" (ex.: época 10 → 100):
tocar ali anima só o resumo (pesos mudando época a época, curvas
crescendo), sem passos internos.

## 9. Extras

### 9.1 Pontos de parada

Janela "Pontos de parada" com lista de regras (ligar/desligar cada uma):

- início de uma fase (forward, erro, retropropagação, atualização);
- chegar na época N (opcional: exemplo K);
- chegar na camada C (opcional: neurônio K);
- erro de treino ou de validação menor que um valor;
- clicar com botão direito num neurônio/ligação: "parar aqui".

Ao tocar, o player verifica as regras a cada passo e pausa no primeiro que
casar. Marcas vermelhas na barra de tempo. Lógica em `pontos_parada.py`
(sem pygame): `def deve_parar(passo, regras) -> bool`.

### 9.2 Painel de fórmulas

Mostra `formula`, `formula_numeros` e `texto` do passo atual. Números com
2 casas (na regressão, as que a escala pedir); sinal negativo entre
parênteses. Fonte monoespaçada.

### 9.3 Comparar épocas

Tela dividida: mesma rede desenhada duas vezes com os pesos de duas épocas
escolhidas (padrão: 1 e a última), mesmo exemplo em foco, ŷ de cada uma
embaixo. Usa só os resumos.

### 9.4 Exportar GIF/MP4

Escolher trecho (passo inicial e final, ou "fase atual", "exemplo atual",
"época atual"), formato, FPS (padrão 30) e resolução (padrão 1280 × 720).
Renderiza fora da tela numa `Surface`, quadro a quadro, sem depender da
velocidade real:

- GIF: Pillow (`Image.save(..., save_all=True)`).
- MP4: `imageio.get_writer(..., fps=30)` com imageio-ffmpeg.

Barra de progresso e botão cancelar. Arquivo salvo em `exportacoes/`
(já no `.gitignore`).

### 9.5 Salvar, carregar e prever (seções 1.11 e 1.12 do enunciado)

- **Salvar rede** (player): escolhe o arquivo (padrão `redes/<base>.json`)
  e chama `persistencia.salvar_rede(caminho, rede, normalizacao, classes)`.
- **Carregar rede** (hub): `persistencia.carregar_rede(caminho)`; erro de
  arquivo mostra a mensagem no hub. Abre o player em **modo previsão**: rede
  desenhada com os pesos salvos, sem linha do tempo de treino.
- **Prever entrada** (player, nos dois modos): caixa com um campo por
  entrada (valida a quantidade e os números). Aplica a normalização salva,
  gera os passos só de forward para essa entrada (`gravador.passos_previsao`)
  e toca a animação; mostra ŷ (e a classe, na classificação). Nenhum peso
  muda.

## 10. Estrutura de arquivos

```
RedesNeuraisApp/
├── interface_grafica.py     # ponto de entrada: python interface_grafica.py
├── executar_interface.bat   # dois cliques (Windows)
├── experimento.py           # NOVO, compartilhado com main.py (ver 10.1)
├── interface/
│   ├── __init__.py
│   │   # --- sem pygame (testáveis) ---
│   ├── gravador.py          # resumos por época, lista de Passo, passos_previsao
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
│   ├── painel_previsao.py   # campos da entrada digitada
│   ├── grafico_erro.py      # curvas de treino e validação em pygame
│   ├── exportar.py          # GIF/MP4
│   └── tema.json            # tema do pygame_gui
└── testes/
    ├── test_experimento.py
    ├── test_gravador.py
    ├── test_layout_rede.py
    ├── test_pontos_parada.py
    └── test_interface.py    # fumaça com SDL_VIDEODRIVER=dummy
```

### 10.1 `experimento.py` — evitar duplicar o `main.py` (feito na etapa 1)

A sequência carregar → dividir → normalizar (opcional) → treinar → avaliar
→ prever/salvar virou uma classe sem `input`/`print`. Ela funciona em fases,
porque o terminal pergunta a divisão e a normalização depois de carregar a
base (e valida os percentuais contra ela):

```python
exp = Experimento(rede, X, Y, classes, nome_base, seed)   # ValueError se a base não bate com a rede
exp.dividir(percentuais=(70, 15, 15))                     # ValueError se inválidos; base < 10 linhas não divide
exp.normalizar_entradas(tamanho_lote=None)                # opcional; min-max do treino, só entradas
exp.treinar(epocas, taxa, ao_fim_da_epoca=None) -> dict   # SGD; acumula exp.historico
exp.avaliar() -> dict          # mse_treino/validacao/teste, classificacao, acuracia, acertos, linhas
exp.prever_entrada(valores) -> (ŷ, classe ou None)        # aplica a normalização salva
exp.salvar(caminho)            # persistencia.salvar_rede
exp.reiniciar()                # pesos iniciais (mesma semente) e histórico zerado

Experimento.da_base(caminho, ocultas, ativacao_saida, usar_bias, seed, n_saidas=1,
                    percentuais=(70, 15, 15), normalizar=True, tamanho_lote=None)  # tudo de uma vez (hub)
Experimento.carregar(caminho)  # rede salva, sem base: modo previsão
```

Atributos lidos pela interface: `rede`, `classes`, `classificacao`,
`separou`, `X_treino`/`Y_treino`, `X_validacao`/`Y_validacao`,
`X_teste`/`Y_teste`, `X_*_original`, `normalizacao`, `historico`,
`nome_base`, `seed`. O `main.py` usa `Experimento` com as mesmas perguntas e
mensagens; `test_main.py` passou sem mudança.

### 10.2 Gravador

```python
def gravar(experimento, epocas, taxa, epocas_detalhadas, nivel, exemplo_em_foco) -> Gravacao
def passos_previsao(rede, x_normalizado, nivel) -> list[Passo]
```

- Repete o laço de `rede/treinamento.treinar` (mesmo gerador
  `np.random.default_rng(seed)`, mesma `permutation` por época), mas chama
  os passos do exemplo separados para copiar o estado no meio:
  1. copia `pesos`/`bias` de todos os neurônios (estado antes do exemplo);
  2. `rede.forward(x)` → copia `entradas`, `z`, `a` de cada neurônio;
  3. `rede.retropropagar(y)` → copia `delta` de cada neurônio;
  4. `rede.atualizar(taxa)` → pesos novos.
  Ao fim da época, MSE de treino e validação com `rede.perda.mse`.
- Resultado tem que ser **idêntico** ao de `treinamento.treinar` com a mesma
  semente (testado).
- Nada é recalculado na interface: todo número de passo vem dessas cópias.
- **Feito na etapa 2.** Detalhes que ficaram decididos:
  - o gravador gera os passos direto no nível pedido (`nivel`); não há
    função de agrupar depois;
  - cada `Passo` tem o campo `pesos` ([(W, b)] valendo naquele passo: os de
    antes do exemplo, e os novos camada a camada nos passos de ajuste), e
    `estado_em(passos, i)` junta ativações e deltas do exemplo até o passo;
  - camada 0 = entrada; índices de neurônio começam em 0 (as fórmulas
    mostram a partir de 1: w₁, z₂...);
  - o ∂E/∂a das ocultas mostrado na fórmula é `Σ δ·w` com os pesos de antes,
    a mesma conta de `Camada.backward`; o teste confere que `δ = ∂E/∂a · f'(z)`
    bate com o δ real do neurônio;
  - medido: Iris (200 épocas) e `dataset.csv` (1000 épocas), 5 épocas
    detalhadas no nível ligação ≈ 1.060 passos; gravar leva ~5 s e ~20 s.

### 10.3 Desempenho

- Desenho com 60 FPS alvo; a rede é desenhada numa `Surface` de fundo
  que só é refeita quando os pesos ou o tamanho mudam; o destaque do
  passo vai por cima.
- Limite de 12 neurônios por camada segura redes grandes.
- Treino SGD com objetos `Neuronio` é mais lento que em matriz: Iris com
  200 épocas leva alguns segundos; `dataset.csv` com 1000 épocas, ~10–20 s.
  O gravador roda em pedaços entre quadros com barra de progresso.
- Gravação: só o exemplo em foco tem passos de ligação; por época
  detalhada ≈ (ligações × 3 + neurônios × 5) passos + 1 por exemplo
  compacto. Se passar de ~200 mil passos, o hub avisa e sugere nível
  "neurônio".

## 11. Ordem de implementação

| # | Etapa | Pronto quando |
|---|---|---|
| 1 | `experimento.py` + `main.py` usando ele + `test_experimento.py` | testes antigos passam sem mudança, novos passam |
| 2 | `passos.py`, `formulas.py`, `gravador.py` + `test_gravador.py` | valores gravados batem com `z`/`a`/`δ` reais; treino gravado = `treinamento.treinar` com a mesma semente |
| 3 | `layout_rede.py` + `test_layout_rede.py` | posições, cores (inclusive linear), espessuras, "⋮" testados |
| 4 | `app.py` + hub (pygame_gui) + rede estática | escolher Iris, montar 4 → 5 → 3, ver a rede com pesos iniciais; divisão/normalização/ativação da saída validadas no hub |
| 5 | Player: barra de tempo, ◀ ▶, ▶/⏸, velocidade, atalhos | navegar para frente e para trás em XOR sem erro |
| 6 | Forward animado + painel de fórmulas | ver cada `w·a`, soma, bias, ativação com números certos |
| 7 | Erro, retropropagação (visão gradiente), ajuste e exemplos compactos | ver `δ` nascendo na saída e voltando; ajuste muda espessuras a cada exemplo |
| 8 | Saltos entre épocas + curvas de treino e validação | tocar da época 10 à 100 mostra pesos e curvas mudando |
| 9 | Salvar, carregar (modo previsão) e prever entrada | rede salva e carregada dá o mesmo ŷ; previsão anima só o forward |
| 10 | Pontos de parada + `test_pontos_parada.py` | "parar na retropropagação da época 100" pausa no passo certo |
| 11 | Comparar épocas | época 1 × última lado a lado |
| 12 | Exportar GIF/MP4 | exportar o forward do exemplo em foco da época 1 do XOR em GIF e MP4 |
| 13 | Tooltip, legenda, tema, `executar_interface.bat`, README | outra pessoa do grupo usa só lendo o README |

Após cada etapa, mostrar ao usuário o que mudou e explicar as decisões.

## 12. Testes

- `test_experimento.py`: XOR 100%, Iris ≥ 90%, `dataset.csv` melhor que
  prever a média; entrada vinda da base; divisão com os percentuais pedidos;
  normalização desligada não muda X; `prever_entrada` normaliza igual ao
  treino; `salvar` + `carregar` dão o mesmo ŷ.
- `test_gravador.py`: `z`, `a` e `δ` dos passos = valores reais dos
  neurônios no exemplo em foco; soma dos produtos `w·a` dos passos de
  ligação + bias = `z`; gradiente do passo = `δ · a`; ajuste do passo:
  `w_novo = w − η · δ · a`; pesos do resumo da época N = pesos de
  `treinamento.treinar` com a mesma semente; MSE de treino/validação do
  resumo = histórico de `treinamento.treinar`; uma época detalhada tem
  N − 1 passos compactos; níveis menores têm menos passos e mesma ordem de
  fases; `passos_previsao` não muda nenhum peso.
- `test_layout_rede.py`: posições dentro da área; espaçamento igual;
  `neuronios_visiveis(30)` = 11 índices + `None` + último; sigmoid 0 =
  branco, 1 = azul; linear negativo = laranja; peso negativo = laranja;
  maior peso = espessura máxima.
- `test_pontos_parada.py`: cada tipo de regra casa no passo certo e só nele.
- `test_interface.py`: com `SDL_VIDEODRIVER=dummy`, abre o app, gera XOR,
  avança e volta 50 passos chamando o laço direto, salva e carrega a rede
  numa pasta temporária, exporta 10 quadros em GIF, fecha. Pulado se pygame
  não estiver instalado.
- Teste visual (cores, animação) é manual: roteiro no fim de cada etapa.

## 13. Fora deste plano

- Desenhar redes com centenas de neurônios por camada.
- Editar pesos clicando nas ligações.
- Treino em lote (batch) ou mini-lote: o enunciado pede SGD como única
  estratégia.
- Normalização Z-score (só min-max, como no núcleo).

## 14. Uso no vídeo

1. Abrir o hub, escolher XOR, montar 2 → [4 sigmoid] → 1, gerar.
2. Época 1, exemplo em foco: forward passo a passo, mostrando cada `w·a` e a
   fórmula; saída perto de 0,5, errada.
3. Erro e retropropagação: o δ voltando pela rede (visão gradiente), com os
   números do exemplo da Aula 13.
4. Ajuste dos pesos logo depois do exemplo; os outros 3 exemplos passando
   como passos compactos (SGD: pesos mudam a cada exemplo).
5. Tocar os saltos até a última época: curva do erro caindo.
6. Forward na última época: agora acerta.
7. Comparar épocas 1 × última.
8. Montar sem camada oculta e mostrar que o XOR não é aprendido.
9. Iris (4 → 5 → 3), divisão 70/15/15, curvas de treino × validação, ponto
   de parada "erro < 0,05", exemplo setosa.
10. Base do professor (`dataset.csv`): 2 → [8 relu] → 1 linear, entradas
    normalizadas, saída em kg/ha sem normalizar; prever uma entrada.
11. Salvar a rede, voltar ao hub, carregar o arquivo e prever a mesma
    entrada: mesmo ŷ.
12. Trechos exportados em MP4 para editar o vídeo final.
