"""Onde e com que cor cada parte da rede é desenhada (sem pygame).

Seção 7 do PLANO_INTERFACE.md:
- camadas em colunas igualmente espaçadas, da entrada (esquerda) à saída;
- neurônios centralizados na vertical, com o mesmo espaçamento em todas as
  camadas; no máximo 12 por camada (11, "⋮" e o último);
- cores azul/laranja (legíveis para daltônicos): azul = positivo,
  laranja = negativo, branco = zero, cinza claro = ainda não calculado;
- espessura da ligação proporcional ao maior |w| da rede (0,5 a 6 px).

As cores são tuplas RGB; o desenho com pygame só as usa.
"""

from dataclasses import dataclass

import numpy as np

BRANCO = (255, 255, 255)
AZUL = (67, 147, 224)
LARANJA = (230, 97, 1)
CINZA_CLARO = (120, 120, 120)
VERDE = (26, 152, 80)
VERMELHO = (215, 48, 39)

MAXIMO_VISIVEIS = 12
RAIO_MINIMO = 10
RAIO_MAXIMO = 24
ESPESSURA_MINIMA = 0.5
ESPESSURA_MAXIMA = 6.0


@dataclass
class Layout:
    """posicoes:    {(camada, neurônio): (x, y)}, só dos neurônios visíveis
    reticencias:   {camada: (x, y)} das camadas com "⋮"
    raio:          raio dos neurônios, em pixels
    visiveis:      por camada, a lista de neuronios_visiveis()
    """

    posicoes: dict
    reticencias: dict
    raio: float
    visiveis: list


def neuronios_visiveis(quantidade):
    """Índices desenhados numa camada; None marca o lugar do "⋮".

    Até 12 neurônios, todos; acima disso, os 11 primeiros, "⋮" e o último.
    """
    if quantidade <= MAXIMO_VISIVEIS:
        return list(range(quantidade))
    return list(range(MAXIMO_VISIVEIS - 1)) + [None, quantidade - 1]


def calcular_layout(tamanhos, area):
    """Posições dos neurônios dentro da área (x, y, largura, altura).

    tamanhos: neurônios por camada, da entrada à saída (ex.: [4, 5, 3]).
    Chamado de novo sempre que a janela muda de tamanho.
    """
    x0, y0, largura, altura = area
    visiveis = [neuronios_visiveis(t) for t in tamanhos]
    maximo_lugares = max(len(v) for v in visiveis)

    # Raio conforme a maior camada (e a largura de cada coluna).
    raio = min(altura / maximo_lugares, largura / len(tamanhos)) * 0.35
    raio = float(np.clip(raio, RAIO_MINIMO, RAIO_MAXIMO))

    # Mesmo espaçamento vertical em todas as camadas; a maior vai quase até as
    # bordas (1 px de folga, para o arredondamento não jogar o círculo para fora).
    espaco = (altura - 2 * raio - 2) / (maximo_lugares - 1) if maximo_lugares > 1 else 0.0
    margem = max(raio, largura * 0.08)
    passo_x = (largura - 2 * margem) / (len(tamanhos) - 1) if len(tamanhos) > 1 else 0.0
    centro_y = y0 + altura / 2

    posicoes, reticencias = {}, {}
    for camada, lugares in enumerate(visiveis):
        x = x0 + margem + camada * passo_x
        for lugar, neuronio in enumerate(lugares):
            y = centro_y + (lugar - (len(lugares) - 1) / 2) * espaco
            if neuronio is None:
                reticencias[camada] = (x, y)
            else:
                posicoes[(camada, neuronio)] = (x, y)
    return Layout(posicoes, reticencias, raio, visiveis)


# ----------------------------------------------------------------- cores

def interpolar(inicio, fim, t):
    """Cor entre inicio (t = 0) e fim (t = 1)."""
    t = float(np.clip(t, 0.0, 1.0))
    return tuple(int(round(a + (b - a) * t)) for a, b in zip(inicio, fim))


def divergente(valor, maior):
    """Branco no zero; azul forte em +maior; laranja forte em −maior."""
    if maior <= 0:
        return BRANCO
    destino = AZUL if valor >= 0 else LARANJA
    return interpolar(BRANCO, destino, abs(valor) / maior)


def cor_ativacao(a, nome_ativacao, maior=None):
    """Preenchimento do neurônio pela saída a do exemplo (visão "ativação").

    sigmoid: 0 = branco, 1 = azul. tanh: −1 laranja, 0 branco, +1 azul.
    relu: 0 = branco, maior valor da camada = azul.
    linear: escala divergente pelo maior |a| da camada.
    None (ainda não calculado) = cinza claro.
    """
    if a is None:
        return CINZA_CLARO
    if nome_ativacao == "sigmoid":
        return interpolar(BRANCO, AZUL, a)
    if nome_ativacao == "tanh":
        return divergente(a, 1.0)
    if nome_ativacao == "relu":
        return interpolar(BRANCO, AZUL, a / maior) if maior else BRANCO
    return divergente(a, maior if maior else abs(a))


def cor_entrada(x, maior):
    """Neurônio de entrada: 0 = branco, maior valor = azul (negativo = laranja)."""
    if x is None:
        return CINZA_CLARO
    return divergente(x, maior)


def cor_gradiente(valor, maior):
    """Visão "gradiente": δ no neurônio ou δ·a na ligação, com escala própria."""
    return divergente(valor, maior)


def cor_peso(w):
    """Ligação (e contorno do bias): azul se positivo, laranja se negativo."""
    return AZUL if w >= 0 else LARANJA


def contorno_saida(acertou):
    """Contorno do neurônio de saída na classificação."""
    return VERDE if acertou else VERMELHO


# ------------------------------------------------------------ espessuras

def maior_peso(pesos):
    """Maior |w| da rede (só os pesos das ligações), de [(W, b), ...]."""
    return max((float(np.abs(W).max()) for W, _ in pesos if W.size), default=0.0)


def espessura(w, maior):
    """Espessura da ligação: 0,5 px (peso 0) a 6 px (maior |w| da rede)."""
    if maior <= 0:
        return ESPESSURA_MINIMA
    return ESPESSURA_MINIMA + (ESPESSURA_MAXIMA - ESPESSURA_MINIMA) * min(abs(w) / maior, 1.0)
