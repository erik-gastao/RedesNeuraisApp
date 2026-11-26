"""Desenha a rede numa superfície do pygame (seção 7 do PLANO_INTERFACE.md).

Só desenha: posições, cores e espessuras vêm de layout_rede.py, e os
números vêm do gravador/da rede. Visão "ativação" (forward) ou "gradiente"
(retropropagação).
"""

import pygame

from interface.layout_rede import (
    BRANCO,
    calcular_layout,
    cor_ativacao,
    cor_entrada,
    cor_gradiente,
    cor_peso,
    espessura,
    interpolar,
    maior_peso,
)

COR_TEXTO = (40, 40, 40)
COR_CONTORNO = (90, 90, 90)


def fonte(tamanho, negrito=False):
    """Fonte do sistema com acentos e símbolos (ŷ, η, δ)."""
    return pygame.font.SysFont("segoeui,arial,dejavusans", tamanho, bold=negrito)


def texto(superficie, conteudo, posicao, tamanho=15, cor=COR_TEXTO, ancora="center", negrito=False):
    imagem = fonte(tamanho, negrito).render(conteudo, True, cor)
    retangulo = imagem.get_rect(**{ancora: posicao})
    superficie.blit(imagem, retangulo)
    return retangulo


def _maior_absoluto(valores):
    return max((abs(v) for v in valores), default=0.0)


def desenhar_rede(superficie, area, rede, pesos, ativacoes=None, deltas=None, visao="ativacao",
                  rotulos_entrada=None, rotulos_saida=None):
    """Desenha a rede na área (x, y, largura, altura) e devolve o Layout usado.

    pesos:     [(W, b), ...] valendo neste momento (do passo ou da rede)
    ativacoes: {(camada, neurônio): a} já calculadas (camada 0 = entradas)
    deltas:    {(camada, neurônio): δ} já calculados (visão "gradiente")
    """
    ativacoes = ativacoes or {}
    deltas = deltas or {}
    tamanhos = [rede.n_entradas] + [camada.n_neuronios for camada in rede.camadas]
    nomes = [None] + [camada.nome_ativacao for camada in rede.camadas]
    layout = calcular_layout(tamanhos, area)
    maior = maior_peso(pesos)

    # Ligações primeiro, para os neurônios ficarem por cima.
    for camada in range(1, len(tamanhos)):
        W = pesos[camada - 1][0]
        maior_gradiente = _maior_absoluto(
            deltas.get((camada, n), 0.0) * ativacoes.get((camada - 1, j), 0.0)
            for n in range(tamanhos[camada]) for j in range(tamanhos[camada - 1])
        )
        for n in layout.visiveis[camada]:
            if n is None:
                continue
            for j in layout.visiveis[camada - 1]:
                if j is None:
                    continue
                w = W[n][j]
                largura = espessura(w, maior)
                cor = cor_peso(w)
                if visao == "gradiente" and (camada, n) in deltas:
                    gradiente = deltas[(camada, n)] * ativacoes.get((camada - 1, j), 0.0)
                    cor = cor_gradiente(gradiente, maior_gradiente)
                if largura < 1:
                    cor = interpolar(BRANCO, cor, largura)   # linha fina = cor mais clara
                pygame.draw.line(superficie, cor, layout.posicoes[(camada - 1, j)],
                                 layout.posicoes[(camada, n)], max(1, round(largura)))

    # Neurônios.
    maior_delta = _maior_absoluto(deltas.values())
    for camada, visiveis in enumerate(layout.visiveis):
        valores_camada = [v for (c, _), v in ativacoes.items() if c == camada]
        maior_camada = _maior_absoluto(valores_camada)
        for n in visiveis:
            if n is None:
                x, y = layout.reticencias[camada]
                for deslocamento in (-layout.raio * 0.5, 0, layout.raio * 0.5):
                    pygame.draw.circle(superficie, COR_CONTORNO, (x, y + deslocamento), 2)
                continue
            posicao = layout.posicoes[(camada, n)]
            a = ativacoes.get((camada, n))
            if visao == "gradiente" and camada > 0 and (camada, n) in deltas:
                preenchimento = cor_gradiente(deltas[(camada, n)], maior_delta)
            elif camada == 0:
                preenchimento = cor_entrada(a, maior_camada)
            else:
                preenchimento = cor_ativacao(a, nomes[camada], maior_camada)
            pygame.draw.circle(superficie, preenchimento, posicao, layout.raio)
            pygame.draw.circle(superficie, COR_CONTORNO, posicao, layout.raio, 2)

    _desenhar_rotulos(superficie, layout, tamanhos, nomes, rotulos_entrada, rotulos_saida)
    return layout


def _desenhar_rotulos(superficie, layout, tamanhos, nomes, rotulos_entrada, rotulos_saida):
    ultima = len(tamanhos) - 1
    base_y = max(y for _, y in layout.posicoes.values()) + layout.raio + 16
    for camada in range(len(tamanhos)):
        x = layout.posicoes[(camada, 0)][0]
        if camada == 0:
            nome = "Entrada"
        elif camada == ultima:
            nome = f"Saída ({nomes[camada]})"
        else:
            nome = f"Oculta {camada} ({nomes[camada]})"
        texto(superficie, nome, (x, base_y), 14)
        if tamanhos[camada] > len([n for n in layout.visiveis[camada] if n is not None]):
            texto(superficie, f"{tamanhos[camada]} neurônios", (x, base_y + 18), 12)

    for indice, rotulo in enumerate(rotulos_entrada or []):
        if (0, indice) in layout.posicoes:
            x, y = layout.posicoes[(0, indice)]
            texto(superficie, rotulo, (x - layout.raio - 6, y), 13, ancora="midright")
    for indice, rotulo in enumerate(rotulos_saida or []):
        if (ultima, indice) in layout.posicoes:
            x, y = layout.posicoes[(ultima, indice)]
            texto(superficie, rotulo, (x + layout.raio + 6, y), 13, ancora="midleft")
