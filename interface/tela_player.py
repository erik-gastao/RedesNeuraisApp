"""Player: toca os passos gravados como um vídeo (seções 6.2, 7 e 8 do plano).

Mantém só um índice na lista de passos e desenha o passo atual com
estado_em(); voltar, pular e tocar não recalculam nada.

Teclas: espaço tocar/pausar · ←/→ passo · Shift+←/→ neurônio (ou camada)
· Home/End ou PgUp/PgDn fase · com Ctrl, exemplo · com Ctrl+Shift, época
· −/+ velocidade · V visão (automática / ativação / gradiente).
"""

import pygame
import pygame_gui

from interface.desenho_rede import COR_TEXTO, desenhar_rede, fonte, texto
from interface.passos import estado_em

COR_FUNDO = (248, 248, 246)
COR_PAINEL = (255, 255, 255)
COR_BORDA = (215, 215, 215)
COR_DESTAQUE = (255, 196, 0)
LARGURA_PAINEL = 340
VELOCIDADES = [0.25, 0.5, 1, 2, 4, 8, 16]
PASSOS_POR_SEGUNDO = 2          # na velocidade 1x
CORES_FASE = {
    "normalizacao": (150, 110, 200),
    "forward": (33, 102, 172),
    "erro": (215, 48, 39),
    "retropropagacao": (230, 97, 1),
    "atualizacao": (26, 152, 80),
    "exemplo": (180, 180, 180),
}
NOMES_FASE = {
    "normalizacao": "NORMALIZAÇÃO", "forward": "FORWARD", "erro": "ERRO",
    "retropropagacao": "RETROPROPAGAÇÃO", "atualizacao": "ATUALIZAÇÃO",
    "exemplo": "EXEMPLO (compacto)",
}
VISOES = ["automática", "ativação", "gradiente"]


# ---------------------------------------------------------- navegação

def _chave(passo, criterio):
    """O que define o grupo do passo para cada tipo de salto."""
    base = (passo.epoca, passo.exemplo)
    if criterio == "epoca":
        return (passo.epoca,)
    if criterio == "exemplo":
        return base
    if criterio == "fase":
        return base + (passo.fase,)
    if criterio == "camada":
        return base + (passo.fase, passo.camada)
    return base + (passo.fase, passo.camada, passo.neuronio)   # "neuronio"


def salto(passos, i, criterio, direcao):
    """Índice do início do próximo grupo (direcao > 0) ou do grupo atual/anterior."""
    if not passos:
        return 0
    atual = _chave(passos[i], criterio)
    if direcao > 0:
        j = i + 1
        while j < len(passos) and _chave(passos[j], criterio) == atual:
            j += 1
        return min(j, len(passos) - 1)
    j = i
    while j > 0 and _chave(passos[j - 1], criterio) == atual:
        j -= 1
    if j == i and j > 0:            # já no início: vai para o início do grupo anterior
        j -= 1
        anterior = _chave(passos[j], criterio)
        while j > 0 and _chave(passos[j - 1], criterio) == anterior:
            j -= 1
    return j


def quebrar(conteudo, tamanho, largura):
    """Divide o texto em linhas que cabem na largura."""
    letra = fonte(tamanho)
    linhas, atual = [], ""
    for palavra in conteudo.split(" "):
        tentativa = f"{atual} {palavra}".strip()
        if letra.size(tentativa)[0] <= largura or not atual:
            atual = tentativa
        else:
            linhas.append(atual)
            atual = palavra
    if atual:
        linhas.append(atual)
    return linhas


# ------------------------------------------------------------------ tela

class TelaPlayer:
    def __init__(self, app, experimento, gravacao, parametros):
        self.app = app
        self.experimento = experimento
        self.gravacao = gravacao
        self.parametros = parametros
        self.passos = gravacao.passos
        self.indice = 0
        self.tocando = False
        self.velocidade = 2          # índice em VELOCIDADES (1x)
        self.acumulado = 0.0
        self.visao = 0               # índice em VISOES
        self.arrastando = False
        self.layout = None
        self.elementos = []
        self.acoes = {}
        self._criar_botoes()

    # --------------------------------------------------------- widgets

    def _botao(self, x, y, largura, conteudo, acao):
        botao = pygame_gui.elements.UIButton(pygame.Rect(x, y, largura, 32), conteudo,
                                             self.app.manager)
        self.elementos.append(botao)
        self.acoes[botao] = acao
        return botao

    def _criar_botoes(self):
        self.fechar()
        self.acoes = {}
        largura, altura = self.app.tamanho
        y = altura - 46
        x = 20
        for conteudo, acao in [
            ("|<<", lambda: self.saltar("fase", -1)),
            ("<<", lambda: self.saltar(self._criterio_neuronio(), -1)),
            ("<", lambda: self.ir_para(self.indice - 1)),
        ]:
            self._botao(x, y, 52, conteudo, acao)
            x += 56
        self.botao_tocar = self._botao(x, y, 80, "", self.alternar_tocar)
        x += 84
        for conteudo, acao in [
            (">", lambda: self.ir_para(self.indice + 1)),
            (">>", lambda: self.saltar(self._criterio_neuronio(), 1)),
            (">>|", lambda: self.saltar("fase", 1)),
        ]:
            self._botao(x, y, 52, conteudo, acao)
            x += 56
        x += 12
        self._botao(x, y, 32, "−", lambda: self.mudar_velocidade(-1))
        x += 36
        self.botao_velocidade = self._botao(x, y, 64, "", lambda: None)
        x += 68
        self._botao(x, y, 32, "+", lambda: self.mudar_velocidade(1))
        x += 48
        self.botao_visao = self._botao(x, y, 190, "", self.alternar_visao)
        self._botao(largura - 140, y, 120, "← Hub", self.voltar)
        self._atualizar_textos()

    def _atualizar_textos(self):
        self.botao_tocar.set_text("Pausar" if self.tocando else "Tocar")
        self.botao_velocidade.set_text(f"{VELOCIDADES[self.velocidade]:g}x")
        self.botao_visao.set_text(f"Visão: {VISOES[self.visao]}")

    def fechar(self):
        for elemento in self.elementos:
            elemento.kill()
        self.elementos = []

    def redimensionar(self):
        self._criar_botoes()

    # ----------------------------------------------------------- ações

    def _criterio_neuronio(self):
        return "camada" if self.gravacao.nivel == "camada" else "neuronio"

    def ir_para(self, indice):
        if self.passos:
            self.indice = max(0, min(indice, len(self.passos) - 1))

    def saltar(self, criterio, direcao):
        self.ir_para(salto(self.passos, self.indice, criterio, direcao))

    def alternar_tocar(self):
        if self.passos and self.indice >= len(self.passos) - 1:
            self.indice = 0
        self.tocando = not self.tocando
        self.acumulado = 0.0
        self._atualizar_textos()

    def mudar_velocidade(self, direcao):
        self.velocidade = max(0, min(self.velocidade + direcao, len(VELOCIDADES) - 1))
        self._atualizar_textos()

    def alternar_visao(self):
        self.visao = (self.visao + 1) % len(VISOES)
        self._atualizar_textos()

    def voltar(self):
        from interface.tela_hub import TelaHub
        self.app.trocar_tela(TelaHub(self.app, self.app.ultimo_formulario))

    # ---------------------------------------------------------- eventos

    def _retangulo_barra(self):
        largura, altura = self.app.tamanho
        return pygame.Rect(20, altura - 82, largura - 40, 20)

    def _indice_na_barra(self, x):
        barra = self._retangulo_barra()
        fracao = (x - barra.x) / max(barra.width, 1)
        return int(max(0.0, min(fracao, 0.9999)) * len(self.passos))

    def tratar_evento(self, evento):
        if evento.type == pygame_gui.UI_BUTTON_PRESSED and evento.ui_element in self.acoes:
            self.acoes[evento.ui_element]()
        elif evento.type == pygame.KEYDOWN:
            self._tecla(evento)
        elif evento.type == pygame.MOUSEBUTTONDOWN and evento.button == 1:
            if self.passos and self._retangulo_barra().inflate(0, 10).collidepoint(evento.pos):
                self.arrastando = True
                self.ir_para(self._indice_na_barra(evento.pos[0]))
        elif evento.type == pygame.MOUSEMOTION and self.arrastando:
            self.ir_para(self._indice_na_barra(evento.pos[0]))
        elif evento.type == pygame.MOUSEBUTTONUP and evento.button == 1:
            self.arrastando = False

    def _tecla(self, evento):
        if self.app.manager.get_focus_set():
            return
        ctrl = evento.mod & pygame.KMOD_CTRL
        shift = evento.mod & pygame.KMOD_SHIFT
        criterio = "epoca" if ctrl and shift else "exemplo" if ctrl else "fase"
        if evento.key == pygame.K_SPACE:
            self.alternar_tocar()
        elif evento.key in (pygame.K_LEFT, pygame.K_RIGHT):
            direcao = 1 if evento.key == pygame.K_RIGHT else -1
            if shift:
                self.saltar(self._criterio_neuronio(), direcao)
            else:
                self.ir_para(self.indice + direcao)
        elif evento.key in (pygame.K_HOME, pygame.K_PAGEUP):
            self.saltar(criterio, -1)
        elif evento.key in (pygame.K_END, pygame.K_PAGEDOWN):
            self.saltar(criterio, 1)
        elif evento.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.mudar_velocidade(-1)
        elif evento.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
            self.mudar_velocidade(1)
        elif evento.key == pygame.K_v:
            self.alternar_visao()

    def atualizar(self, dt):
        if not self.tocando or not self.passos:
            return
        self.acumulado += dt * PASSOS_POR_SEGUNDO * VELOCIDADES[self.velocidade]
        avancar = int(self.acumulado)
        if avancar:
            self.acumulado -= avancar
            self.ir_para(self.indice + avancar)
            if self.indice >= len(self.passos) - 1:
                self.tocando = False
                self._atualizar_textos()

    # ---------------------------------------------------------- desenho

    def _visao_do_passo(self, passo):
        if VISOES[self.visao] != "automática":
            return "gradiente" if VISOES[self.visao] == "gradiente" else "ativacao"
        return "gradiente" if passo and passo.fase in ("retropropagacao", "atualizacao") else "ativacao"

    def _rotulos(self, ativacoes):
        rede = self.experimento.rede
        descricao = self.app.ultima_descricao
        entrada = [f"x{i + 1}" for i in range(rede.n_entradas)]
        if descricao and descricao["cabecalho"]:
            entrada = descricao["cabecalho"][:rede.n_entradas]
        ultima = len(rede.camadas)
        nomes = self.experimento.classes or [f"ŷ{i + 1}" for i in range(rede.n_saidas)]
        saida = []
        for n, nome in enumerate(nomes):
            a = ativacoes.get((ultima, n))
            saida.append(f"{nome}  {a:.2f}" if a is not None else nome)
        return entrada, saida

    def desenhar(self, superficie):
        largura, altura = self.app.tamanho
        superficie.fill(COR_FUNDO)
        rede = self.experimento.rede
        passo = self.passos[self.indice] if self.passos else None
        if passo:
            estado = estado_em(self.passos, self.indice)
        else:
            ultimo = self.gravacao.resumos[-1]
            estado = {"pesos": ultimo["pesos"], "ativacoes": {}, "deltas": {}, "fase": None}

        area = (120, 74, largura - LARGURA_PAINEL - 260, altura - 214)
        visao = self._visao_do_passo(passo)
        entrada, saida = self._rotulos(estado["ativacoes"])
        self.layout = desenhar_rede(superficie, area, rede, estado["pesos"], estado["ativacoes"],
                                    estado["deltas"], visao, entrada, saida)
        if passo:
            self._destacar(superficie, passo)
        texto(superficie, f"{self.experimento.nome_base} — {self.app.ultima_previa}",
              (20, 14), 15, ancora="topleft", negrito=True)
        self._desenhar_painel(superficie, passo)
        self._desenhar_barra(superficie)

    def _destacar(self, superficie, passo):
        """Realça o que o passo atual está calculando."""
        layout = self.layout
        posicoes = layout.posicoes
        raio = layout.raio

        def anel(chave, largura=4):
            if chave in posicoes:
                pygame.draw.circle(superficie, COR_DESTAQUE, posicoes[chave], raio + 5, largura)

        c, n = passo.camada, passo.neuronio
        if passo.tipo in ("ligacao", "gradiente"):
            if (c - 1, passo.origem) in posicoes and (c, n) in posicoes:
                pygame.draw.line(superficie, COR_DESTAQUE, posicoes[(c - 1, passo.origem)],
                                 posicoes[(c, n)], 5)
            anel((c, n), 3)
        elif passo.tipo in ("soma", "bias", "ativacao", "delta"):
            anel((c, n))
            v = passo.valores
            if passo.tipo == "delta":
                legenda = f"δ = {v['delta']:.4f}"
            elif "a" in v:
                legenda = f"z = {v['z']:.2f}   a = {v['a']:.2f}"
            elif "z" in v:
                legenda = f"z = {v['z']:.2f}"
            else:
                legenda = f"Σ = {v['soma']:.2f}"
            if (c, n) in posicoes:
                x, y = posicoes[(c, n)]
                texto(superficie, legenda, (x, y - raio - 16), 14, negrito=True)
        elif passo.tipo in ("camada_forward", "camada_delta", "ajuste", "erro_saida"):
            for (camada, neuronio) in posicoes:
                if camada == c:
                    anel((camada, neuronio))
        elif passo.tipo in ("entrada_rede", "entrada"):
            for (camada, neuronio) in posicoes:
                if camada == 0 and (passo.tipo == "entrada_rede" or neuronio == n):
                    anel((camada, neuronio))
        elif passo.tipo == "exemplo_compacto":
            ultima = len(self.experimento.rede.camadas)
            for (camada, neuronio) in posicoes:
                if camada == ultima:
                    anel((camada, neuronio), 2)

    def _desenhar_painel(self, superficie, passo):
        largura, altura = self.app.tamanho
        x = largura - LARGURA_PAINEL - 20
        painel = pygame.Rect(x, 44, LARGURA_PAINEL, altura - 140)
        pygame.draw.rect(superficie, COR_PAINEL, painel, border_radius=6)
        pygame.draw.rect(superficie, COR_BORDA, painel, 1, border_radius=6)
        y = painel.y + 12
        margem = x + 14
        largura_texto = LARGURA_PAINEL - 28

        if passo is None:
            texto(superficie, "Nenhuma época detalhada: só o resumo.", (margem, y), 15,
                  ancora="topleft")
        else:
            titulo = f"ÉPOCA {passo.epoca}" if passo.epoca else "ANTES DO TREINO"
            if passo.exemplo is not None:
                titulo += f" · EXEMPLO {passo.exemplo + 1}"
            texto(superficie, titulo, (margem, y), 16, ancora="topleft", negrito=True)
            y += 24
            texto(superficie, NOMES_FASE[passo.fase], (margem, y), 14,
                  cor=CORES_FASE[passo.fase], ancora="topleft", negrito=True)
            y += 22
            if passo.camada:
                onde = f"Camada {passo.camada}"
                if passo.neuronio is not None:
                    onde += f" · neurônio {passo.neuronio + 1}"
                texto(superficie, onde, (margem, y), 14, ancora="topleft")
                y += 22
            y += 6
            for linha in quebrar(passo.formula, 16, largura_texto):
                texto(superficie, linha, (margem, y), 16, ancora="topleft", negrito=True)
                y += 22
            for linha in quebrar(passo.formula_numeros, 15, largura_texto):
                texto(superficie, linha, (margem, y), 15, ancora="topleft")
                y += 20
            y += 8
            for linha in quebrar(passo.texto, 14, largura_texto):
                texto(superficie, linha, (margem, y), 14, cor=(90, 90, 90), ancora="topleft")
                y += 19

        grafico = pygame.Rect(x + 12, painel.bottom - 190, LARGURA_PAINEL - 24, 170)
        self._desenhar_curva(superficie, grafico, passo)

    def _desenhar_curva(self, superficie, area, passo):
        """Curvas do erro de treino (azul) e validação (laranja) por época."""
        resumos = self.gravacao.resumos
        treino = [r["erro_treino"] for r in resumos]
        validacao = [r["erro_validacao"] for r in resumos if r["erro_validacao"] is not None]
        pygame.draw.rect(superficie, (250, 250, 250), area)
        pygame.draw.rect(superficie, COR_BORDA, area, 1)
        texto(superficie, "Erro (MSE) por época", (area.x + 6, area.y + 4), 13,
              ancora="topleft", negrito=True)
        if len(treino) < 1:
            return
        maior = max(treino + validacao) or 1.0
        topo, base = area.y + 24, area.bottom - 18

        def ponto(i, valor, total):
            x = area.x + 8 + (area.width - 16) * (i / max(total - 1, 1))
            y = base - (base - topo) * (valor / maior)
            return x, y

        for serie, cor in ((treino, (33, 102, 172)), (validacao, (230, 97, 1))):
            if len(serie) > 1:
                pygame.draw.lines(superficie, cor, False,
                                  [ponto(i, v, len(serie)) for i, v in enumerate(serie)], 2)
        texto(superficie, f"treino {treino[-1]:.4g}", (area.x + 8, base + 2), 12,
              cor=(33, 102, 172), ancora="topleft")
        if validacao:
            texto(superficie, f"validação {validacao[-1]:.4g}", (area.centerx, base + 2), 12,
                  cor=(230, 97, 1), ancora="topleft")
        if passo and passo.epoca:
            x, _ = ponto(passo.epoca - 1, 0, len(treino))
            pygame.draw.line(superficie, COR_DESTAQUE, (x, topo), (x, base), 2)

    def _desenhar_barra(self, superficie):
        """Barra de tempo: uma faixa colorida por fase; clique/arraste pula."""
        barra = self._retangulo_barra()
        pygame.draw.rect(superficie, (230, 230, 230), barra)
        total = len(self.passos)
        if total == 0:
            return
        for coluna in range(barra.width):
            passo = self.passos[min(int(coluna / barra.width * total), total - 1)]
            pygame.draw.line(superficie, CORES_FASE[passo.fase],
                             (barra.x + coluna, barra.y + 3), (barra.x + coluna, barra.bottom - 3))
        # Início de cada época detalhada.
        anterior = None
        for i, passo in enumerate(self.passos):
            if passo.epoca != anterior and passo.epoca:
                x = barra.x + barra.width * i / total
                pygame.draw.line(superficie, COR_TEXTO, (x, barra.y - 4), (x, barra.bottom), 1)
                texto(superficie, f"época {passo.epoca}", (x + 3, barra.y - 16), 11, ancora="topleft")
            anterior = passo.epoca
        x = barra.x + barra.width * (self.indice + 0.5) / total
        pygame.draw.line(superficie, (0, 0, 0), (x, barra.y - 6), (x, barra.bottom + 6), 3)
        texto(superficie, f"passo {self.indice + 1} / {total}",
              (barra.right, barra.y - 16), 12, ancora="topright")
