"""Tela de espera enquanto o gravador treina a rede (barra de progresso).

O treino roda numa thread separada, para a janela continuar respondendo.
Ao terminar, abre o player.
"""

import threading

import pygame
import pygame_gui

from interface.desenho_rede import texto
from interface.gravador import gravar

COR_FUNDO = (12, 12, 12)
AZUL = (67, 147, 224)


class TelaGravando:
    def __init__(self, app, experimento, parametros):
        self.app = app
        self.experimento = experimento
        self.parametros = parametros
        self.epoca = 0
        self.gravacao = None
        self.erro = None
        self.botao_hub = None
        self._criar_botao()
        self.thread = threading.Thread(target=self._gravar, daemon=True)
        self.thread.start()

    def _gravar(self):
        p = self.parametros
        try:
            self.gravacao = gravar(
                self.experimento, p["epocas"], p["taxa"], p["epocas_detalhadas"],
                nivel=p["nivel"], exemplo_em_foco=p["exemplo_em_foco"],
                ao_fim_da_epoca=lambda epoca: setattr(self, "epoca", epoca),
            )
        except Exception as erro:   # mostrado na tela; a janela não fecha
            self.erro = str(erro)

    def _criar_botao(self):
        if self.botao_hub:
            self.botao_hub.kill()
        self.botao_hub = pygame_gui.elements.UIButton(
            pygame.Rect(20, self.app.tamanho[1] - 50, 120, 34), "← Hub", self.app.manager)
        self.botao_hub.hide()

    def fechar(self):
        self.botao_hub.kill()

    def redimensionar(self):
        self._criar_botao()

    def tratar_evento(self, evento):
        if evento.type == pygame_gui.UI_BUTTON_PRESSED and evento.ui_element is self.botao_hub:
            from interface.tela_hub import TelaHub
            self.app.trocar_tela(TelaHub(self.app, self.app.ultimo_formulario))

    def atualizar(self, dt):
        if self.erro:
            self.botao_hub.show()
        elif self.gravacao is not None:
            from interface.tela_player import TelaPlayer
            self.app.trocar_tela(TelaPlayer(self.app, self.experimento, self.gravacao,
                                            self.parametros))

    def desenhar(self, superficie):
        largura, altura = self.app.tamanho
        superficie.fill(COR_FUNDO)
        total = self.parametros["epocas"]
        if self.erro:
            texto(superficie, f"Não foi possível gravar o treino: {self.erro}",
                  (largura // 2, altura // 2), 16, cor=(255, 107, 94))
            return
        texto(superficie, f"Treinando e gravando os passos… época {self.epoca} de {total}",
              (largura // 2, altura // 2 - 40), 18)
        barra = pygame.Rect(largura // 2 - 300, altura // 2, 600, 22)
        pygame.draw.rect(superficie, (60, 60, 60), barra, border_radius=4)
        cheio = barra.copy()
        cheio.width = int(barra.width * self.epoca / max(total, 1))
        pygame.draw.rect(superficie, AZUL, cheio, border_radius=4)
        texto(superficie, self.app.ultima_previa, (largura // 2, altura // 2 + 50), 15)
