"""Tela que mostra a rede montada com os pesos iniciais (etapa 4).

Na etapa 5 ela vira o player: barra de tempo, controles e passos do gravador.
"""

import pygame
import pygame_gui

from interface.desenho_rede import desenhar_rede, texto

COR_FUNDO = (248, 248, 246)


class TelaRede:
    def __init__(self, app, experimento, parametros):
        self.app = app
        self.experimento = experimento
        self.parametros = parametros
        self.layout = None
        self.botoes = []
        self._criar_botoes()

    # ------------------------------------------------------------ widgets

    def _criar_botoes(self):
        for botao in self.botoes:
            botao.kill()
        largura, altura = self.app.tamanho
        self.botao_hub = pygame_gui.elements.UIButton(
            pygame.Rect(20, altura - 50, 120, 34), "← Hub", self.app.manager)
        self.botoes = [self.botao_hub]

    def fechar(self):
        for botao in self.botoes:
            botao.kill()

    def redimensionar(self):
        self._criar_botoes()

    # ------------------------------------------------------------- eventos

    def tratar_evento(self, evento):
        if evento.type == pygame_gui.UI_BUTTON_PRESSED and evento.ui_element is self.botao_hub:
            self.voltar()

    def voltar(self):
        from interface.tela_hub import TelaHub
        self.app.trocar_tela(TelaHub(self.app, self.app.ultimo_formulario))

    def atualizar(self, dt):
        pass

    # -------------------------------------------------------------- desenho

    def rotulos(self):
        """Nomes das entradas (cabeçalho da base ou x1..xn) e das saídas."""
        rede = self.experimento.rede
        descricao = self.app.ultima_descricao
        entrada = [f"x{i + 1}" for i in range(rede.n_entradas)]
        if descricao and descricao["cabecalho"]:
            entrada = descricao["cabecalho"][:rede.n_entradas]
        saida = self.experimento.classes or [f"ŷ{i + 1}" for i in range(rede.n_saidas)]
        return entrada, saida

    def desenhar(self, superficie):
        largura, altura = self.app.tamanho
        superficie.fill(COR_FUNDO)
        rede = self.experimento.rede
        texto(superficie, f"{self.experimento.nome_base} — rede com os pesos iniciais",
              (20, 18), 20, ancora="topleft", negrito=True)
        texto(superficie, self.app.ultima_previa, (20, 48), 15, ancora="topleft")
        pesos = [(camada.W, camada.b) for camada in rede.camadas]
        entrada, saida = self.rotulos()
        area = (140, 90, largura - 300, altura - 190)
        self.layout = desenhar_rede(superficie, area, rede, pesos,
                                    rotulos_entrada=entrada, rotulos_saida=saida)
        texto(superficie, "Ligação azul = peso positivo · laranja = negativo · espessura = |w|",
              (largura // 2, altura - 30), 13)
