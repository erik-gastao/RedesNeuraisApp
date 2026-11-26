"""Laço principal da interface gráfica: janela, telas e eventos.

A janela nunca fecha com erro: uma exceção inesperada vira uma mensagem em
vermelho no rodapé (e o detalhe vai para o terminal).
"""

import ctypes
import sys
import traceback

import pygame
import pygame_gui

from interface.desenho_rede import texto

TAMANHO_INICIAL = (1280, 720)
TITULO = "Construtor de Redes Neurais"

def _fonte_do_sistema(manager):
    """Usa a Segoe UI (tem →, ●, η, ŷ) se existir; senão a Noto Sans do pygame_gui."""
    regular = pygame.font.match_font("segoeui")
    negrito = pygame.font.match_font("segoeui", bold=True)
    if not regular:
        return "noto_sans"
    manager.add_font_paths("segoeui", regular, negrito or regular)
    return "segoeui"


def tema(fonte):
    """Tema do pygame_gui: rótulos à esquerda, cores de seção, erro e aviso."""
    def letra(tamanho, negrito=False):
        return {"name": fonte, "size": str(tamanho), "bold": "1" if negrito else "0"}

    return {
        "defaults": {"font": letra(14)},
        "label": {"font": letra(14), "misc": {"text_horiz_alignment": "left"},
                  "colours": {"normal_text": "#EBEBEB", "dark_bg": "#00000000"}},
        "button": {"font": letra(14)},
        # Campos numéricos com contraste alto: número branco em fundo preto.
        "text_entry_line": {"font": letra(14), "colours": {
            "dark_bg": "#000000", "normal_text": "#FFFFFF", "text_cursor": "#FFFFFF",
            "selected_bg": "#2166AC", "selected_text": "#FFFFFF",
            "normal_border": "#9A9A9A", "disabled_dark_bg": "#3A3A3A",
            "disabled_text": "#9A9A9A"}},
        "drop_down_menu": {"font": letra(14)},
        "#titulo": {"font": letra(22, True)},
        "#secao": {"font": letra(14, True), "colours": {"normal_text": "#6BAED6"}},
        "#erro": {"colours": {"normal_text": "#FF6B5E"}},
        "#aviso": {"colours": {"normal_text": "#FDAE61"}},
        "#previa": {"font": letra(16, True)},
    }


def _nitidez_no_windows():
    """Avisa o Windows que a janela lida com DPI, para ele não esticar (e borrar)
    a imagem quando a escala da tela é 125%, 150%..."""
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass


class App:
    def __init__(self, tamanho=TAMANHO_INICIAL):
        _nitidez_no_windows()
        pygame.init()
        pygame.display.set_caption(TITULO)
        self.tamanho = tamanho
        self.janela = pygame.display.set_mode(tamanho, pygame.RESIZABLE)
        self.manager = pygame_gui.UIManager(tamanho)
        self.manager.get_theme().load_theme(tema(_fonte_do_sistema(self.manager)))
        self.relogio = pygame.time.Clock()
        self.rodando = True
        self.erro = None
        # Para voltar ao hub com os mesmos campos.
        self.ultimo_formulario = None
        self.ultima_descricao = None
        self.ultima_previa = ""

        from interface.tela_hub import TelaHub
        self.tela = TelaHub(self)

    def trocar_tela(self, nova):
        self.tela.fechar()
        self.tela = nova
        self.erro = None

    def _falha(self, erro):
        traceback.print_exc()
        self.erro = f"Erro inesperado: {erro}"

    def passo(self):
        """Um quadro: eventos, atualização e desenho."""
        dt = self.relogio.tick(60) / 1000
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                self.rodando = False
                continue
            if evento.type == pygame.VIDEORESIZE:
                self.tamanho = (max(evento.w, 800), max(evento.h, 600))
                self.janela = pygame.display.set_mode(self.tamanho, pygame.RESIZABLE)
                self.manager.set_window_resolution(self.tamanho)
                self.tela.redimensionar()
            self.manager.process_events(evento)
            try:
                self.tela.tratar_evento(evento)
            except Exception as erro:   # a janela não pode fechar com erro
                self._falha(erro)

        self.manager.update(dt)
        try:
            self.tela.atualizar(dt)
            self.tela.desenhar(self.janela)
        except Exception as erro:
            self._falha(erro)
        self.manager.draw_ui(self.janela)
        if self.erro:
            texto(self.janela, self.erro, (self.tamanho[0] // 2, self.tamanho[1] - 12), 14,
                  cor=(255, 107, 94))
        pygame.display.flip()

    def rodar(self):
        while self.rodando:
            self.passo()
        pygame.quit()
