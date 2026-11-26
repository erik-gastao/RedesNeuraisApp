"""Teste de fumaça da interface gráfica, sem abrir janela (SDL_VIDEODRIVER=dummy).

Pulado se o pygame não estiver instalado.

    python -m unittest discover testes
"""

import importlib.util
import os
import unittest
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

TEM_PYGAME = (importlib.util.find_spec("pygame") is not None
              and importlib.util.find_spec("pygame_gui") is not None)
RAIZ = Path(__file__).resolve().parent.parent
IRIS = RAIZ / "Datasets" / "bezdekIris.data"
DATASET = RAIZ / "Datasets" / "dataset.csv"


@unittest.skipUnless(TEM_PYGAME, "pygame-ce e pygame_gui não instalados")
class TesteHubERede(unittest.TestCase):

    def setUp(self):
        import pygame
        import pygame_gui
        from interface.app import App
        from interface.tela_hub import TelaHub
        from interface.tela_player import TelaPlayer
        self.pygame, self.pygame_gui = pygame, pygame_gui
        self.TelaHub, self.TelaPlayer = TelaHub, TelaPlayer
        self.app = App((1280, 720))
        self.addCleanup(pygame.quit)

    def quadros(self, n=3):
        for _ in range(n):
            self.app.passo()

    def esperar_player(self):
        """Roda quadros até a gravação (thread) terminar e o player abrir."""
        import time
        limite = time.time() + 60
        while not isinstance(self.app.tela, self.TelaPlayer) and time.time() < limite:
            self.app.passo()
        self.quadros()

    def clicar(self, botao):
        """Simula o evento que o pygame_gui gera ao clicar no botão."""
        self.pygame.event.post(self.pygame.event.Event(
            self.pygame_gui.UI_BUTTON_PRESSED, {"ui_element": botao}))
        self.quadros(1)

    def test_abre_no_hub_sem_base_e_gerar_desligado(self):
        self.quadros()
        hub = self.app.tela
        self.assertIsInstance(hub, self.TelaHub)
        self.assertIn("base", hub.erros)
        self.assertFalse(hub.botao_gerar.is_enabled)
        self.assertIsNone(self.app.erro)

    def test_iris_4_5_3_mostra_a_rede_com_os_pesos_iniciais(self):
        hub = self.app.tela
        hub.escolher_base(IRIS)
        self.quadros()
        self.assertEqual(hub.erros, {})
        self.assertEqual(hub.rotulo_base.text, "150 amostras · 4 entradas · 3 classes")
        self.assertTrue(hub.botao_gerar.is_enabled)

        hub.form.epocas = "5"
        hub.form.epocas_detalhadas = "1, fim"
        hub.validar()
        self.clicar(hub.botao_gerar)
        self.esperar_player()
        tela = self.app.tela
        self.assertIsInstance(tela, self.TelaPlayer)
        for tecla in ("right", "right", "space", "end", "left"):
            self.pygame.event.post(self.pygame.event.Event(
                self.pygame.KEYDOWN, {"key": self.pygame.key.key_code(tecla), "mod": 0}))
            self.quadros(1)
        rede = tela.experimento.rede
        self.assertEqual([c.W.shape for c in rede.camadas], [(5, 4), (3, 5)])
        self.assertEqual(len(tela.layout.posicoes), 4 + 5 + 3)
        self.assertIsNone(self.app.erro)

    def test_voltar_ao_hub_mantem_os_campos(self):
        hub = self.app.tela
        hub.escolher_base(IRIS)
        hub.form.semente = "7"
        hub.form.epocas = "3"
        hub.form.epocas_detalhadas = "1, fim"
        hub.validar()
        hub.gerar()
        self.esperar_player()
        self.app.tela.voltar()
        novo = self.app.tela
        self.assertIsInstance(novo, self.TelaHub)
        self.assertEqual(novo.form.semente, "7")
        self.assertEqual(novo.descricao["amostras"], 150)

    def test_percentuais_invalidos_mostram_erro_e_desligam_gerar(self):
        hub = self.app.tela
        hub.escolher_base(IRIS)
        hub.form.divisao = ["70", "20", "20"]
        hub.validar()
        self.quadros()
        self.assertIn("somar 100", hub.rotulos_erro["divisao"].text)
        self.assertFalse(hub.botao_gerar.is_enabled)

    def test_base_numerica_sugere_linear_e_avisa_com_sigmoid(self):
        hub = self.app.tela
        hub.escolher_base(DATASET)
        self.assertEqual(hub.form.ativacao_saida, "linear")
        self.assertEqual(hub.rotulo_aviso.text, "")
        hub.form.ativacao_saida = "sigmoid"
        hub.validar()
        self.assertIn("Use a ativação linear", hub.rotulo_aviso.text)

    def test_adicionar_e_remover_camada(self):
        hub = self.app.tela
        hub.escolher_base(IRIS)
        self.clicar(next(b for b, acao in hub.ao_clicar.items()
                         if b.text == "+ adicionar camada"))
        self.assertEqual(len(self.app.tela.form.ocultas), 2)
        self.assertIn("[4 sigmoid]", self.app.tela.rotulo_previa.text)
        self.app.tela.remover_camada(0)
        self.assertEqual(len(self.app.tela.form.ocultas), 1)

    def test_redimensionar_nao_quebra(self):
        hub = self.app.tela
        hub.escolher_base(IRIS)
        hub.form.epocas = "3"
        hub.form.epocas_detalhadas = "1, fim"
        hub.validar()
        hub.gerar()
        self.esperar_player()
        self.pygame.event.post(self.pygame.event.Event(self.pygame.VIDEORESIZE, {"w": 1600, "h": 900}))
        self.quadros()
        self.assertEqual(self.app.tamanho, (1600, 900))
        maior_x = max(x for x, _ in self.app.tela.layout.posicoes.values())
        self.assertGreater(maior_x, 900)   # painel lateral ocupa a direita
        self.assertIsNone(self.app.erro)


if __name__ == "__main__":
    unittest.main()
