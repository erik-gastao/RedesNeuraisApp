"""Testes do graficos.py e da opção de gráfico do menu.

    python -m unittest discover testes
"""

import os

# Sem janela durante os testes: o matplotlib só desenha em arquivo.
os.environ["MPLBACKEND"] = "Agg"

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import main
from graficos import grafico_do_erro
from test_main import executar

TEM_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None
XOR = str(Path(__file__).resolve().parent.parent / "bases" / "xor.csv")
PNG = b"\x89PNG"  # todo arquivo PNG começa com estes bytes


@unittest.skipUnless(TEM_MATPLOTLIB, "matplotlib não instalado")
class TesteGraficoDoErro(unittest.TestCase):

    def setUp(self):
        pasta = tempfile.TemporaryDirectory()
        self.addCleanup(pasta.cleanup)
        self.pasta = Path(pasta.name)

    def test_salva_png_e_cria_a_pasta(self):
        caminho = self.pasta / "nova_pasta" / "erro.png"
        devolvido = grafico_do_erro([0.25, 0.2, 0.1, 0.05], caminho, mostrar=False)
        self.assertEqual(devolvido, caminho)
        self.assertEqual(caminho.read_bytes()[:4], PNG)

    def test_uma_epoca_so(self):
        caminho = self.pasta / "erro.png"
        grafico_do_erro([0.25], caminho, mostrar=False)
        self.assertTrue(caminho.exists())

    def test_nao_deixa_figura_aberta(self):
        import matplotlib.pyplot as plt
        grafico_do_erro([0.3, 0.1], self.pasta / "erro.png", mostrar=False)
        self.assertEqual(plt.get_fignums(), [])

    def test_curva_de_validacao(self):
        import matplotlib.pyplot as plt
        figuras = []
        fechar = plt.close
        with patch.object(plt, "close", side_effect=lambda f=None: (figuras.append(f), fechar(f))):
            grafico_do_erro([0.3, 0.2, 0.1], self.pasta / "erro.png", mostrar=False,
                            historico_validacao=[0.35, 0.25, 0.2])
        eixo = figuras[0].axes[0]
        self.assertEqual(len(eixo.get_lines()), 2)
        self.assertEqual([t.get_text() for t in eixo.get_legend().get_texts()],
                         ["treino", "validação"])

    def test_opcao_do_menu_salva_o_grafico(self):
        with patch.object(main, "PASTA_DE_GRAFICOS", self.pasta):
            saida = executar([
                "s", "2", "1", "4", "sigmoid", "1", "", "s", "42", XOR, "n", "0.5", "200",
                "2", "100",   # treinar mais: o gráfico junta as 300 épocas
                "4",
                "0",
            ])
        arquivos = list(self.pasta.glob("erro_xor_*.png"))
        self.assertEqual(len(arquivos), 1)
        self.assertEqual(arquivos[0].read_bytes()[:4], PNG)
        self.assertIn("Gráfico salvo em:", saida)
        self.assertNotIn("Traceback", saida)


class TesteSemMatplotlib(unittest.TestCase):

    def test_menu_avisa_e_continua(self):
        with patch.object(main, "grafico_do_erro", side_effect=ImportError):
            saida = executar([
                "s", "2", "0", "1", "", "s", "1", XOR, "n", "", "10", "4", "0",
            ])
        self.assertIn("pip install matplotlib", saida)
        self.assertTrue(saida.rstrip().endswith("Até mais!"))


if __name__ == "__main__":
    unittest.main()
