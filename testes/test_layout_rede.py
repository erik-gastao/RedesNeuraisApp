"""Testes do layout da rede no player (interface/layout_rede.py, sem pygame).

    python -m unittest discover testes
"""

import unittest

import numpy as np

from interface.layout_rede import (
    AZUL,
    BRANCO,
    CINZA_CLARO,
    LARANJA,
    VERDE,
    VERMELHO,
    calcular_layout,
    contorno_saida,
    cor_ativacao,
    cor_entrada,
    cor_gradiente,
    cor_peso,
    espessura,
    maior_peso,
    neuronios_visiveis,
)

AREA = (100, 50, 800, 600)   # x, y, largura, altura


def dentro(ponto, raio, area=AREA):
    x, y = ponto
    ax, ay, largura, altura = area
    return ax <= x - raio and x + raio <= ax + largura and ay <= y - raio and y + raio <= ay + altura


def mais_perto(cor, a, b):
    """True se a cor está mais perto de a do que de b."""
    return np.linalg.norm(np.subtract(cor, a)) < np.linalg.norm(np.subtract(cor, b))


class TesteNeuroniosVisiveis(unittest.TestCase):

    def test_camada_pequena_mostra_todos(self):
        self.assertEqual(neuronios_visiveis(5), [0, 1, 2, 3, 4])
        self.assertEqual(neuronios_visiveis(12), list(range(12)))

    def test_camada_grande_mostra_11_reticencias_e_o_ultimo(self):
        self.assertEqual(neuronios_visiveis(30), list(range(11)) + [None, 29])
        self.assertEqual(neuronios_visiveis(13), list(range(11)) + [None, 12])


class TestePosicoes(unittest.TestCase):

    def test_todos_dentro_da_area(self):
        layout = calcular_layout([4, 30, 3], AREA)
        for ponto in layout.posicoes.values():
            self.assertTrue(dentro(ponto, layout.raio), ponto)
        for ponto in layout.reticencias.values():
            self.assertTrue(dentro(ponto, layout.raio), ponto)

    def test_uma_posicao_por_neuronio_visivel(self):
        layout = calcular_layout([4, 30, 3], AREA)
        self.assertEqual(len(layout.posicoes), 4 + 12 + 3)
        self.assertIn((1, 29), layout.posicoes)
        self.assertNotIn((1, 15), layout.posicoes)
        self.assertEqual(list(layout.reticencias), [1])

    def test_colunas_igualmente_espacadas_da_esquerda_para_a_direita(self):
        layout = calcular_layout([2, 4, 4, 1], AREA)
        xs = [layout.posicoes[(c, 0)][0] for c in range(4)]
        self.assertEqual(xs, sorted(xs))
        np.testing.assert_allclose(np.diff(xs), np.diff(xs)[0])

    def test_mesmo_espacamento_vertical_e_camadas_centralizadas(self):
        layout = calcular_layout([2, 5, 1], AREA)
        centro = AREA[1] + AREA[3] / 2
        passos = []
        for camada, tamanho in enumerate([2, 5, 1]):
            ys = [layout.posicoes[(camada, n)][1] for n in range(tamanho)]
            self.assertAlmostEqual(np.mean(ys), centro)
            passos += list(np.diff(ys))
        np.testing.assert_allclose(passos, passos[0])

    def test_reticencias_ficam_entre_o_11o_e_o_ultimo(self):
        layout = calcular_layout([2, 30, 1], AREA)
        y_reticencias = layout.reticencias[1][1]
        self.assertLess(layout.posicoes[(1, 10)][1], y_reticencias)
        self.assertLess(y_reticencias, layout.posicoes[(1, 29)][1])

    def test_raio_entre_10_e_24(self):
        self.assertEqual(calcular_layout([2, 3, 1], AREA).raio, 24)
        self.assertEqual(calcular_layout([2, 30, 1], (0, 0, 300, 150)).raio, 10)

    def test_recalcula_ao_mudar_o_tamanho(self):
        pequeno = calcular_layout([2, 3, 1], (0, 0, 640, 360))
        grande = calcular_layout([2, 3, 1], (0, 0, 1280, 720))
        self.assertGreater(grande.posicoes[(2, 0)][0], pequeno.posicoes[(2, 0)][0])


class TesteCores(unittest.TestCase):

    def test_sigmoid_branco_em_0_e_azul_em_1(self):
        self.assertEqual(cor_ativacao(0.0, "sigmoid"), BRANCO)
        self.assertEqual(cor_ativacao(1.0, "sigmoid"), AZUL)

    def test_tanh_laranja_branco_azul(self):
        self.assertEqual(cor_ativacao(-1.0, "tanh"), LARANJA)
        self.assertEqual(cor_ativacao(0.0, "tanh"), BRANCO)
        self.assertEqual(cor_ativacao(1.0, "tanh"), AZUL)

    def test_relu_usa_o_maior_valor_da_camada(self):
        self.assertEqual(cor_ativacao(0.0, "relu", maior=4.0), BRANCO)
        self.assertEqual(cor_ativacao(4.0, "relu", maior=4.0), AZUL)

    def test_linear_negativo_puxa_para_laranja(self):
        self.assertEqual(cor_ativacao(-3.0, "linear", maior=3.0), LARANJA)
        self.assertEqual(cor_ativacao(3.0, "linear", maior=3.0), AZUL)
        self.assertTrue(mais_perto(cor_ativacao(-1.0, "linear", maior=3.0), LARANJA, AZUL))

    def test_ainda_nao_calculado_e_cinza(self):
        self.assertEqual(cor_ativacao(None, "sigmoid"), CINZA_CLARO)
        self.assertEqual(cor_entrada(None, 1.0), CINZA_CLARO)

    def test_entrada_usa_o_maior_valor(self):
        self.assertEqual(cor_entrada(0.0, 2.0), BRANCO)
        self.assertEqual(cor_entrada(2.0, 2.0), AZUL)

    def test_peso_positivo_azul_negativo_laranja(self):
        self.assertEqual(cor_peso(0.7), AZUL)
        self.assertEqual(cor_peso(-0.7), LARANJA)

    def test_gradiente_usa_sinal_e_escala_propria(self):
        self.assertEqual(cor_gradiente(0.02, 0.02), AZUL)
        self.assertEqual(cor_gradiente(-0.02, 0.02), LARANJA)
        self.assertEqual(cor_gradiente(0.0, 0.02), BRANCO)

    def test_contorno_da_saida(self):
        self.assertEqual(contorno_saida(True), VERDE)
        self.assertEqual(contorno_saida(False), VERMELHO)


class TesteEspessuras(unittest.TestCase):

    def test_maior_peso_tem_espessura_maxima(self):
        self.assertEqual(espessura(-2.0, 2.0), 6.0)
        self.assertEqual(espessura(0.0, 2.0), 0.5)
        self.assertAlmostEqual(espessura(1.0, 2.0), 3.25)

    def test_rede_sem_pesos_nao_divide_por_zero(self):
        self.assertEqual(espessura(0.0, 0.0), 0.5)

    def test_maior_peso_da_rede(self):
        pesos = [(np.array([[0.5, -3.0]]), np.array([9.0])), (np.array([[1.0]]), np.array([0.0]))]
        self.assertEqual(maior_peso(pesos), 3.0)   # só os pesos, não os vieses


if __name__ == "__main__":
    unittest.main()
