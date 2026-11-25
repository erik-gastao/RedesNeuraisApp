"""Testes do experimento.py: o fluxo do programa sem input()/print().

    python -m unittest discover testes
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from experimento import Experimento
from rede.rede_neural import RedeNeural
from rede.treinamento import treinar

RAIZ = Path(__file__).resolve().parent.parent
XOR = RAIZ / "bases" / "xor.csv"
IRIS = RAIZ / "Datasets" / "bezdekIris.data"
DATASET_PROFESSOR = RAIZ / "Datasets" / "dataset.csv"


def iris(**opcoes):
    return Experimento.da_base(IRIS, [(5, "sigmoid")], "sigmoid", True, 1, n_saidas=3, **opcoes)


class TesteAprendizado(unittest.TestCase):

    def test_xor_aprende(self):
        exp = Experimento.da_base(XOR, [(4, "sigmoid")], "sigmoid", True, 42, normalizar=False)
        exp.treinar(3000, 0.5)
        avaliacao = exp.avaliar()
        self.assertEqual(avaliacao["acuracia"], 1.0)
        self.assertEqual(avaliacao["acertos"], 4)

    def test_iris_acima_de_90(self):
        exp = iris()
        exp.treinar(200, 0.1)
        self.assertGreaterEqual(exp.avaliar()["acuracia"], 0.9)

    def test_dataset_do_professor_melhor_que_a_media(self):
        exp = Experimento.da_base(DATASET_PROFESSOR, [(8, "relu")], "linear", True, 1)
        exp.treinar(300, 0.00001)
        avaliacao = exp.avaliar()
        self.assertFalse(avaliacao["classificacao"])
        self.assertIsNone(avaliacao["acuracia"])
        Y = np.genfromtxt(DATASET_PROFESSOR, delimiter=",", skip_header=1)[:, -1]
        self.assertLess(avaliacao["mse_teste"], 0.8 * np.var(Y))

    def test_avaliar_traz_mse_dos_tres_conjuntos_e_linhas_do_teste(self):
        exp = iris()
        exp.treinar(5, 0.1)
        avaliacao = exp.avaliar()
        for chave in ("mse_treino", "mse_validacao", "mse_teste"):
            self.assertIsInstance(avaliacao[chave], float)
        self.assertEqual(len(avaliacao["linhas"]), 22)
        linha = avaliacao["linhas"][0]
        self.assertEqual(set(linha), {"entrada", "esperado", "y_hat", "classificado", "acertou"})


class TesteMontagem(unittest.TestCase):

    def test_entradas_e_saidas_vem_da_base(self):
        exp = iris()
        self.assertEqual(exp.rede.n_entradas, 4)
        self.assertEqual(exp.rede.n_saidas, 3)
        self.assertEqual(exp.classes, ["Iris-setosa", "Iris-versicolor", "Iris-virginica"])
        self.assertTrue(exp.classificacao)

    def test_base_incompativel_com_a_rede_da_valueerror(self):
        rede = RedeNeural(3, [], 1, True, 1)
        X, Y = np.zeros((4, 2)), np.zeros((4, 1))
        with self.assertRaises(ValueError):
            Experimento(rede, X, Y, None, "xor.csv", seed=1)

    def test_divisao_com_os_percentuais_pedidos(self):
        exp = iris(percentuais=(60, 20, 20))
        self.assertTrue(exp.separou)
        self.assertEqual([len(exp.X_treino), len(exp.X_validacao), len(exp.X_teste)], [90, 30, 30])

    def test_percentuais_invalidos_dao_valueerror(self):
        exp = iris()
        with self.assertRaises(ValueError):
            exp.dividir((70, 30))

    def test_base_pequena_nao_separa_e_fica_sem_validacao(self):
        exp = Experimento.da_base(XOR, [(4, "sigmoid")], "sigmoid", True, 42)
        self.assertFalse(exp.separou)
        self.assertIsNone(exp.X_validacao)
        self.assertEqual(len(exp.X_teste), 4)


class TesteNormalizacao(unittest.TestCase):

    def test_desligada_nao_muda_X(self):
        exp = iris(normalizar=False)
        self.assertIsNone(exp.normalizacao)
        np.testing.assert_array_equal(exp.X_treino, exp.X_treino_original)

    def test_usa_minimo_e_maximo_so_do_treino(self):
        exp = iris()
        self.assertEqual(exp.normalizacao["funcao"], "min-max")
        np.testing.assert_array_equal(exp.normalizacao["minimos"], exp.X_treino_original.min(axis=0))
        np.testing.assert_array_equal(exp.normalizacao["maximos"], exp.X_treino_original.max(axis=0))
        self.assertGreaterEqual(exp.X_treino.min(), 0)
        self.assertLessEqual(exp.X_treino.max(), 1)

    def test_saidas_nunca_sao_normalizadas(self):
        exp = Experimento.da_base(DATASET_PROFESSOR, [(8, "relu")], "linear", True, 1)
        self.assertGreater(exp.Y_treino.max(), 1000)

    def test_por_lotes_muda_so_o_treino(self):
        inteiro = iris()
        lotes = iris(tamanho_lote=30)
        self.assertFalse(np.allclose(inteiro.X_treino, lotes.X_treino))
        np.testing.assert_array_equal(inteiro.X_teste, lotes.X_teste)
        self.assertEqual(inteiro.normalizacao, lotes.normalizacao)


class TesteTreino(unittest.TestCase):

    def test_treinar_mais_continua_o_historico(self):
        exp = iris()
        exp.treinar(5, 0.1)
        exp.treinar(3, 0.1)
        self.assertEqual(len(exp.historico["treino"]), 8)
        self.assertEqual(len(exp.historico["validacao"]), 8)

    def test_mesmo_resultado_que_o_treino_direto(self):
        exp = iris()
        novo = exp.treinar(5, 0.1)
        rede = RedeNeural(4, [(5, "sigmoid")], 3, True, 1, "sigmoid")
        direto = treinar(rede, exp.X_treino, exp.Y_treino, 5, 0.1,
                         exp.X_validacao, exp.Y_validacao, seed=1)
        self.assertEqual(novo, direto)

    def test_reiniciar_volta_aos_pesos_iniciais(self):
        exp = iris()
        iniciais = [camada.W for camada in exp.rede.camadas]
        exp.treinar(3, 0.1)
        exp.reiniciar()
        self.assertEqual(exp.historico, {"treino": [], "validacao": []})
        for camada, W in zip(exp.rede.camadas, iniciais):
            np.testing.assert_array_equal(camada.W, W)


class TestePrevisaoEPersistencia(unittest.TestCase):

    def test_prever_entrada_normaliza_igual_ao_treino(self):
        exp = iris()
        exp.treinar(20, 0.1)
        y_hat, classe = exp.prever_entrada(exp.X_teste_original[0])
        np.testing.assert_array_equal(y_hat, exp.rede.prever(exp.X_teste[0])[0])
        self.assertIn(classe, exp.classes)

    def test_prever_entrada_na_regressao_nao_tem_classe(self):
        exp = Experimento.da_base(DATASET_PROFESSOR, [(8, "relu")], "linear", True, 1)
        _, classe = exp.prever_entrada([300, 60])
        self.assertIsNone(classe)

    def test_prever_com_quantidade_errada_da_valueerror(self):
        with self.assertRaises(ValueError):
            iris().prever_entrada([1.0, 2.0])

    def test_salvar_e_carregar_dao_o_mesmo_y_hat(self):
        exp = iris()
        exp.treinar(20, 0.1)
        with tempfile.TemporaryDirectory() as pasta:
            caminho = exp.salvar(Path(pasta) / "iris.json")
            carregado = Experimento.carregar(caminho)
        entrada = exp.X_teste_original[3]
        np.testing.assert_array_equal(exp.prever_entrada(entrada)[0],
                                      carregado.prever_entrada(entrada)[0])
        self.assertEqual(carregado.classes, exp.classes)

    def test_carregado_fica_em_modo_previsao(self):
        exp = iris()
        with tempfile.TemporaryDirectory() as pasta:
            carregado = Experimento.carregar(exp.salvar(Path(pasta) / "iris.json"))
        self.assertIsNone(carregado.X_treino)
        self.assertTrue(carregado.classificacao)


if __name__ == "__main__":
    unittest.main()
