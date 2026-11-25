"""Testes do dados.py. Rodar na pasta raiz com:

    python -m unittest discover testes
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from dados import (
    acuracia,
    aplicar_normalizacao,
    desnormalizar,
    carregar_csv,
    listar_bases,
    normalizar,
    normalizar_por_lotes,
    separar_treino_teste,
)

PASTA_BASES = Path(__file__).resolve().parent.parent / "bases"
PASTA_DATASET = Path(__file__).resolve().parent.parent / "Dataset"


class TesteCarregarCsv(unittest.TestCase):

    def setUp(self):
        # Pasta temporária para criar CSVs de teste; apagada no fim.
        pasta = tempfile.TemporaryDirectory()
        self.addCleanup(pasta.cleanup)
        self.pasta = Path(pasta.name)

    def criar(self, conteudo):
        caminho = self.pasta / "base.csv"
        caminho.write_text(conteudo, encoding="utf-8")
        return caminho

    def test_carrega_xor(self):
        X, Y, _ = carregar_csv(PASTA_BASES / "xor.csv", n_saidas=1)
        np.testing.assert_array_equal(X, [[0, 0], [0, 1], [1, 0], [1, 1]])
        np.testing.assert_array_equal(Y, [[0], [1], [1], [0]])

    def test_carrega_and(self):
        X, Y, _ = carregar_csv(PASTA_BASES / "and.csv", n_saidas=1)
        self.assertEqual(X.shape, (4, 2))
        np.testing.assert_array_equal(Y, [[0], [0], [0], [1]])

    def test_sem_cabecalho(self):
        X, Y, _ = carregar_csv(self.criar("0,0,0\n1,1,1\n"), n_saidas=1)
        self.assertEqual(X.shape, (2, 2))
        self.assertEqual(Y.shape, (2, 1))

    def test_ultimas_colunas_sao_a_saida(self):
        X, Y, _ = carregar_csv(self.criar("a,b,c,d\n1,2,3,4\n5,6,7,8\n"), n_saidas=2)
        np.testing.assert_array_equal(X, [[1, 2], [5, 6]])
        np.testing.assert_array_equal(Y, [[3, 4], [7, 8]])

    def test_ignora_linhas_em_branco_e_espacos(self):
        X, Y, _ = carregar_csv(self.criar("x,y\n\n 1.5 , 0\n2.5,1\n\n"), n_saidas=1)
        np.testing.assert_array_equal(X, [[1.5], [2.5]])
        np.testing.assert_array_equal(Y, [[0], [1]])

    def test_arquivo_inexistente(self):
        with self.assertRaises(FileNotFoundError):
            carregar_csv(self.pasta / "nao_existe.csv", n_saidas=1)

    def test_arquivo_vazio(self):
        with self.assertRaisesRegex(ValueError, "vazio"):
            carregar_csv(self.criar(""), n_saidas=1)

    def test_so_cabecalho(self):
        with self.assertRaisesRegex(ValueError, "cabeçalho"):
            carregar_csv(self.criar("x1,x2,y\n"), n_saidas=1)

    def test_colunas_insuficientes(self):
        with self.assertRaisesRegex(ValueError, "coluna"):
            carregar_csv(self.criar("x,y\n0,1\n"), n_saidas=2)

    def test_linha_com_numero_diferente_de_colunas(self):
        with self.assertRaisesRegex(ValueError, "Linha 3"):
            carregar_csv(self.criar("x1,x2,y\n0,0,0\n0,1\n"), n_saidas=1)

    def test_valor_nao_numerico(self):
        with self.assertRaisesRegex(ValueError, "Linha 3.*'abc'"):
            carregar_csv(self.criar("x1,x2,y\n0,0,0\n0,abc,1\n"), n_saidas=1)


class TesteClasseEmTexto(unittest.TestCase):
    """Bases com o nome da classe na última coluna (ex.: Iris)."""

    def setUp(self):
        pasta = tempfile.TemporaryDirectory()
        self.addCleanup(pasta.cleanup)
        self.pasta = Path(pasta.name)

    def criar(self, conteudo):
        caminho = self.pasta / "base.csv"
        caminho.write_text(conteudo, encoding="utf-8")
        return caminho

    def test_saida_numerica_nao_tem_classes(self):
        _, _, classes = carregar_csv(PASTA_BASES / "xor.csv", n_saidas=1)
        self.assertIsNone(classes)

    def test_classe_vira_uma_coluna_por_classe(self):
        caminho = self.criar("1,2,gato\n3,4,cao\n5,6,gato\n7,8,ave\n")
        X, Y, classes = carregar_csv(caminho, n_saidas=3)
        self.assertEqual(classes, ["ave", "cao", "gato"])  # ordem alfabética
        np.testing.assert_array_equal(X, [[1, 2], [3, 4], [5, 6], [7, 8]])
        np.testing.assert_array_equal(
            Y, [[0, 0, 1], [0, 1, 0], [0, 0, 1], [1, 0, 0]]
        )

    def test_primeira_linha_com_classe_nao_e_cabecalho(self):
        # Sem cabeçalho: a primeira linha tem texto, mas é dado.
        X, _, _ = carregar_csv(self.criar("1,2,a\n3,4,b\n"), n_saidas=2)
        self.assertEqual(X.shape, (2, 2))

    def test_cabecalho_com_classe_em_texto(self):
        caminho = self.criar("largura,altura,especie\n1,2,a\n3,4,b\n")
        X, Y, classes = carregar_csv(caminho, n_saidas=2)
        np.testing.assert_array_equal(X, [[1, 2], [3, 4]])
        self.assertEqual(classes, ["a", "b"])

    def test_numero_de_saidas_diferente_do_numero_de_classes(self):
        caminho = self.criar("1,2,a\n3,4,b\n5,6,c\n")
        with self.assertRaisesRegex(ValueError, "3 classe.*3 neurônio.*mas tem 1"):
            carregar_csv(caminho, n_saidas=1)

    def test_entrada_nao_numerica(self):
        with self.assertRaisesRegex(ValueError, "Linha 2.*'xyz'"):
            carregar_csv(self.criar("1,2,a\n3,xyz,b\n"), n_saidas=2)

    def test_iris(self):
        for nome in ("iris.data", "bezdekIris.data"):
            X, Y, classes = carregar_csv(PASTA_DATASET / nome, n_saidas=3)
            self.assertEqual(X.shape, (150, 4), nome)
            self.assertEqual(Y.shape, (150, 3), nome)
            self.assertEqual(
                classes, ["Iris-setosa", "Iris-versicolor", "Iris-virginica"], nome
            )
            # 50 amostras de cada classe, e cada amostra em uma classe só.
            np.testing.assert_array_equal(Y.sum(axis=0), [50, 50, 50])
            np.testing.assert_array_equal(Y.sum(axis=1), np.ones(150))
            # A primeira linha do arquivo não pode ser descartada como cabeçalho.
            np.testing.assert_array_equal(X[0], [5.1, 3.5, 1.4, 0.2])


class TesteListarBases(unittest.TestCase):

    def test_lista_so_arquivos_de_base_em_ordem(self):
        with tempfile.TemporaryDirectory() as pasta:
            pasta = Path(pasta)
            for nome in ("b.csv", "a.data", "c.TXT", "leia.names", "Index"):
                (pasta / nome).write_text("1,0\n", encoding="utf-8")
            (pasta / "subpasta.csv").mkdir()
            nomes = [arquivo.name for arquivo in listar_bases([pasta])]
        self.assertEqual(nomes, ["a.data", "b.csv", "c.TXT"])

    def test_pasta_inexistente_e_ignorada(self):
        self.assertEqual(listar_bases([Path("pasta_que_nao_existe")]), [])

    def test_pastas_do_projeto(self):
        nomes = [arquivo.name for arquivo in listar_bases([PASTA_BASES, PASTA_DATASET])]
        self.assertEqual(
            nomes, ["and.csv", "xor.csv", "bezdekIris.data", "iris.data"]
        )


class TesteNormalizar(unittest.TestCase):

    def test_leva_cada_coluna_para_zero_a_um(self):
        X = np.array([[10.0, 1.0], [20.0, 3.0], [30.0, 5.0]])
        X_norm, minimos, maximos = normalizar(X)
        np.testing.assert_allclose(X_norm, [[0, 0], [0.5, 0.5], [1, 1]])
        np.testing.assert_array_equal(minimos, [10, 1])
        np.testing.assert_array_equal(maximos, [30, 5])

    def test_coluna_constante_vira_zero(self):
        X = np.array([[7.0, 1.0], [7.0, 2.0]])
        X_norm, _, _ = normalizar(X)
        np.testing.assert_array_equal(X_norm[:, 0], [0, 0])
        self.assertTrue(np.all(np.isfinite(X_norm)))

    def test_nao_altera_o_x_original(self):
        X = np.array([[10.0], [20.0]])
        normalizar(X)
        np.testing.assert_array_equal(X, [[10.0], [20.0]])

    def test_aplicar_em_entrada_nova_usa_a_mesma_escala(self):
        X = np.array([[10.0], [30.0]])
        _, minimos, maximos = normalizar(X)
        nova = aplicar_normalizacao(np.array([[20.0]]), minimos, maximos)
        np.testing.assert_allclose(nova, [[0.5]])


class TesteNormalizarPorLotes(unittest.TestCase):

    def test_cada_lote_usa_sua_propria_escala(self):
        X = np.array([[0.0], [10.0], [100.0], [200.0]])
        X_norm = normalizar_por_lotes(X, 2)
        # Lote 1: 0..10; lote 2: 100..200. Cada um vai de 0 a 1.
        np.testing.assert_allclose(X_norm, [[0], [1], [0], [1]])

    def test_lote_maior_que_a_base_e_igual_a_normalizar(self):
        X = np.array([[3.0, 1.0], [5.0, 9.0], [4.0, 2.0]])
        np.testing.assert_allclose(normalizar_por_lotes(X, 10), normalizar(X)[0])

    def test_sobra_de_uma_linha_entra_no_lote_anterior(self):
        X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
        X_norm = normalizar_por_lotes(X, 2)
        # Lotes: [0, 1] e [2, 3, 4] (o 4 sozinho viraria 0).
        np.testing.assert_allclose(X_norm.ravel(), [0, 1, 0, 0.5, 1])

    def test_lote_menor_que_dois_e_erro(self):
        with self.assertRaises(ValueError):
            normalizar_por_lotes(np.zeros((4, 1)), 1)


class TesteDesnormalizar(unittest.TestCase):

    def test_desfaz_a_normalizacao(self):
        Y = np.array([[100.0], [125.0], [150.0]])
        Y_norm, minimos, maximos = normalizar(Y)
        np.testing.assert_allclose(desnormalizar(Y_norm, minimos, maximos), Y)

    def test_coluna_constante_volta_inteira(self):
        Y = np.array([[7.0], [7.0]])
        Y_norm, minimos, maximos = normalizar(Y)
        np.testing.assert_allclose(desnormalizar(Y_norm, minimos, maximos), Y)

    def test_zero_e_um_nao_mudam(self):
        Y = np.array([[0.0, 1.0], [1.0, 0.0]])
        Y_norm, minimos, maximos = normalizar(Y)
        np.testing.assert_array_equal(Y_norm, Y)


class TesteSepararTreinoTeste(unittest.TestCase):

    def base(self, n):
        X = np.arange(n * 2, dtype=float).reshape(n, 2)
        Y = np.arange(n, dtype=float).reshape(n, 1)
        return X, Y

    def test_base_pequena_nao_separa(self):
        X, Y = self.base(4)
        X_treino, Y_treino, X_teste, Y_teste, separou = separar_treino_teste(X, Y, seed=1)
        self.assertFalse(separou)
        np.testing.assert_array_equal(X_treino, X)
        np.testing.assert_array_equal(X_teste, X)
        np.testing.assert_array_equal(Y_treino, Y)
        np.testing.assert_array_equal(Y_teste, Y)

    def test_tamanhos(self):
        X, Y = self.base(50)
        X_treino, Y_treino, X_teste, Y_teste, separou = separar_treino_teste(X, Y, seed=1)
        self.assertTrue(separou)
        self.assertEqual(X_treino.shape, (40, 2))
        self.assertEqual(Y_treino.shape, (40, 1))
        self.assertEqual(X_teste.shape, (10, 2))
        self.assertEqual(Y_teste.shape, (10, 1))

    def test_nenhuma_amostra_se_perde_nem_se_repete(self):
        X, Y = self.base(50)
        _, Y_treino, _, Y_teste, _ = separar_treino_teste(X, Y, seed=1)
        juntos = np.sort(np.concatenate([Y_treino, Y_teste]).ravel())
        np.testing.assert_array_equal(juntos, Y.ravel())

    def test_x_e_y_continuam_alinhados(self):
        # Na base de teste, a linha i tem X = [2i, 2i + 1] e Y = [i].
        X, Y = self.base(50)
        X_treino, Y_treino, X_teste, Y_teste, _ = separar_treino_teste(X, Y, seed=1)
        np.testing.assert_array_equal(X_treino[:, 0], 2 * Y_treino[:, 0])
        np.testing.assert_array_equal(X_teste[:, 0], 2 * Y_teste[:, 0])

    def test_embaralha(self):
        X, Y = self.base(50)
        _, _, _, Y_teste, _ = separar_treino_teste(X, Y, seed=1)
        self.assertFalse(np.array_equal(Y_teste.ravel(), np.arange(10)))

    def test_mesma_seed_mesma_separacao(self):
        X, Y = self.base(50)
        a = separar_treino_teste(X, Y, seed=5)
        b = separar_treino_teste(X, Y, seed=5)
        np.testing.assert_array_equal(a[2], b[2])


class TesteAcuracia(unittest.TestCase):

    def test_tudo_certo(self):
        Y = np.array([[0.0], [1.0]])
        self.assertEqual(acuracia(Y, Y.copy()), 1.0)

    def test_metade_certa(self):
        Y = np.array([[0.0], [1.0], [1.0], [0.0]])
        classificado = np.array([[0.0], [1.0], [0.0], [1.0]])
        self.assertEqual(acuracia(Y, classificado), 0.5)

    def test_com_varias_saidas_todas_precisam_bater(self):
        Y = np.array([[1.0, 0.0], [0.0, 1.0]])
        classificado = np.array([[1.0, 0.0], [1.0, 1.0]])  # 2ª amostra erra 1 saída
        self.assertEqual(acuracia(Y, classificado), 0.5)


if __name__ == "__main__":
    unittest.main()
