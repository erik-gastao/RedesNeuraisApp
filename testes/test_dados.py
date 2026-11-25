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
    carregar_csv,
    listar_bases,
    normalizar,
    normalizar_por_lotes,
    e_classificacao,
    separar_treino_validacao_teste,
)

PASTA_BASES = Path(__file__).resolve().parent.parent / "bases"
PASTA_DATASET = Path(__file__).resolve().parent.parent / "Datasets"


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
            nomes, ["and.csv", "xor.csv", "bezdekIris.data", "dataset.csv", "iris.data"]
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


class TesteSepararTresConjuntos(unittest.TestCase):
    """Divisão aleatória em treino, validação e teste (seção 1.9)."""

    def base(self, n):
        X = np.arange(n * 2, dtype=float).reshape(n, 2)
        Y = np.arange(n, dtype=float).reshape(n, 1)
        return X, Y

    def separar(self, n, percentuais=(70, 15, 15), seed=1):
        X, Y = self.base(n)
        return separar_treino_validacao_teste(X, Y, percentuais, seed=seed)

    def test_tamanhos_70_15_15(self):
        X_tr, Y_tr, X_va, Y_va, X_te, Y_te, separou = self.separar(100)
        self.assertTrue(separou)
        self.assertEqual([len(X_tr), len(X_va), len(X_te)], [70, 15, 15])
        self.assertEqual([len(Y_tr), len(Y_va), len(Y_te)], [70, 15, 15])

    def test_dataset_do_professor(self):
        # 179 linhas: 15% = 26,85 → 27 na validação e 27 no teste.
        conjuntos = self.separar(179)
        self.assertEqual([len(conjuntos[i]) for i in (0, 2, 4)], [125, 27, 27])

    def test_percentuais_configuraveis(self):
        conjuntos = self.separar(50, (60, 20, 20))
        self.assertEqual([len(conjuntos[i]) for i in (0, 2, 4)], [30, 10, 10])

    def test_nenhuma_amostra_se_perde_nem_se_repete(self):
        X_tr, Y_tr, X_va, Y_va, X_te, Y_te, _ = self.separar(50)
        juntos = np.sort(np.concatenate([Y_tr, Y_va, Y_te]).ravel())
        np.testing.assert_array_equal(juntos, np.arange(50))

    def test_x_e_y_continuam_alinhados(self):
        # Linha i: X = [2i, 2i + 1] e Y = [i].
        X_tr, Y_tr, X_va, Y_va, X_te, Y_te, _ = self.separar(50)
        for X_parte, Y_parte in ((X_tr, Y_tr), (X_va, Y_va), (X_te, Y_te)):
            np.testing.assert_array_equal(X_parte[:, 0], 2 * Y_parte[:, 0])

    def test_embaralha(self):
        Y_tr = self.separar(50)[1]
        self.assertFalse(np.array_equal(Y_tr.ravel(), np.arange(len(Y_tr))))

    def test_mesma_seed_mesma_divisao(self):
        a = self.separar(50, seed=5)
        b = self.separar(50, seed=5)
        for parte_a, parte_b in zip(a[:6], b[:6]):
            np.testing.assert_array_equal(parte_a, parte_b)

    def test_validacao_zero_por_cento(self):
        X_tr, Y_tr, X_va, Y_va, X_te, Y_te, _ = self.separar(50, (80, 0, 20))
        self.assertEqual(X_va.shape, (0, 2))
        self.assertEqual(Y_va.shape, (0, 1))
        self.assertEqual([len(X_tr), len(X_te)], [40, 10])

    def test_soma_diferente_de_100_e_erro(self):
        with self.assertRaises(ValueError) as erro:
            self.separar(50, (70, 20, 20))
        self.assertIn("100", str(erro.exception))

    def test_percentual_negativo_e_erro(self):
        with self.assertRaises(ValueError):
            self.separar(50, (110, -5, -5))

    def test_percentual_nan_e_erro_em_portugues(self):
        with self.assertRaises(ValueError) as erro:
            self.separar(50, (float("nan"), float("nan"), float("nan")))
        self.assertIn("números", str(erro.exception))

    def test_treino_zero_e_erro(self):
        with self.assertRaises(ValueError):
            self.separar(50, (0, 50, 50))

    def test_precisa_de_tres_percentuais(self):
        with self.assertRaises(ValueError):
            self.separar(50, (70, 30))

    def test_base_pequena_nao_separa(self):
        # Menos de 10 linhas (XOR, AND): os 3 conjuntos são a base inteira.
        X, Y = self.base(4)
        X_tr, Y_tr, X_va, Y_va, X_te, Y_te, separou = separar_treino_validacao_teste(X, Y, seed=1)
        self.assertFalse(separou)
        for parte in (X_tr, X_va, X_te):
            np.testing.assert_array_equal(parte, X)
        for parte in (Y_tr, Y_va, Y_te):
            np.testing.assert_array_equal(parte, Y)


class TesteTipoDeProblema(unittest.TestCase):

    def test_classes_em_texto_e_classificacao(self):
        self.assertTrue(e_classificacao(np.array([[1.0, 0.0, 0.0]]), ["a", "b", "c"]))

    def test_saida_zero_ou_um_e_classificacao(self):
        self.assertTrue(e_classificacao(np.array([[0.0], [1.0]]), None))

    def test_saida_numerica_e_regressao(self):
        self.assertFalse(e_classificacao(np.array([[2252.4], [1.0]]), None))


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
