"""Testes do main.py: roda o programa inteiro com respostas simuladas.

    python -m unittest discover testes
"""

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

import main

PASTA_BASES = Path(__file__).resolve().parent.parent / "bases"
XOR = str(PASTA_BASES / "xor.csv")


def executar(respostas):
    """Roda main.main() respondendo às perguntas com a lista dada.

    Devolve tudo o que o programa imprimiu. Se as respostas acabarem, o
    input() levanta EOFError, que o programa trata como saída normal.
    """
    saida = io.StringIO()
    with patch("builtins.input", side_effect=list(respostas) + [EOFError()]):
        with redirect_stdout(saida):
            main.main()
    return saida.getvalue()


class TesteFluxoCompleto(unittest.TestCase):

    def test_responder_n_encerra(self):
        saida = executar(["n"])
        self.assertIn("Até mais!", saida)
        self.assertNotIn("Rede montada", saida)

    def test_xor_ponta_a_ponta(self):
        saida = executar([
            "s",
            "2",            # entrada
            "1",            # camadas ocultas
            "4", "sigmoid",
            "1",            # saída
            "s",            # bias
            "42",           # semente
            XOR,
            "0.5",          # taxa
            "10000",        # épocas
            "1", "1", "0",  # menu: testar entrada (1, 0)
            "2", "100",     # menu: treinar mais 100 épocas
            "0",            # sair
        ])
        self.assertIn("Oculta 1: 4 neurônios, sigmoid — 2 × 4 = 8 pesos (+4 bias)", saida)
        self.assertIn("Total de pesos: 12", saida)
        self.assertIn("Base pequena", saida)
        self.assertIn("Época  1000/10000 — erro (MSE):", saida)
        self.assertIn("Época 10000/10000 — erro (MSE):", saida)
        self.assertIn("Acurácia: 100,00% (4/4)", saida)
        self.assertIn("classe: 1", saida)          # XOR de (1, 0) = 1
        self.assertIn("Época 100/100", saida)
        self.assertTrue(saida.rstrip().endswith("Até mais!"))

    def test_entradas_invalidas_nao_quebram(self):
        saida = executar([
            "talvez", "s",          # s/n inválido
            "abc", "0", "-3", "2",  # entrada: texto, zero, negativo
            "-1", "1",              # ocultas negativo
            "", "2.5", "3",         # neurônios: vazio, decimal
            "linear", "TANH",       # ativação inexistente; maiúscula é aceita
            "1",
            "x", "n",               # bias inválido
            "abc", "-5", "",        # semente inválida, negativa, Enter
            "nao_existe.csv",       # arquivo inexistente
            XOR,
            "0", "-1", "abc", "0,5",  # taxa: zero, negativa, texto, vírgula decimal
            "0", "abc", "50",       # épocas
            "9", "",                # opção de menu inválida
            "1", "abc", "1", "0",   # entrada digitada inválida
            "0",
        ])
        self.assertNotIn("Traceback", saida)
        self.assertIn("Não foi possível carregar a base: Arquivo não encontrado", saida)
        self.assertIn("Oculta 1: 3 neurônios, tanh", saida)
        self.assertIn("Época 50/50", saida)
        self.assertTrue(saida.rstrip().endswith("Até mais!"))

    def test_base_incompativel_e_cancelar(self):
        # Rede com 3 entradas não combina com o XOR (2 entradas).
        saida = executar([
            "s", "3", "0", "1", "n", "1",
            XOR,   # incompatível
            "",    # cancela → volta para montar outra rede
            "2", "0", "1", "s", "1",
            XOR, "", "",  # Enter = taxa 0.5 e 5000 épocas
            "0",
        ])
        self.assertIn("Base incompatível", saida)
        self.assertIn("precisa de 4 colunas, mas tem 3", saida)
        self.assertIn("--- Nova rede ---", saida)
        self.assertIn("Época 5000/5000", saida)

    def test_criar_outra_rede_pelo_menu(self):
        saida = executar([
            "s", "2", "0", "1", "s", "1", XOR, "", "10",
            "3",
            "2", "0", "1", "s", "1", XOR, "", "10",
            "0",
        ])
        self.assertEqual(saida.count("Rede montada:"), 2)

    def test_fim_da_entrada_no_meio_sai_sem_erro(self):
        saida = executar(["s", "2"])
        self.assertTrue(saida.rstrip().endswith("Até mais!"))


class TesteSeletorDeBase(unittest.TestCase):
    """Menu de bases: escolher pelo número da lista."""

    def test_lista_as_bases_do_projeto(self):
        saida = executar(["s", "2", "0", "1", "s", "1", ""])
        self.assertIn("1) and.csv  (bases/)", saida)
        self.assertIn("2) xor.csv  (bases/)", saida)
        self.assertIn("3) bezdekIris.data  (Datasets/)", saida)
        self.assertIn("4) dataset.csv  (Datasets/)", saida)
        self.assertIn("5) iris.data  (Datasets/)", saida)

    def test_escolher_xor_pelo_numero(self):
        saida = executar([
            "s", "2", "1", "4", "sigmoid", "1", "s", "42",
            "9",   # número fora da lista
            "2",   # xor.csv
            "0.5", "10000", "0",
        ])
        self.assertIn("Opção inválida: escolha um número de 1 a 5", saida)
        self.assertIn("Acurácia: 100,00% (4/4)", saida)

    def test_iris_ponta_a_ponta(self):
        saida = executar([
            "s", "4", "1", "5", "sigmoid", "3", "s", "1",
            "3",              # bezdekIris.data
            "",               # lote: Enter = base inteira
            "0.5", "5000",
            "1", "5.1", "3.5", "1.4", "0.2",   # medidas de uma setosa
            "1", "6.3", "3.3", "6.0", "2.5",   # medidas de uma virginica
            "0",
        ])
        self.assertIn("maior ŷ na classificação", saida)
        self.assertIn("Base carregada: 150 amostras.", saida)
        self.assertIn("Classes (uma por neurônio de saída): "
                      "Iris-setosa, Iris-versicolor, Iris-virginica", saida)
        self.assertIn("Base separada: 120 amostras para treino e 30 para teste.", saida)
        self.assertIn("classe: Iris-setosa", saida)
        self.assertIn("classe: Iris-virginica", saida)
        acuracia = float(saida.split("Acurácia: ")[1].split("%")[0].replace(",", "."))
        self.assertGreaterEqual(acuracia, 90)

    def test_iris_incompativel_conta_as_colunas_do_arquivo(self):
        # Rede 3 → 3: o arquivo da Iris tem 5 colunas (4 medidas + classe em
        # texto). A classe é 1 coluna no arquivo, mesmo virando 3 saídas.
        saida = executar(["s", "3", "0", "3", "s", "1", "3", ""])
        self.assertIn("Base incompatível", saida)
        self.assertIn("precisa de 4 colunas, mas tem 5", saida)

    def test_sem_bases_na_lista_numero_pede_caminho(self):
        with tempfile.TemporaryDirectory() as vazia:
            with patch.object(main, "PASTAS_DE_BASES", [Path(vazia)]):
                saida = executar(["s", "2", "0", "1", "s", "1", "1", ""])
        self.assertNotIn("de 1 a 0", saida)
        self.assertIn("Nenhuma base na lista: digite o caminho de um arquivo.", saida)

    def test_iris_com_numero_errado_de_saidas(self):
        saida = executar(["s", "4", "0", "1", "s", "1", "3", ""])
        self.assertIn("3 classe(s) (Iris-setosa, Iris-versicolor, Iris-virginica)", saida)
        self.assertIn("precisa de 3 neurônio(s) de saída", saida)


class TesteBaseGrande(unittest.TestCase):
    """Base com 10+ linhas: separa treino/teste, normaliza e limita a tabela."""

    def test_separa_normaliza_e_limita_tabela(self):
        # 200 amostras: classe 1 quando a entrada (entre 0 e 1000) passa de 500.
        gerador = np.random.default_rng(0)
        x = gerador.uniform(0, 1000, size=200)
        linhas = ["valor,classe"] + [f"{v:.2f},{int(v > 500)}" for v in x]

        with tempfile.TemporaryDirectory() as pasta:
            caminho = Path(pasta) / "limiar.csv"
            caminho.write_text("\n".join(linhas), encoding="utf-8")
            saida = executar([
                "s", "1", "0", "1", "s", "1",
                str(caminho),
                "",           # lote: Enter = base inteira
                "0.5", "5000",
                "1", "900",   # bem acima do limiar → classe 1
                "1", "100",   # bem abaixo → classe 0
                "0",
            ])

        self.assertIn("Base carregada: 200 amostras.", saida)
        self.assertIn("Base separada: 160 amostras para treino e 40 para teste.", saida)
        self.assertIn("(mostrando 20 de 40 amostras)", saida)
        self.assertIn("classe: 1", saida)
        self.assertIn("classe: 0", saida)
        # Sem normalização, valores na casa de 1000 saturariam a sigmoid.
        acuracia = float(saida.split("Acurácia: ")[1].split("%")[0].replace(",", "."))
        self.assertGreater(acuracia, 90)


class TesteNormalizacaoNovas(unittest.TestCase):
    """Saída numérica normalizada e normalização por lotes."""

    def test_saida_numerica_e_normalizada_e_mostrada_na_escala_original(self):
        # Regressão: y = 100 + 50·x, com x entre 0 e 1 → y entre 100 e 150.
        # Sem normalizar Y, a sigmoid (0 a 1) nunca chegaria nesses valores.
        gerador = np.random.default_rng(0)
        x = gerador.uniform(0, 1, size=60)
        linhas = ["x,y"] + [f"{v:.4f},{100 + 50 * v:.4f}" for v in x]

        with tempfile.TemporaryDirectory() as pasta:
            caminho = Path(pasta) / "reta.csv"
            caminho.write_text("\n".join(linhas), encoding="utf-8")
            saida = executar([
                "s", "1", "1", "4", "sigmoid", "1", "s", "1",
                str(caminho), "", "0.5", "5000",
                "1", "0.5",   # esperado ≈ 125
                "0",
            ])

        y_hat = float(saida.split("ŷ = ")[1].split()[0])
        self.assertAlmostEqual(y_hat, 125, delta=3)
        erro = float(saida.split("Erro final (MSE): ")[1].split()[0])
        self.assertLess(erro, 0.01)   # MSE na escala normalizada

    def test_iris_com_normalizacao_por_lotes(self):
        saida = executar([
            "s", "4", "1", "5", "sigmoid", "3", "s", "1",
            "3",
            "1", "121", "abc",  # lote inválido: abaixo de 2, acima de 120, texto
            "30",               # lotes de 30 registros
            "0.5", "5000",
            "0",
        ])
        self.assertIn("o mínimo é 2", saida)
        self.assertIn("o máximo é 120", saida)
        self.assertIn("normalizadas em 4 lote(s) de ~30 registros", saida)
        acuracia = float(saida.split("Acurácia: ")[1].split("%")[0].replace(",", "."))
        self.assertGreaterEqual(acuracia, 80)


if __name__ == "__main__":
    unittest.main()
