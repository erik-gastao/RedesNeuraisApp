"""Testes do main.py: roda o programa inteiro com respostas simuladas.

    python -m unittest discover testes

Ordem das perguntas (rede nova): criar? → entrada → ocultas (neurônios e
ativação de cada) → saída → ativação da saída → bias → semente → base →
divisão em % (só bases com 10+ linhas) → normalizar entradas? → lote (só
se normalizou e dividiu) → taxa → épocas → menu.
"""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

import main

RAIZ = Path(__file__).resolve().parent.parent
XOR = str(RAIZ / "bases" / "xor.csv")
DATASET_PROFESSOR = str(RAIZ / "Datasets" / "dataset.csv")


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


def numero_depois(saida, marcador):
    """Primeiro número que aparece depois do marcador (vírgula ou ponto)."""
    return float(saida.split(marcador)[1].split()[0].replace(",", ".").rstrip("%"))


def base_limiar(pasta):
    """200 amostras: classe 1 quando a entrada (entre 0 e 1000) passa de 500."""
    gerador = np.random.default_rng(0)
    x = gerador.uniform(0, 1000, size=200)
    caminho = Path(pasta) / "limiar.csv"
    caminho.write_text(
        "\n".join(["valor,classe"] + [f"{v:.2f},{int(v > 500)}" for v in x]), encoding="utf-8"
    )
    return str(caminho)


def base_reta(pasta):
    """Regressão: y = 100 + 50·x, com x entre 0 e 1 → y entre 100 e 150."""
    gerador = np.random.default_rng(0)
    x = gerador.uniform(0, 1, size=60)
    caminho = Path(pasta) / "reta.csv"
    caminho.write_text(
        "\n".join(["x,y"] + [f"{v:.4f},{100 + 50 * v:.4f}" for v in x]), encoding="utf-8"
    )
    return str(caminho)


class TesteFluxoCompleto(unittest.TestCase):

    def test_responder_n_n_encerra(self):
        saida = executar(["n", "n"])
        self.assertIn("Até mais!", saida)
        self.assertNotIn("Rede montada", saida)

    def test_xor_ponta_a_ponta(self):
        saida = executar([
            "s",
            "2",            # entrada
            "1",            # camadas ocultas
            "4", "sigmoid",
            "1",            # saída
            "",             # ativação da saída: Enter = sigmoid
            "s",            # bias
            "42",           # semente
            XOR,
            "n",            # não normalizar
            "0.5",          # taxa
            "3000",         # épocas
            "1", "1", "0",  # menu: testar entrada (1, 0)
            "2", "100",     # menu: treinar mais 100 épocas
            "0",            # sair
        ])
        self.assertIn("Oculta 1: 4 neurônios, sigmoid — 2 × 4 = 8 pesos (+4 bias)", saida)
        self.assertIn("Total de pesos: 12", saida)
        self.assertIn("Base pequena", saida)
        self.assertIn("Época  300/3000 — erro treino (MSE):", saida)
        self.assertIn("Época 3000/3000 — erro treino (MSE):", saida)
        self.assertNotIn("| validação", saida)   # base pequena: sem validação
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
            "degrau", "TANH",       # ativação inexistente; maiúscula é aceita
            "1",
            "xyz", "",              # ativação da saída inválida, Enter
            "x", "n",               # bias inválido
            "abc", "-5", "",        # semente inválida, negativa, Enter
            "nao_existe.csv",       # arquivo inexistente
            XOR,
            "talvez", "n",          # normalizar: inválido
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
            "s", "3", "0", "1", "", "n", "1",
            XOR,   # incompatível
            "",    # cancela → volta para montar outra rede
            "2", "0", "1", "", "s", "1",
            XOR, "n", "", "",  # Enter = taxa 0.1 e 1000 épocas
            "0",
        ])
        self.assertIn("Base incompatível", saida)
        self.assertIn("precisa de 4 colunas, mas tem 3", saida)
        self.assertIn("--- Nova rede ---", saida)
        self.assertIn("Época 1000/1000", saida)

    def test_criar_outra_rede_pelo_menu(self):
        saida = executar([
            "s", "2", "0", "1", "", "s", "1", XOR, "n", "", "10",
            "3",
            "2", "0", "1", "", "s", "1", XOR, "n", "", "10",
            "0",
        ])
        self.assertEqual(saida.count("Rede montada:"), 2)

    def test_fim_da_entrada_no_meio_sai_sem_erro(self):
        saida = executar(["s", "2"])
        self.assertTrue(saida.rstrip().endswith("Até mais!"))


class TesteSeletorDeBase(unittest.TestCase):
    """Menu de bases: escolher pelo número da lista."""

    def test_lista_as_bases_do_projeto(self):
        saida = executar(["s", "2", "0", "1", "", "s", "1", ""])
        self.assertIn("1) and.csv  (bases/)", saida)
        self.assertIn("2) xor.csv  (bases/)", saida)
        self.assertIn("3) bezdekIris.data  (Datasets/)", saida)
        self.assertIn("4) dataset.csv  (Datasets/)", saida)
        self.assertIn("5) iris.data  (Datasets/)", saida)

    def test_escolher_xor_pelo_numero(self):
        saida = executar([
            "s", "2", "1", "4", "sigmoid", "1", "", "s", "42",
            "9",   # número fora da lista
            "2",   # xor.csv
            "n", "0.5", "3000", "0",
        ])
        self.assertIn("Opção inválida: escolha um número de 1 a 5", saida)
        self.assertIn("Acurácia: 100,00% (4/4)", saida)

    def test_iris_ponta_a_ponta(self):
        saida = executar([
            "s", "4", "1", "5", "sigmoid", "3", "", "s", "1",
            "3",              # bezdekIris.data
            "",               # divisão: Enter = 70/15/15
            "s",              # normalizar entradas
            "",               # lote: Enter = base inteira
            "0.1", "200",
            "1", "5.1", "3.5", "1.4", "0.2",   # medidas de uma setosa
            "1", "6.3", "3.3", "6.0", "2.5",   # medidas de uma virginica
            "0",
        ])
        self.assertIn("maior ŷ na classificação", saida)
        self.assertIn("Base carregada: 150 amostras.", saida)
        self.assertIn("Classes (uma por neurônio de saída): "
                      "Iris-setosa, Iris-versicolor, Iris-virginica", saida)
        # 15% de 150 = 22,5 → 22 (arredondamento do Python).
        self.assertIn("Base separada: 106 para treino, 22 para validação e 22 para teste.", saida)
        self.assertIn("| validação:", saida)
        self.assertIn("MSE treino:", saida)
        self.assertIn("classe: Iris-setosa", saida)
        self.assertIn("classe: Iris-virginica", saida)
        self.assertGreaterEqual(numero_depois(saida, "Acurácia: "), 90)

    def test_iris_incompativel_conta_as_colunas_do_arquivo(self):
        # Rede 3 → 3: o arquivo da Iris tem 5 colunas (4 medidas + classe em
        # texto). A classe é 1 coluna no arquivo, mesmo virando 3 saídas.
        saida = executar(["s", "3", "0", "3", "", "s", "1", "3", ""])
        self.assertIn("Base incompatível", saida)
        self.assertIn("precisa de 4 colunas, mas tem 5", saida)

    def test_sem_bases_na_lista_numero_pede_caminho(self):
        with tempfile.TemporaryDirectory() as vazia:
            with patch.object(main, "PASTAS_DE_BASES", [Path(vazia)]):
                saida = executar(["s", "2", "0", "1", "", "s", "1", "1", ""])
        self.assertNotIn("de 1 a 0", saida)
        self.assertIn("Nenhuma base na lista: digite o caminho de um arquivo.", saida)

    def test_iris_com_numero_errado_de_saidas(self):
        saida = executar(["s", "4", "0", "1", "", "s", "1", "3", ""])
        self.assertIn("3 classe(s) (Iris-setosa, Iris-versicolor, Iris-virginica)", saida)
        self.assertIn("precisa de 3 neurônio(s) de saída", saida)


class TesteBaseGrande(unittest.TestCase):
    """Base com 10+ linhas: divide em 3 conjuntos, normaliza e limita a tabela."""

    def test_divide_normaliza_e_limita_tabela(self):
        with tempfile.TemporaryDirectory() as pasta:
            saida = executar([
                "s", "1", "0", "1", "", "s", "1",
                base_limiar(pasta),
                "",           # divisão: Enter = 70/15/15
                "s", "",      # normalizar; lote = base inteira
                "0.5", "200",
                "1", "900",   # bem acima do limiar → classe 1
                "1", "100",   # bem abaixo → classe 0
                "0",
            ])
        self.assertIn("Base carregada: 200 amostras.", saida)
        self.assertIn("Base separada: 140 para treino, 30 para validação e 30 para teste.", saida)
        self.assertIn("(mostrando 20 de 30 amostras)", saida)
        self.assertIn("classe: 1", saida)
        self.assertIn("classe: 0", saida)
        self.assertGreater(numero_depois(saida, "Acurácia: "), 90)

    def test_percentuais_invalidos_reperguntam(self):
        with tempfile.TemporaryDirectory() as pasta:
            saida = executar([
                "s", "1", "0", "1", "", "s", "1",
                base_limiar(pasta),
                "70/30", "50/50/50", "-10/60/50", "abc", "0/50/50",
                "60 20 20",
                "n", "0.5", "5", "0",
            ])
        self.assertGreaterEqual(saida.count("Valor inválido"), 5)
        self.assertIn("Base separada: 120 para treino, 40 para validação e 40 para teste.", saida)
        self.assertNotIn("Traceback", saida)

    def test_divisao_aceita_virgula_decimal(self):
        with tempfile.TemporaryDirectory() as pasta:
            saida = executar([
                "s", "1", "0", "1", "", "s", "1", base_limiar(pasta),
                "70,5/14,5/15",   # com "/" a vírgula é decimal
                "n", "0.5", "1", "0",
            ])
        self.assertIn("Base separada: 141 para treino, 29 para validação e 30 para teste.", saida)

    def test_divisao_so_com_virgulas_separa(self):
        with tempfile.TemporaryDirectory() as pasta:
            saida = executar([
                "s", "1", "0", "1", "", "s", "1", base_limiar(pasta),
                "70,15,15", "n", "0.5", "1", "0",
            ])
        self.assertIn("Base separada: 140 para treino, 30 para validação e 30 para teste.", saida)

    def test_validacao_zero_por_cento(self):
        with tempfile.TemporaryDirectory() as pasta:
            saida = executar([
                "s", "1", "0", "1", "", "s", "1", base_limiar(pasta),
                "80/0/20", "s", "", "0.5", "20",
                "2", "5",   # treinar mais também funciona sem validação
                "0",
            ])
        self.assertIn("Base separada: 160 para treino, 0 para validação e 40 para teste.", saida)
        self.assertNotIn("| validação", saida)
        self.assertIn("MSE treino:", saida)
        self.assertNotIn("Traceback", saida)

    def test_iris_com_normalizacao_por_lotes(self):
        saida = executar([
            "s", "4", "1", "5", "sigmoid", "3", "", "s", "1",
            "3", "",
            "s",                # normalizar
            "1", "107", "abc",  # lote inválido: abaixo de 2, acima de 106, texto
            "30",               # lotes de 30 registros
            "0.1", "200",
            "0",
        ])
        self.assertIn("o mínimo é 2", saida)
        self.assertIn("o máximo é 106", saida)
        self.assertIn("normalizadas em 4 lote(s) de ~30 registros", saida)
        self.assertIn("foge do padrão do enunciado", saida)
        self.assertGreaterEqual(numero_depois(saida, "Acurácia: "), 80)


class TesteRegressao(unittest.TestCase):
    """Saída numérica: não é normalizada; a saída linear alcança os valores."""

    def test_regressao_linear_mostra_escala_original(self):
        with tempfile.TemporaryDirectory() as pasta:
            saida = executar([
                "s", "1", "0", "1", "linear", "s", "1",
                base_reta(pasta), "", "s", "", "0.1", "300",
                "1", "0.5",   # esperado ≈ 125
                "0",
            ])
        self.assertAlmostEqual(numero_depois(saida, "ŷ = "), 125, delta=3)
        self.assertNotIn("Acurácia", saida)
        self.assertIn("| erro", saida)          # coluna de erro na tabela
        self.assertNotIn("classe:", saida)
        self.assertLess(numero_depois(saida, "teste: "), 5)

    def test_regressao_com_sigmoid_avisa(self):
        with tempfile.TemporaryDirectory() as pasta:
            saida = executar(["s", "1", "0", "1", "", "s", "1", base_reta(pasta)])
        self.assertIn("a saída esperada vai de 100.137 a 149.861 e não é normalizada", saida)
        self.assertIn("só produz valores entre 0 e 1. Use a ativação linear na saída.", saida)

    def test_saida_entre_0_e_1_com_sigmoid_nao_avisa(self):
        # Regressão com y entre 0,2 e 0,8: a sigmoid alcança, então sem aviso.
        with tempfile.TemporaryDirectory() as pasta:
            caminho = Path(pasta) / "dentro.csv"
            caminho.write_text("x,y\n" + "\n".join(
                f"{i / 19:.4f},{0.2 + 0.6 * i / 19:.4f}" for i in range(20)), encoding="utf-8")
            saida = executar(["s", "1", "0", "1", "", "s", "1", str(caminho)])
        self.assertNotIn("Atenção", saida)

    def test_relu_com_saida_negativa_avisa(self):
        with tempfile.TemporaryDirectory() as pasta:
            caminho = Path(pasta) / "negativa.csv"
            caminho.write_text("x,y\n" + "\n".join(
                f"{i / 19:.4f},{-5 + i / 19:.4f}" for i in range(20)), encoding="utf-8")
            saida = executar(["s", "1", "0", "1", "relu", "s", "1", str(caminho)])
        self.assertIn("a ativação relu só produz valores maiores ou iguais a 0", saida)

    def test_tanh_com_saida_entre_menos_1_e_1_nao_avisa(self):
        with tempfile.TemporaryDirectory() as pasta:
            caminho = Path(pasta) / "simetrica.csv"
            caminho.write_text("x,y\n" + "\n".join(
                f"{i / 19:.4f},{-0.9 + 1.8 * i / 19:.4f}" for i in range(20)), encoding="utf-8")
            saida = executar(["s", "1", "0", "1", "tanh", "s", "1", str(caminho)])
        self.assertNotIn("Atenção", saida)

    def test_dataset_do_professor(self):
        Y = np.genfromtxt(DATASET_PROFESSOR, delimiter=",", skip_header=1)[:, -1]
        saida = executar([
            "s", "2", "1", "8", "relu", "1", "linear", "s", "1",
            DATASET_PROFESSOR, "", "s", "",
            "0.00001", "300",
            "0",
        ])
        self.assertIn("Base separada: 125 para treino, 27 para validação e 27 para teste.", saida)
        self.assertIn("| validação:", saida)
        self.assertNotIn("Acurácia", saida)
        # Melhor que prever sempre a média (MSE = variância).
        self.assertLess(numero_depois(saida, "teste: "), 0.8 * np.var(Y))


class TesteSalvarCarregar(unittest.TestCase):
    """Menu 5 salva a rede; no início, 'n' e depois 's' carrega uma rede salva."""

    def test_salvar_carregar_e_prever_igual(self):
        with tempfile.TemporaryDirectory() as pasta:
            arquivo = str(Path(pasta) / "xor.json")
            primeira = executar([
                "s", "2", "1", "4", "sigmoid", "1", "", "s", "42", XOR, "n", "0.5", "300",
                "1", "1", "0",
                "5", arquivo,
                "0",
            ])
            segunda = executar(["n", "s", arquivo, "1", "1", "0", "0"])
        self.assertIn("Rede salva em:", primeira)
        self.assertIn("Rede carregada:", segunda)
        self.assertEqual(numero_depois(primeira, "ŷ = "), numero_depois(segunda, "ŷ = "))

    def test_rede_carregada_aplica_a_normalizacao_salva(self):
        with tempfile.TemporaryDirectory() as pasta:
            arquivo = str(Path(pasta) / "limiar.json")
            primeira = executar([
                "s", "1", "0", "1", "", "s", "1", base_limiar(pasta), "", "s", "",
                "0.5", "50",
                "1", "900",
                "5", arquivo,
                "0",
            ])
            dados = json.loads(Path(arquivo).read_text(encoding="utf-8"))
            segunda = executar(["n", "s", arquivo, "1", "900", "0"])
        self.assertEqual(dados["normalizacao"]["funcao"], "min-max")
        self.assertEqual(numero_depois(primeira, "ŷ = "), numero_depois(segunda, "ŷ = "))
        self.assertIn("classe: 1", segunda)

    def test_salvar_com_enter_usa_pasta_redes(self):
        with tempfile.TemporaryDirectory() as pasta:
            with patch.object(main, "PASTA_DE_REDES", Path(pasta)):
                saida = executar([
                    "s", "2", "0", "1", "", "s", "1", XOR, "n", "", "5", "5", "", "0",
                ])
            self.assertTrue((Path(pasta) / "xor.json").exists())
        self.assertIn("Rede salva em:", saida)

    def test_cancelar_carregamento_volta_ao_inicio(self):
        saida = executar([
            "n", "s", "",          # quer carregar, mas cancela com Enter
            "s", "2", "0", "1", "", "s", "1", XOR, "n", "", "5", "0",   # volta e cria
        ])
        self.assertIn("Rede montada:", saida)
        self.assertTrue(saida.rstrip().endswith("Até mais!"))

    def test_carregar_arquivo_invalido_nao_quebra(self):
        with tempfile.TemporaryDirectory() as pasta:
            corrompido = Path(pasta) / "ruim.json"
            corrompido.write_text("{ isso não é json", encoding="utf-8")
            saida = executar(["n", "s", str(Path(pasta) / "nao_existe.json"), str(corrompido), ""])
        self.assertEqual(saida.count("Não foi possível carregar a rede"), 2)
        self.assertNotIn("Traceback", saida)
        self.assertTrue(saida.rstrip().endswith("Até mais!"))


if __name__ == "__main__":
    unittest.main()
