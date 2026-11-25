"""Testes do gravador do player (interface/gravador.py, passos.py, formulas.py).

Tudo aqui roda sem pygame.

    python -m unittest discover testes
"""

import unittest
from pathlib import Path

import numpy as np

from experimento import Experimento
from interface.formulas import formula_ligacao, numero
from interface.gravador import gravar, passos_previsao
from interface.passos import estado_em
from rede.rede_neural import RedeNeural
from rede.treinamento import treinar

RAIZ = Path(__file__).resolve().parent.parent
XOR = RAIZ / "bases" / "xor.csv"
IRIS = RAIZ / "Datasets" / "bezdekIris.data"


def xor(**opcoes):
    return Experimento.da_base(XOR, [(3, "sigmoid")], "sigmoid", True, 42, normalizar=False, **opcoes)


def iris():
    return Experimento.da_base(IRIS, [(5, "tanh")], "sigmoid", True, 1, n_saidas=3)


def do_exemplo_em_foco(gravacao, epoca):
    """Passos detalhados (não compactos) do exemplo em foco numa época."""
    return [p for p in gravacao.passos
            if p.epoca == epoca and p.fase not in ("exemplo", "normalizacao")]


def rede_com_pesos(experimento, pesos):
    """Rede com a mesma arquitetura e os pesos dados [(W, b), ...]."""
    ocultas = [(c.n_neuronios, c.nome_ativacao) for c in experimento.rede.camadas[:-1]]
    rede = RedeNeural(experimento.rede.n_entradas, ocultas, experimento.rede.n_saidas,
                      experimento.rede.usar_bias, 0, experimento.rede.ativacao_saida)
    for camada, (W, b) in zip(rede.camadas, pesos):
        for neuronio, w, bias in zip(camada.neuronios, W, b):
            neuronio.pesos = w.copy()
            neuronio.bias = float(bias)
    return rede


class TesteTreinoGravado(unittest.TestCase):

    def test_treino_gravado_igual_ao_treino_direto(self):
        exp = xor()
        gravacao = gravar(exp, 20, 0.5, [1, 2, "fim"])
        direta = RedeNeural(2, [(3, "sigmoid")], 1, True, 42)
        historico = treinar(direta, exp.X_treino, exp.Y_treino, 20, 0.5, seed=42)
        for gravada, camada in zip(exp.rede.camadas, direta.camadas):
            np.testing.assert_array_equal(gravada.W, camada.W)
            np.testing.assert_array_equal(gravada.b, camada.b)
        self.assertEqual([r["erro_treino"] for r in gravacao.resumos], historico["treino"])
        self.assertEqual(exp.historico["treino"], historico["treino"])

    def test_validacao_gravada_igual_ao_treino_direto(self):
        exp = iris()
        gravacao = gravar(exp, 3, 0.1, [1])
        direta = RedeNeural(4, [(5, "tanh")], 3, True, 1)
        historico = treinar(direta, exp.X_treino, exp.Y_treino, 3, 0.1,
                            exp.X_validacao, exp.Y_validacao, seed=1)
        self.assertEqual([r["erro_validacao"] for r in gravacao.resumos], historico["validacao"])

    def test_resumo_de_cada_epoca_tem_os_pesos_do_fim_da_epoca(self):
        exp = xor()
        gravacao = gravar(exp, 6, 0.5, [1])
        direta = RedeNeural(2, [(3, "sigmoid")], 1, True, 42)
        treinar(direta, exp.X_treino, exp.Y_treino, 4, 0.5, seed=42)
        self.assertEqual(len(gravacao.resumos), 6)
        resumo = gravacao.resumos[3]
        self.assertEqual(resumo["epoca"], 4)
        # Os 4 primeiros sorteios da ordem são os mesmos nos dois treinos.
        for (W, b), camada in zip(resumo["pesos"], direta.camadas):
            np.testing.assert_array_equal(W, camada.W)
            np.testing.assert_array_equal(b, camada.b)

    def test_classificacao_tem_acuracia_no_resumo(self):
        gravacao = gravar(xor(), 2, 0.5, [1])
        self.assertTrue(0 <= gravacao.resumos[-1]["acuracia"] <= 1)

    def test_fim_vira_a_ultima_epoca(self):
        gravacao = gravar(xor(), 20, 0.5, [1, 2, "fim", 2])
        self.assertEqual(gravacao.epocas_detalhadas, [1, 2, 20])

    def test_exemplo_em_foco_fora_do_treino_da_valueerror(self):
        with self.assertRaises(ValueError):
            gravar(xor(), 1, 0.5, [1], exemplo_em_foco=4)


class TestePassosDoExemploEmFoco(unittest.TestCase):

    def setUp(self):
        self.exp = xor()
        self.gravacao = gravar(self.exp, 3, 0.5, [1, 3], exemplo_em_foco=2)
        self.passos = do_exemplo_em_foco(self.gravacao, 3)

    def test_ordem_das_fases(self):
        fases = []
        for passo in self.passos:
            if not fases or fases[-1] != passo.fase:
                fases.append(passo.fase)
        self.assertEqual(fases, ["forward", "erro", "retropropagacao", "atualizacao"])

    def test_z_a_e_delta_batem_com_a_rede_real(self):
        # Refaz o exemplo numa rede com os pesos de antes dele.
        rede = rede_com_pesos(self.exp, self.passos[0].pesos)
        x, y = self.exp.X_treino[2], self.exp.Y_treino[2]
        rede.forward(x)
        rede.retropropagar(y)
        ativacoes = [p for p in self.passos if p.tipo == "ativacao"]
        deltas = [p for p in self.passos if p.tipo == "delta"]
        self.assertEqual(len(ativacoes), 4)   # 3 ocultos + 1 saída
        self.assertEqual(len(deltas), 4)
        for passo in ativacoes:
            neuronio = rede.camadas[passo.camada - 1].neuronios[passo.neuronio]
            self.assertEqual(passo.valores["z"], neuronio.z)
            self.assertEqual(passo.valores["a"], neuronio.a)
        for passo in deltas:
            neuronio = rede.camadas[passo.camada - 1].neuronios[passo.neuronio]
            self.assertEqual(passo.valores["delta"], neuronio.delta)

    def test_produtos_mais_bias_dao_z(self):
        for passo in [p for p in self.passos if p.tipo == "bias"]:
            produtos = [p.valores["produto"] for p in self.passos
                        if p.tipo == "ligacao" and p.camada == passo.camada
                        and p.neuronio == passo.neuronio]
            self.assertAlmostEqual(sum(produtos) + passo.valores["b"], passo.valores["z"])

    def test_gradiente_e_delta_vezes_a(self):
        gradientes = [p for p in self.passos if p.tipo == "gradiente"]
        self.assertEqual(len(gradientes), 2 * 3 + 3 * 1)   # uma por ligação
        for passo in gradientes:
            self.assertAlmostEqual(passo.valores["gradiente"],
                                   passo.valores["delta"] * passo.valores["a"])

    def test_delta_e_dE_da_vezes_derivada(self):
        for passo in [p for p in self.passos if p.tipo == "delta"]:
            self.assertAlmostEqual(passo.valores["delta"],
                                   passo.valores["dE_da"] * passo.valores["derivada"])

    def test_ajuste_aplica_w_menos_eta_delta_a(self):
        ajustes = [p for p in self.passos if p.tipo == "ajuste"]
        self.assertEqual([p.camada for p in ajustes], [1, 2])
        for passo in ajustes:
            v = passo.valores
            esperado = v["W_antes"] - 0.5 * np.outer(v["deltas"], v["entradas"])
            np.testing.assert_allclose(v["W_depois"], esperado)
            np.testing.assert_allclose(v["b_depois"], v["b_antes"] - 0.5 * v["deltas"])

    def test_pesos_dos_passos_mudam_so_no_ajuste(self):
        antes = self.passos[0].pesos
        ultimo = self.passos[-1].pesos
        for passo in self.passos:
            if passo.fase != "atualizacao":
                self.assertIs(passo.pesos, antes)
        exp = self.exp
        # Depois do último ajuste, os pesos são os de depois do exemplo.
        rede = rede_com_pesos(exp, antes)
        rede.treinar_exemplo(exp.X_treino[2], exp.Y_treino[2], 0.5)
        for (W, b), camada in zip(ultimo, rede.camadas):
            np.testing.assert_array_equal(W, camada.W)
            np.testing.assert_array_equal(b, camada.b)

    def test_cada_passo_tem_formula_e_texto(self):
        for passo in self.gravacao.passos:
            self.assertTrue(passo.formula, passo.tipo)
            self.assertTrue(passo.formula_numeros, passo.tipo)
            self.assertTrue(passo.texto, passo.tipo)

    def test_epoca_detalhada_tem_n_menos_1_passos_compactos(self):
        compactos = [p for p in self.gravacao.passos if p.epoca == 1 and p.tipo == "exemplo_compacto"]
        self.assertEqual(len(compactos), 3)
        self.assertEqual(sorted(p.valores["indice"] for p in compactos), [0, 1, 3])

    def test_epoca_nao_detalhada_nao_tem_passos(self):
        self.assertEqual([p for p in self.gravacao.passos if p.epoca == 2], [])


class TesteNiveis(unittest.TestCase):

    def test_niveis_menores_tem_menos_passos_e_mesma_ordem_de_fases(self):
        def fases(passos):
            ordem = []
            for passo in passos:
                if not ordem or ordem[-1] != passo.fase:
                    ordem.append(passo.fase)
            return ordem

        gravacoes = {nivel: gravar(xor(), 2, 0.5, [2], nivel=nivel)
                     for nivel in ("ligacao", "neuronio", "camada")}
        tamanhos = [len(gravacoes[n].passos) for n in ("ligacao", "neuronio", "camada")]
        self.assertGreater(tamanhos[0], tamanhos[1])
        self.assertGreater(tamanhos[1], tamanhos[2])
        self.assertEqual(fases(gravacoes["ligacao"].passos), fases(gravacoes["camada"].passos))
        self.assertEqual(fases(gravacoes["neuronio"].passos), fases(gravacoes["camada"].passos))

    def test_nivel_desconhecido_da_valueerror(self):
        with self.assertRaises(ValueError):
            gravar(xor(), 1, 0.5, [1], nivel="sinapse")


class TesteNormalizacao(unittest.TestCase):

    def test_fase_de_normalizacao_mostra_o_exemplo_em_foco(self):
        exp = iris()
        gravacao = gravar(exp, 1, 0.1, [1], exemplo_em_foco=5)
        normalizacao = [p for p in gravacao.passos if p.fase == "normalizacao"]
        self.assertEqual(len(normalizacao), 4)
        self.assertIs(gravacao.passos[0], normalizacao[0])
        for j, passo in enumerate(normalizacao):
            self.assertEqual(passo.valores["original"], exp.X_treino_original[5][j])
            self.assertAlmostEqual(passo.valores["normalizado"], exp.X_treino[5][j])

    def test_sem_normalizacao_nao_tem_essa_fase(self):
        gravacao = gravar(xor(), 1, 0.5, [1])
        self.assertFalse([p for p in gravacao.passos if p.fase == "normalizacao"])


class TestePrevisao(unittest.TestCase):

    def test_passos_previsao_nao_mudam_os_pesos(self):
        exp = xor()
        antes = [camada.W for camada in exp.rede.camadas]
        passos = passos_previsao(exp.rede, np.array([1.0, 0.0]))
        for camada, W in zip(exp.rede.camadas, antes):
            np.testing.assert_array_equal(camada.W, W)
        self.assertTrue(all(p.fase == "forward" for p in passos))
        saida = [p for p in passos if p.tipo == "ativacao" and p.camada == 2]
        self.assertEqual(saida[0].valores["a"], exp.rede.prever([1.0, 0.0])[0, 0])


class TesteEstado(unittest.TestCase):

    def test_estado_junta_ativacoes_e_deltas_ate_o_passo(self):
        exp = xor()
        gravacao = gravar(exp, 1, 0.5, [1], exemplo_em_foco=0)
        passos = gravacao.passos
        primeira_da_saida = next(i for i, p in enumerate(passos)
                                 if p.tipo == "ligacao" and p.camada == 2)
        estado = estado_em(passos, primeira_da_saida)
        self.assertEqual(set(estado["ativacoes"]), {(0, 0), (0, 1), (1, 0), (1, 1), (1, 2)})
        self.assertEqual(estado["deltas"], {})
        self.assertEqual(estado["fase"], "forward")

        ultimo = max(i for i, p in enumerate(passos) if p.tipo == "ajuste")
        estado = estado_em(passos, ultimo)
        self.assertEqual(len(estado["deltas"]), 4)
        self.assertIs(estado["pesos"], passos[ultimo].pesos)


class TesteFormulas(unittest.TestCase):

    def test_numero_com_duas_casas_e_negativo_entre_parenteses(self):
        self.assertEqual(numero(0.5), "0.50")
        self.assertEqual(numero(-1.2), "(−1.20)")

    def test_formula_da_ligacao(self):
        generica, com_numeros = formula_ligacao(0, 0.4, 0.8)
        self.assertEqual(generica, "w·a")
        self.assertEqual(com_numeros, "w₁·a₁ = 0.40·0.80 = 0.32")


if __name__ == "__main__":
    unittest.main()
