"""Testes do projeto. Rodar na pasta raiz com:

    python -m unittest discover testes
"""

import unittest
from pathlib import Path

import numpy as np

from dados import (
    acuracia,
    aplicar_normalizacao,
    carregar_csv,
    normalizar,
    separar_treino_teste,
)
from rede.ativacoes import (
    ATIVACOES,
    classificar,
    degrau,
    derivada_relu,
    derivada_sigmoid,
    derivada_linear,
    derivada_tanh,
    linear,
    maior_saida,
    relu,
    sigmoid,
    tanh,
)
from rede.inicializacao import pesos_iniciais
from rede.camada import Camada
from rede.neuronio import Neuronio
from rede.perda import derivada_mse, mse
from rede.rede_neural import RedeNeural
from rede.treinamento import treinar

# Pasta bases/ na raiz do projeto (este arquivo fica em testes/).
PASTA_BASES = Path(__file__).resolve().parent.parent / "bases"
PASTA_DATASET = Path(__file__).resolve().parent.parent / "Datasets"


class TesteAtivacoes(unittest.TestCase):
    """Testes 1 do plano: funções de ativação e derivadas."""

    def test_sigmoid_em_zero(self):
        self.assertEqual(sigmoid(np.array([0.0]))[0], 0.5)

    def test_derivada_sigmoid_em_zero(self):
        z = np.array([0.0])
        a = sigmoid(z)
        self.assertEqual(derivada_sigmoid(z, a)[0], 0.25)

    def test_sigmoid_nao_estoura_com_valores_extremos(self):
        # Sem o np.clip, e^(1000) daria overflow.
        resultado = sigmoid(np.array([-1000.0, 1000.0]))
        self.assertAlmostEqual(resultado[0], 0.0)
        self.assertAlmostEqual(resultado[1], 1.0)

    def test_relu(self):
        resultado = relu(np.array([-2.0, 3.0]))
        self.assertEqual(resultado[0], 0.0)
        self.assertEqual(resultado[1], 3.0)

    def test_derivada_relu(self):
        z = np.array([-2.0, 0.0, 3.0])
        np.testing.assert_array_equal(derivada_relu(z, relu(z)), [0.0, 0.0, 1.0])

    def test_tanh_em_zero(self):
        self.assertEqual(tanh(np.array([0.0]))[0], 0.0)

    def test_derivada_tanh_em_zero(self):
        z = np.array([0.0])
        self.assertEqual(derivada_tanh(z, tanh(z))[0], 1.0)

    def test_degrau(self):
        np.testing.assert_array_equal(degrau(np.array([-0.1, 0.0, 0.1])), [0.0, 1.0, 1.0])

    def test_classificar_usa_limiar_meio(self):
        y_hat = np.array([0.49, 0.5, 0.51])
        np.testing.assert_array_equal(classificar(y_hat), [0.0, 1.0, 1.0])

    def test_maior_saida(self):
        y_hat = np.array([[0.2, 0.7, 0.1], [0.4, 0.3, 0.1], [0.9, 0.8, 0.95]])
        # 2ª linha: nenhum passa de 0,5; 3ª: todos passam. Sempre sai 1 classe.
        np.testing.assert_array_equal(
            maior_saida(y_hat), [[0, 1, 0], [1, 0, 0], [0, 0, 1]]
        )

    def test_ativacoes_preservam_a_forma(self):
        z = np.zeros((4, 3))
        for nome, (funcao, derivada) in ATIVACOES.items():
            a = funcao(z)
            self.assertEqual(a.shape, (4, 3), nome)
            self.assertEqual(derivada(z, a).shape, (4, 3), nome)

    def test_derivadas_batem_com_diferenca_numerica(self):
        # f'(z) ≈ (f(z + ε) − f(z − ε)) / (2ε)
        z = np.array([-1.5, -0.3, 0.4, 2.0])  # sem o 0, onde a ReLU não tem derivada
        eps = 1e-6
        for nome, (funcao, derivada) in ATIVACOES.items():
            numerica = (funcao(z + eps) - funcao(z - eps)) / (2 * eps)
            np.testing.assert_allclose(derivada(z, funcao(z)), numerica, atol=1e-6, err_msg=nome)

    def test_linear_e_identidade(self):
        z = np.array([-2.0, 0.0, 3.0])
        np.testing.assert_array_equal(linear(z), z)
        np.testing.assert_array_equal(derivada_linear(z, linear(z)), [1.0, 1.0, 1.0])

    def test_dicionario_tem_as_quatro_ativacoes(self):
        self.assertEqual(set(ATIVACOES), {"sigmoid", "relu", "tanh", "linear"})


class TesteInicializacao(unittest.TestCase):
    """Pesos iniciais: Xavier/Glorot (sigmoid, tanh, linear) e He (ReLU)."""

    def test_xavier_dentro_do_limite(self):
        pesos = pesos_iniciais(4, 5, "tanh", np.random.default_rng(0))
        self.assertEqual(pesos.shape, (5, 4))
        self.assertTrue(np.all(np.abs(pesos) <= np.sqrt(6 / 9)))

    def test_he_dentro_do_limite_e_maior_que_xavier(self):
        # Com 1 entrada e 1000 neurônios: Xavier limita em √(6/1001) ≈ 0,077
        # e He em √6 ≈ 2,45. A ReLU precisa sortear valores bem maiores.
        pesos = pesos_iniciais(1, 1000, "relu", np.random.default_rng(0))
        self.assertTrue(np.all(np.abs(pesos) <= np.sqrt(6)))
        self.assertTrue(np.any(np.abs(pesos) > np.sqrt(6 / 1001)))

    def test_sigmoid_e_linear_usam_xavier(self):
        for nome in ("sigmoid", "linear"):
            pesos = pesos_iniciais(1, 1000, nome, np.random.default_rng(0))
            self.assertTrue(np.all(np.abs(pesos) <= np.sqrt(6 / 1001)), nome)

    def test_mesma_seed_mesmos_pesos(self):
        a = pesos_iniciais(3, 2, "relu", np.random.default_rng(7))
        b = pesos_iniciais(3, 2, "relu", np.random.default_rng(7))
        np.testing.assert_array_equal(a, b)


class TesteFormas(unittest.TestCase):
    """Testes 2 do plano: formas das matrizes e do forward."""

    def setUp(self):
        self.rede = RedeNeural(2, [(3, "sigmoid")], 1, usar_bias=True, seed=1)

    def test_formas_das_matrizes_de_pesos(self):
        oculta, saida = self.rede.camadas
        self.assertEqual(oculta.W.shape, (3, 2))
        self.assertEqual(saida.W.shape, (1, 3))
        self.assertEqual(oculta.b.shape, (3,))
        self.assertEqual(saida.b.shape, (1,))

    def test_forma_da_saida_do_forward(self):
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.assertEqual(self.rede.forward(X).shape, (4, 1))

    def test_forward_aceita_uma_amostra_so(self):
        self.assertEqual(self.rede.forward([0, 1]).shape, (1, 1))

    def test_forward_recusa_numero_errado_de_entradas(self):
        with self.assertRaises(ValueError):
            self.rede.forward(np.zeros((4, 3)))

    def test_camada_guarda_valores_do_forward(self):
        # Cada neurônio guarda entradas, z e a da última amostra processada.
        X = np.array([[0.0, 1.0], [1.0, 1.0]])
        self.rede.forward(X)
        oculta, saida = self.rede.camadas
        for neuronio in oculta.neuronios:
            np.testing.assert_array_equal(neuronio.entradas, X[1])
            self.assertAlmostEqual(neuronio.a, float(sigmoid(np.array(neuronio.z))))
        # A entrada da saída é a saída da oculta.
        np.testing.assert_array_equal(
            saida.neuronios[0].entradas, [n.a for n in oculta.neuronios]
        )

    def test_vieses_comecam_em_zero(self):
        for camada in self.rede.camadas:
            np.testing.assert_array_equal(camada.b, 0)

    def test_mesma_seed_gera_mesmos_pesos(self):
        outra = RedeNeural(2, [(3, "sigmoid")], 1, usar_bias=True, seed=1)
        for a, b in zip(self.rede.camadas, outra.camadas):
            np.testing.assert_array_equal(a.W, b.W)

    def test_sem_camada_oculta(self):
        rede = RedeNeural(2, [], 1, usar_bias=False, seed=1)
        self.assertEqual(len(rede.camadas), 1)
        self.assertEqual(rede.camadas[0].W.shape, (1, 2))
        self.assertEqual(rede.forward(np.zeros((4, 2))).shape, (4, 1))

    def test_varias_ocultas_e_varias_saidas(self):
        rede = RedeNeural(4, [(5, "relu"), (3, "tanh")], 2, usar_bias=True, seed=1)
        formas = [camada.W.shape for camada in rede.camadas]
        self.assertEqual(formas, [(5, 4), (3, 5), (2, 3)])
        self.assertEqual(rede.forward(np.zeros((7, 4))).shape, (7, 2))

    def test_ativacao_desconhecida_da_erro(self):
        with self.assertRaises(ValueError):
            RedeNeural(2, [(3, "degrau")], 1, usar_bias=True, seed=1)

    def test_classificar_devolve_so_zero_ou_um(self):
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        classes = self.rede.classificar(X)
        self.assertEqual(classes.shape, (4, 1))
        self.assertTrue(np.all((classes == 0) | (classes == 1)))

    def test_varias_saidas_classifica_pelo_maior(self):
        # Com mais de 1 saída, exatamente uma classe por amostra.
        rede = RedeNeural(4, [(5, "sigmoid")], 3, usar_bias=True, seed=1)
        X = np.random.default_rng(0).uniform(0, 1, size=(20, 4))
        classes = rede.classificar(X)
        np.testing.assert_array_equal(classes.sum(axis=1), np.ones(20))
        np.testing.assert_array_equal(
            np.argmax(classes, axis=1), np.argmax(rede.prever(X), axis=1)
        )
        self.assertIn("maior ŷ na classificação", rede.resumo()[2])

    def test_resumo(self):
        self.assertEqual(
            self.rede.resumo(),
            [
                "Entrada: 2 neurônios (sem ativação)",
                "Oculta 1: 3 neurônios, sigmoid — 2 × 3 = 6 pesos (+3 bias)",
                "Saída: 1 neurônio, sigmoid no treino / degrau na classificação"
                " — 3 × 1 = 3 pesos (+1 bias)",
                "Total de pesos: 9",
            ],
        )


    def test_resumo_saida_linear(self):
        rede = RedeNeural(2, [], 1, usar_bias=False, seed=1, ativacao_saida="linear")
        self.assertEqual(
            rede.resumo()[1],
            "Saída: 1 neurônio, linear — degrau na classificação — 2 × 1 = 2 pesos",
        )


class TesteNeuronio(unittest.TestCase):
    """Entidade Neurônio (seção 1.1): pesos, bias, z, a e delta."""

    def neuronio(self, usar_bias=True):
        return Neuronio(np.array([0.4, -1.2]), 0.3, "sigmoid", usar_bias)

    def test_ativar_guarda_entradas_z_e_a(self):
        n = self.neuronio()
        a = n.ativar(np.array([0.8, 0.1]))
        self.assertAlmostEqual(n.z, 0.4 * 0.8 - 1.2 * 0.1 + 0.3)
        self.assertAlmostEqual(a, 1 / (1 + np.exp(-n.z)))
        self.assertEqual(n.a, a)
        np.testing.assert_array_equal(n.entradas, [0.8, 0.1])

    def test_delta_e_derivada_vezes_dE_da(self):
        n = self.neuronio()
        n.ativar(np.array([0.8, 0.1]))
        self.assertAlmostEqual(n.calcular_delta(2.0), 2.0 * n.a * (1 - n.a))
        self.assertEqual(n.delta, n.calcular_delta(2.0))

    def test_atualizar_usa_delta_e_entradas(self):
        n = self.neuronio()
        n.ativar(np.array([0.8, 0.1]))
        delta = n.calcular_delta(1.5)
        n.atualizar(0.1)
        np.testing.assert_allclose(n.pesos, [0.4 - 0.1 * delta * 0.8, -1.2 - 0.1 * delta * 0.1])
        self.assertAlmostEqual(n.bias, 0.3 - 0.1 * delta)

    def test_sem_bias_o_bias_nao_muda(self):
        n = Neuronio(np.array([0.4, -1.2]), 0.0, "sigmoid", usar_bias=False)
        n.ativar(np.array([0.8, 0.1]))
        n.calcular_delta(1.5)
        n.atualizar(0.1)
        self.assertEqual(n.bias, 0.0)

    def test_backward_usa_pesos_antes_da_atualizacao(self):
        # Seção 1.8: todos os deltas antes de qualquer atualização.
        camada = Camada(3, 2, "tanh", True, np.random.default_rng(0))
        camada.forward(np.array([0.2, -0.5, 0.9]))
        W_antigo = camada.W
        retorno = camada.backward(np.array([0.3, -0.7]))
        camada.atualizar(0.5)
        np.testing.assert_allclose(retorno, camada.deltas @ W_antigo)
        self.assertFalse(np.allclose(camada.W, W_antigo))


class TesteExemploAula13(unittest.TestCase):
    """Exemplo XOR dos slides da Aula 13 (Multi-Layer Perceptron).

    Pesos dos slides; vieses começam em 0. Os slides fazem a média dos
    gradientes das 4 amostras (dividem por N = 4). No SGD cada amostra
    usa N = 1, então delta_SGD / 4 = delta do slide.
    """

    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    Y = np.array([[0.0], [1.0], [1.0], [0.0]])

    def setUp(self):
        self.rede = RedeNeural(2, [(3, "sigmoid")], 1, usar_bias=True, seed=0)
        oculta, saida = self.rede.camadas
        for neuronio, pesos in zip(
            oculta.neuronios, [[-0.420, 0.361], [-0.730, -0.617], [-0.941, -0.482]]
        ):
            neuronio.pesos = np.array(pesos)
        saida.neuronios[0].pesos = np.array([-0.015, -0.902, 0.136])

    def test_forward_e_mse(self):
        y_hat = self.rede.forward(self.X)
        np.testing.assert_allclose(y_hat[:, 0], [0.4036, 0.4322, 0.4351, 0.4583], atol=1e-4)
        self.assertAlmostEqual(mse(self.Y, y_hat), 0.25363, delta=1e-4)

    def test_deltas_e_gradientes_da_saida(self):
        oculta, saida = self.rede.camadas
        deltas, gradientes_w, gradiente_b = [], np.zeros(3), 0.0
        for x, y in zip(self.X, self.Y):
            self.rede.forward(x)
            self.rede.retropropagar(y)
            delta = saida.neuronios[0].delta / 4
            deltas.append(delta)
            gradientes_w += delta * np.array([n.a for n in oculta.neuronios])
            gradiente_b += delta
        np.testing.assert_allclose(deltas, [0.04857, -0.06967, -0.06943, 0.05690], atol=1e-5)
        np.testing.assert_allclose(gradientes_w, [-0.01669, -0.01097, -0.01075], atol=1e-5)
        self.assertAlmostEqual(gradiente_b, -0.03363, delta=1e-5)

    def test_deltas_da_camada_oculta(self):
        # Slide 14 (usa os pesos originais da saída, antes do ajuste).
        esperado = [
            [-0.00018215, -0.01095348, 0.00165152],
            [0.00025294, 0.01430602, -0.00223646],
            [0.00024919, 0.01374150, -0.00190632],
            [-0.00021317, -0.00840496, 0.00121082],
        ]
        oculta = self.rede.camadas[0]
        for x, y, linha in zip(self.X, self.Y, esperado):
            self.rede.forward(x)
            self.rede.retropropagar(y)
            np.testing.assert_allclose(oculta.deltas / 4, linha, atol=1e-6)


class TesteExemploDaAula(unittest.TestCase):
    """Teste 3 do plano: exemplo feito à mão na aula (sem bias)."""

    def test_entrada_zero_zero(self):
        rede = RedeNeural(2, [(3, "sigmoid")], 1, usar_bias=False, seed=7)
        oculta, saida = rede.camadas
        # Pesos da saída atribuídos manualmente; os da oculta ficam aleatórios.
        saida.neuronios[0].pesos = np.array([-0.015, -0.902, 0.136])

        y_hat = rede.forward(np.array([[0.0, 0.0]]))

        # Entrada (0, 0) sem bias: z = 0 em todos os ocultos, qualquer que
        # seja o peso, então a = sigmoid(0) = 0,5.
        self.assertEqual([n.a for n in oculta.neuronios], [0.5, 0.5, 0.5])
        # z da saída = 0,5 · (−0,015 − 0,902 + 0,136) = −0,3905
        # ŷ = sigmoid(−0,3905) ≈ 0,4036
        self.assertAlmostEqual(y_hat[0, 0], 0.4036, delta=1e-4)


class TestePerda(unittest.TestCase):
    """MSE e sua derivada."""

    def test_mse_com_uma_saida(self):
        Y = np.array([[0.0], [1.0]])
        Y_hat = np.array([[0.5], [0.0]])
        # (0,5² + 1²) / 2 = 0,625
        self.assertAlmostEqual(mse(Y, Y_hat), 0.625)

    def test_mse_zero_quando_acerta_tudo(self):
        Y = np.array([[0.0, 1.0], [1.0, 0.0]])
        self.assertEqual(mse(Y, Y.copy()), 0.0)

    def test_derivada_mse(self):
        Y = np.array([[0.0], [1.0]])
        Y_hat = np.array([[0.5], [0.0]])
        # (2/2) · (ŷ − y)
        np.testing.assert_allclose(derivada_mse(Y, Y_hat), [[0.5], [-1.0]])


class TesteGradiente(unittest.TestCase):
    """Teste 4 do plano: o gradiente do backpropagation bate com o numérico.

    Para cada peso w, mexe um pouquinho nele e mede o quanto o erro mudou:
        ∂E/∂w ≈ (E(w + ε) − E(w − ε)) / (2ε)
    Se o backpropagation estiver certo, dW tem que dar o mesmo valor.
    """

    EPS = 1e-5

    def conferir(self, analitico, numerico, descricao):
        diferenca = np.linalg.norm(analitico - numerico)
        tamanho = np.linalg.norm(analitico) + np.linalg.norm(numerico)
        if tamanho < 1e-12:
            # Ex.: neurônio ReLU com z < 0 neste exemplo: os dois gradientes são 0.
            return
        self.assertLess(diferenca / tamanho, 1e-5, descricao)

    def conferir_rede(self, rede, x, y, descricao):
        # SGD: gradiente de UM exemplo. Para o peso j do neurônio,
        # ∂E/∂w_j = delta · entrada_j; para o bias, ∂E/∂b = delta.
        Y = y.reshape(1, -1)
        rede.forward(x)
        rede.retropropagar(y)
        analiticos = [
            [(n.delta * n.entradas.copy(), n.delta) for n in camada.neuronios]
            for camada in rede.camadas
        ]

        for i, camada in enumerate(rede.camadas):
            for k, neuronio in enumerate(camada.neuronios):
                dW, db = analiticos[i][k]
                numerico = np.zeros_like(neuronio.pesos)
                for j in range(len(neuronio.pesos)):
                    original = neuronio.pesos[j]
                    neuronio.pesos[j] = original + self.EPS
                    erro_mais = mse(Y, rede.forward(x))
                    neuronio.pesos[j] = original - self.EPS
                    erro_menos = mse(Y, rede.forward(x))
                    neuronio.pesos[j] = original
                    numerico[j] = (erro_mais - erro_menos) / (2 * self.EPS)
                self.conferir(dW, numerico, f"{descricao}, camada {i}, neurônio {k}, pesos")
                if camada.usar_bias:
                    original = neuronio.bias
                    neuronio.bias = original + self.EPS
                    erro_mais = mse(Y, rede.forward(x))
                    neuronio.bias = original - self.EPS
                    erro_menos = mse(Y, rede.forward(x))
                    neuronio.bias = original
                    numerico_b = (erro_mais - erro_menos) / (2 * self.EPS)
                    self.assertAlmostEqual(db, numerico_b, delta=1e-6,
                                           msg=f"{descricao}, camada {i}, neurônio {k}, bias")

    def dados(self, n_entradas, n_saidas):
        gerador = np.random.default_rng(0)
        x = gerador.uniform(-1, 1, size=n_entradas)
        y = gerador.integers(0, 2, size=n_saidas).astype(float)
        return x, y

    def test_cada_ativacao_com_bias(self):
        X, Y = self.dados(3, 2)
        for nome in ATIVACOES:
            rede = RedeNeural(3, [(4, nome)], 2, usar_bias=True, seed=3)
            self.conferir_rede(rede, X, Y, nome)

    def test_saida_linear(self):
        X, Y = self.dados(3, 1)
        rede = RedeNeural(3, [(4, "relu")], 1, usar_bias=True, seed=3, ativacao_saida="linear")
        self.conferir_rede(rede, X, Y, "saída linear")

    def test_duas_ocultas_com_ativacoes_diferentes(self):
        X, Y = self.dados(3, 1)
        rede = RedeNeural(3, [(4, "tanh"), (3, "sigmoid")], 1, usar_bias=True, seed=3)
        self.conferir_rede(rede, X, Y, "tanh + sigmoid")

    def test_sem_bias(self):
        X, Y = self.dados(2, 1)
        rede = RedeNeural(2, [(3, "sigmoid")], 1, usar_bias=False, seed=3)
        self.conferir_rede(rede, X, Y, "sem bias")

    def test_sem_camada_oculta(self):
        X, Y = self.dados(2, 1)
        rede = RedeNeural(2, [], 1, usar_bias=True, seed=3)
        self.conferir_rede(rede, X, Y, "sem oculta")


class TesteTreino(unittest.TestCase):
    """Treino com SGD (seção 1.10): rede/treinamento.py."""

    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    Y = np.array([[0.0], [0.0], [0.0], [1.0]])

    def rede(self, usar_bias=True):
        return RedeNeural(2, [(3, "sigmoid")], 1, usar_bias=usar_bias, seed=1)

    def test_treinar_exemplo_muda_os_pesos_de_todas_as_camadas(self):
        rede = self.rede()
        antes = [camada.W for camada in rede.camadas]
        rede.treinar_exemplo(self.X[3], self.Y[3], 0.5)
        for camada, W in zip(rede.camadas, antes):
            self.assertFalse(np.allclose(camada.W, W))

    def test_historico_tem_treino_e_validacao_por_epoca(self):
        historico = treinar(self.rede(), self.X, self.Y, 20, 0.5,
                            X_validacao=self.X, Y_validacao=self.Y, seed=0)
        self.assertEqual(len(historico["treino"]), 20)
        self.assertEqual(len(historico["validacao"]), 20)

    def test_sem_validacao_lista_vazia(self):
        historico = treinar(self.rede(), self.X, self.Y, 5, 0.5, seed=0)
        self.assertEqual(historico["validacao"], [])

    def test_erro_diminui(self):
        historico = treinar(self.rede(), self.X, self.Y, 200, 0.5, seed=0)
        self.assertLess(historico["treino"][-1], historico["treino"][0])

    def test_callback_recebe_epoca_treino_e_validacao(self):
        chamadas = []
        historico = treinar(
            self.rede(), self.X, self.Y, 5, 0.5, X_validacao=self.X, Y_validacao=self.Y,
            seed=0, ao_fim_da_epoca=lambda *args: chamadas.append(args),
        )
        self.assertEqual([c[0] for c in chamadas], [1, 2, 3, 4, 5])
        self.assertEqual([c[1] for c in chamadas], historico["treino"])
        self.assertEqual([c[2] for c in chamadas], historico["validacao"])

    def test_callback_sem_validacao_recebe_none(self):
        chamadas = []
        treinar(self.rede(), self.X, self.Y, 2, 0.5, seed=0,
                ao_fim_da_epoca=lambda *args: chamadas.append(args))
        self.assertEqual([c[2] for c in chamadas], [None, None])

    def test_mesma_seed_mesmo_treino(self):
        a = treinar(self.rede(), self.X, self.Y, 10, 0.5, seed=3)
        b = treinar(self.rede(), self.X, self.Y, 10, 0.5, seed=3)
        self.assertEqual(a, b)

    def test_sem_bias_o_bias_continua_zero(self):
        rede = self.rede(usar_bias=False)
        treinar(rede, self.X, self.Y, 50, 0.5, seed=0)
        for camada in rede.camadas:
            np.testing.assert_array_equal(camada.b, 0)

    def test_y_com_forma_errada_da_erro(self):
        with self.assertRaises(ValueError):
            treinar(self.rede(), self.X, np.zeros((4, 2)), 1, 0.5)

    def test_saida_linear_aprende_uma_reta(self):
        # Regressão: y = 3x + 2, sem normalizar Y (a saída linear alcança).
        x = np.linspace(0, 1, 50).reshape(-1, 1)
        y = 3 * x + 2
        rede = RedeNeural(1, [], 1, usar_bias=True, seed=1, ativacao_saida="linear")
        historico = treinar(rede, x, y, 200, 0.1, seed=0)
        self.assertLess(historico["treino"][-1], 1e-3)


class TesteAprendizado(unittest.TestCase):
    """A rede aprende de verdade as bases AND, XOR e Iris (com SGD)."""

    def carregar(self, nome):
        tabela = np.genfromtxt(PASTA_BASES / nome, delimiter=",", skip_header=1)
        return tabela[:, :-1], tabela[:, -1:]

    def acertos(self, rede, X, Y):
        return int(np.sum(np.all(rede.classificar(X) == Y, axis=1)))

    def test_and_sem_camada_oculta(self):
        # AND é linearmente separável: um único neurônio (perceptron) resolve.
        X, Y = self.carregar("and.csv")
        rede = RedeNeural(2, [], 1, usar_bias=True, seed=42)
        treinar(rede, X, Y, 1000, 0.5, seed=42)
        self.assertEqual(self.acertos(rede, X, Y), 4)

    def test_xor_com_camada_oculta(self):
        X, Y = self.carregar("xor.csv")
        rede = RedeNeural(2, [(4, "sigmoid")], 1, usar_bias=True, seed=42)
        historico = treinar(rede, X, Y, 3000, 0.5, seed=42)
        self.assertEqual(self.acertos(rede, X, Y), 4)
        self.assertLess(historico["treino"][-1], 0.05)

    def test_iris(self):
        # 3 classes → 3 neurônios de saída; a classe é o de maior ŷ.
        X, Y, _ = carregar_csv(PASTA_DATASET / "bezdekIris.data", n_saidas=3)
        X_treino, Y_treino, X_teste, Y_teste, _ = separar_treino_teste(X, Y, seed=1)
        X_treino, minimos, maximos = normalizar(X_treino)
        X_teste = aplicar_normalizacao(X_teste, minimos, maximos)

        rede = RedeNeural(4, [(5, "sigmoid")], 3, usar_bias=True, seed=1)
        treinar(rede, X_treino, Y_treino, 200, 0.1, seed=1)

        self.assertGreaterEqual(acuracia(Y_teste, rede.classificar(X_teste)), 0.9)
        self.assertGreaterEqual(acuracia(Y_treino, rede.classificar(X_treino)), 0.9)

    def test_xor_sem_camada_oculta_nao_aprende(self):
        # XOR não é linearmente separável: sem oculta nunca chega a 100%.
        X, Y = self.carregar("xor.csv")
        for seed in range(5):
            rede = RedeNeural(2, [], 1, usar_bias=True, seed=seed)
            treinar(rede, X, Y, 1000, 0.5, seed=seed)
            self.assertLess(self.acertos(rede, X, Y), 4, f"seed {seed}")


if __name__ == "__main__":
    unittest.main()
