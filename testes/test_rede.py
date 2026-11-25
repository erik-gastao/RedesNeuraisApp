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
    derivada_tanh,
    maior_saida,
    relu,
    sigmoid,
    tanh,
)
from rede.perda import derivada_mse, mse
from rede.rede_neural import RedeNeural

# Pasta bases/ na raiz do projeto (este arquivo fica em testes/).
PASTA_BASES = Path(__file__).resolve().parent.parent / "bases"
PASTA_DATASET = Path(__file__).resolve().parent.parent / "Dataset"


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

    def test_dicionario_tem_as_tres_ativacoes(self):
        self.assertEqual(set(ATIVACOES), {"sigmoid", "relu", "tanh"})


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
        X = np.array([[0.0, 1.0], [1.0, 1.0]])
        self.rede.forward(X)
        oculta, saida = self.rede.camadas
        np.testing.assert_array_equal(oculta.A_anterior, X)
        self.assertEqual(oculta.Z.shape, (2, 3))
        self.assertEqual(oculta.A.shape, (2, 3))
        # A entrada da saída é a saída da oculta.
        np.testing.assert_array_equal(saida.A_anterior, oculta.A)

    def test_pesos_iniciais_entre_menos_um_e_um(self):
        for camada in self.rede.camadas:
            self.assertTrue(np.all(camada.W >= -1) and np.all(camada.W <= 1))
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
            RedeNeural(2, [(3, "linear")], 1, usar_bias=True, seed=1)

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


class TesteExemploDaAula(unittest.TestCase):
    """Teste 3 do plano: exemplo feito à mão na aula (sem bias)."""

    def test_entrada_zero_zero(self):
        rede = RedeNeural(2, [(3, "sigmoid")], 1, usar_bias=False, seed=7)
        oculta, saida = rede.camadas
        # Pesos da saída atribuídos manualmente; os da oculta ficam aleatórios.
        saida.W = np.array([[-0.015, -0.902, 0.136]])

        y_hat = rede.forward(np.array([[0.0, 0.0]]))

        # Entrada (0, 0) sem bias: z = 0 em todos os ocultos, qualquer que
        # seja o peso, então a = sigmoid(0) = 0,5.
        np.testing.assert_array_equal(oculta.A, [[0.5, 0.5, 0.5]])
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

    def gradiente_numerico(self, rede, parametro, X, Y):
        numerico = np.zeros_like(parametro)
        for indice in np.ndindex(parametro.shape):
            original = parametro[indice]
            parametro[indice] = original + self.EPS
            erro_mais = mse(Y, rede.forward(X))
            parametro[indice] = original - self.EPS
            erro_menos = mse(Y, rede.forward(X))
            parametro[indice] = original
            numerico[indice] = (erro_mais - erro_menos) / (2 * self.EPS)
        return numerico

    def conferir(self, analitico, numerico, descricao):
        diferenca = np.linalg.norm(analitico - numerico)
        tamanho = np.linalg.norm(analitico) + np.linalg.norm(numerico)
        self.assertLess(diferenca / tamanho, 1e-5, descricao)

    def conferir_rede(self, rede, X, Y, descricao):
        # Gradientes do backpropagation (sem atualizar os pesos).
        dA = derivada_mse(Y, rede.forward(X))
        for camada in reversed(rede.camadas):
            dA = camada.backward(dA)

        for i, camada in enumerate(rede.camadas):
            self.conferir(
                camada.dW,
                self.gradiente_numerico(rede, camada.W, X, Y),
                f"{descricao}, camada {i}, W",
            )
            if camada.usar_bias:
                self.conferir(
                    camada.db,
                    self.gradiente_numerico(rede, camada.b, X, Y),
                    f"{descricao}, camada {i}, b",
                )

    def dados(self, n_entradas, n_saidas):
        gerador = np.random.default_rng(0)
        X = gerador.uniform(-1, 1, size=(5, n_entradas))
        Y = gerador.integers(0, 2, size=(5, n_saidas)).astype(float)
        return X, Y

    def test_cada_ativacao_com_bias(self):
        X, Y = self.dados(3, 2)
        for nome in ATIVACOES:
            rede = RedeNeural(3, [(4, nome)], 2, usar_bias=True, seed=3)
            self.conferir_rede(rede, X, Y, nome)

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
    """Comportamento do laço de treino."""

    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    Y = np.array([[0.0], [0.0], [0.0], [1.0]])

    def test_historico_tem_um_erro_por_epoca(self):
        rede = RedeNeural(2, [(3, "sigmoid")], 1, usar_bias=True, seed=1)
        historico = rede.treinar(self.X, self.Y, epocas=20, taxa=0.5)
        self.assertEqual(len(historico), 20)

    def test_erro_diminui(self):
        rede = RedeNeural(2, [(3, "sigmoid")], 1, usar_bias=True, seed=1)
        historico = rede.treinar(self.X, self.Y, epocas=200, taxa=0.5)
        self.assertLess(historico[-1], historico[0])

    def test_funcao_de_progresso_e_chamada_a_cada_epoca(self):
        rede = RedeNeural(2, [(3, "sigmoid")], 1, usar_bias=True, seed=1)
        chamadas = []
        historico = rede.treinar(
            self.X, self.Y, epocas=5, taxa=0.5,
            ao_fim_da_epoca=lambda epoca, erro: chamadas.append((epoca, erro)),
        )
        self.assertEqual([epoca for epoca, _ in chamadas], [1, 2, 3, 4, 5])
        self.assertEqual([erro for _, erro in chamadas], historico)

    def test_sem_bias_o_bias_continua_zero(self):
        rede = RedeNeural(2, [(3, "sigmoid")], 1, usar_bias=False, seed=1)
        rede.treinar(self.X, self.Y, epocas=50, taxa=0.5)
        for camada in rede.camadas:
            np.testing.assert_array_equal(camada.b, 0)

    def test_y_com_forma_errada_da_erro(self):
        rede = RedeNeural(2, [(3, "sigmoid")], 1, usar_bias=True, seed=1)
        with self.assertRaises(ValueError):
            rede.treinar(self.X, np.zeros((4, 2)), epocas=1, taxa=0.5)


class TesteAprendizado(unittest.TestCase):
    """Teste 5 do plano: a rede aprende de verdade as bases AND e XOR."""

    def carregar(self, nome):
        # Leitura direta (o dados.py, da etapa 5, ainda não é usado aqui).
        # Pula o cabeçalho; a última coluna é a saída esperada.
        tabela = np.genfromtxt(PASTA_BASES / nome, delimiter=",", skip_header=1)
        return tabela[:, :-1], tabela[:, -1:]

    def acertos(self, rede, X, Y):
        return int(np.sum(np.all(rede.classificar(X) == Y, axis=1)))

    def test_and_sem_camada_oculta(self):
        # AND é linearmente separável: uma reta divide as classes, então
        # um único neurônio (perceptron) resolve.
        X, Y = self.carregar("and.csv")
        rede = RedeNeural(2, [], 1, usar_bias=True, seed=42)
        rede.treinar(X, Y, epocas=5000, taxa=0.5)
        self.assertEqual(self.acertos(rede, X, Y), 4)

    def test_xor_com_camada_oculta(self):
        X, Y = self.carregar("xor.csv")
        rede = RedeNeural(2, [(4, "sigmoid")], 1, usar_bias=True, seed=42)
        historico = rede.treinar(X, Y, epocas=10000, taxa=0.5)
        self.assertEqual(self.acertos(rede, X, Y), 4)
        self.assertLess(historico[-1], 0.05)

    def test_iris(self):
        # 3 classes → 3 neurônios de saída; a classe é o de maior ŷ.
        # Versicolor e virginica se misturam um pouco, então o esperado é
        # perto de 95%, não 100%.
        X, Y, _ = carregar_csv(PASTA_DATASET / "bezdekIris.data", n_saidas=3)
        X_treino, Y_treino, X_teste, Y_teste, _ = separar_treino_teste(X, Y, seed=1)
        X_treino, minimos, maximos = normalizar(X_treino)
        X_teste = aplicar_normalizacao(X_teste, minimos, maximos)

        rede = RedeNeural(4, [(5, "sigmoid")], 3, usar_bias=True, seed=1)
        rede.treinar(X_treino, Y_treino, epocas=5000, taxa=0.5)

        self.assertGreaterEqual(acuracia(Y_teste, rede.classificar(X_teste)), 0.9)
        self.assertGreaterEqual(acuracia(Y_treino, rede.classificar(X_treino)), 0.9)

    def test_xor_sem_camada_oculta_nao_aprende(self):
        # XOR não é linearmente separável: nenhuma reta divide as classes.
        # Sem camada oculta a rede nunca chega a 100%, com qualquer seed.
        X, Y = self.carregar("xor.csv")
        for seed in range(5):
            rede = RedeNeural(2, [], 1, usar_bias=True, seed=seed)
            rede.treinar(X, Y, epocas=10000, taxa=0.5)
            self.assertLess(self.acertos(rede, X, Y), 4, f"seed {seed}")


if __name__ == "__main__":
    unittest.main()
