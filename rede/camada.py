"""Classe Camada: um grupo de neurônios totalmente conectado à camada anterior."""

import numpy as np

from rede.ativacoes import ATIVACOES


class Camada:
    """Uma camada da rede (oculta ou de saída).

    Cada neurônio desta camada recebe a saída de TODOS os neurônios da camada
    anterior. Por isso os pesos formam uma matriz W com forma
    (n_neuronios, n_entradas): a linha i guarda os pesos que chegam no
    neurônio i, e a coluna j corresponde ao neurônio j da camada anterior.
    """

    def __init__(self, n_entradas, n_neuronios, nome_ativacao, usar_bias, gerador):
        """
        n_entradas:    quantos neurônios tem a camada anterior
        n_neuronios:   quantos neurônios tem esta camada
        nome_ativacao: "sigmoid", "relu" ou "tanh"
        usar_bias:     se False, o bias fica zerado e nunca é atualizado
        gerador:       np.random.Generator usado para sortear os pesos
        """
        if nome_ativacao not in ATIVACOES:
            opcoes = ", ".join(ATIVACOES)
            raise ValueError(
                f"Ativação desconhecida: '{nome_ativacao}'. Opções: {opcoes}."
            )

        self.n_entradas = n_entradas
        self.n_neuronios = n_neuronios
        self.nome_ativacao = nome_ativacao
        self.usar_bias = usar_bias
        self.ativacao, self.derivada_ativacao = ATIVACOES[nome_ativacao]

        # Pesos iniciais aleatórios, distribuição uniforme entre -1 e 1.
        self.W = gerador.uniform(-1, 1, size=(n_neuronios, n_entradas))
        # Bias começa em zero (um por neurônio).
        self.b = np.zeros(n_neuronios)

        # Valores guardados pelo forward; o backward vai precisar deles.
        self.A_anterior = None
        self.Z = None
        self.A = None

        # Gradientes calculados pelo backward e usados pelo atualizar.
        self.dW = np.zeros_like(self.W)  # ∂E/∂W, mesma forma de W
        self.db = np.zeros_like(self.b)  # ∂E/∂b, mesma forma de b

    def forward(self, A_anterior):
        """Calcula a saída da camada (propagação para frente).

        A_anterior: saídas da camada anterior, forma (N, n_entradas),
                    onde N é o número de amostras.
        Devolve A, forma (N, n_neuronios).
        """
        # Soma ponderada: para cada amostra, z_i = Σ_j (w_ij · a_j) + b_i.
        # Em matriz: Z = A_anterior · Wᵀ + b
        Z = A_anterior @ self.W.T + self.b
        # Saída do neurônio: a = f(z)
        A = self.ativacao(Z)

        self.A_anterior = A_anterior
        self.Z = Z
        self.A = A
        return A

    def backward(self, dA):
        """Calcula os gradientes da camada (propagação para trás).

        dA: ∂E/∂A desta camada, forma (N, n_neuronios). Diz quanto o erro
            muda quando a saída de cada neurônio muda.
        Guarda dW e db, e devolve ∂E/∂A da camada ANTERIOR, forma
        (N, n_entradas), para o backpropagation continuar para trás.

        Tudo vem da regra da cadeia, seguindo o caminho W → Z → A → E.
        Precisa ser chamado depois do forward (usa A_anterior, Z e A).
        """
        # Passo A → Z: como a = f(z), então ∂E/∂z = ∂E/∂a · f'(z).
        dZ = dA * self.derivada_ativacao(self.Z, self.A)

        # Passo Z → W: como z_i = Σ_j (w_ij · a_j) + b_i, então
        # ∂z_i/∂w_ij = a_j (a saída do neurônio j da camada anterior).
        # Logo ∂E/∂w_ij = Σ_amostras (dz_i · a_j). Em matriz: dW = dZᵀ · A_anterior.
        # A soma sobre as amostras acontece no produto; o 1/N já veio na
        # derivada do erro.
        self.dW = dZ.T @ self.A_anterior

        # Passo Z → b: ∂z_i/∂b_i = 1, então ∂E/∂b_i = Σ_amostras dz_i.
        if self.usar_bias:
            self.db = dZ.sum(axis=0)

        # Passo Z → A_anterior: ∂z_i/∂a_j = w_ij. Cada neurônio j da camada
        # anterior influencia todos os neurônios daqui, então soma em i:
        # ∂E/∂a_j = Σ_i (dz_i · w_ij). Em matriz: dA_anterior = dZ · W.
        # Usa o W atual: por isso o backward de todas as camadas roda antes
        # de qualquer atualização de pesos.
        return dZ @ self.W

    def atualizar(self, taxa):
        """Gradiente descendente: W = W − η · ∂E/∂W (e o mesmo para o bias).

        taxa: taxa de aprendizado (η). O sinal de menos faz o peso andar na
        direção em que o erro diminui.
        """
        self.W -= taxa * self.dW
        if self.usar_bias:
            self.b -= taxa * self.db

    def numero_de_pesos(self):
        """Quantidade de pesos da camada: n_neuronios × n_entradas.

        Como a ligação é total, dá para conferir pela camada anterior:
        cada neurônio daqui tem um peso para cada neurônio de lá.
        """
        return self.n_neuronios * self.n_entradas
