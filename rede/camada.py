"""Classe Camada: um grupo de neurônios totalmente conectado à camada anterior."""

import numpy as np

from rede.ativacoes import ATIVACOES
from rede.inicializacao import pesos_iniciais
from rede.neuronio import Neuronio


class Camada:
    """Uma camada da rede (oculta ou de saída), formada por Neuronios.

    Cada neurônio recebe a saída de TODOS os neurônios da camada anterior,
    e todos usam a mesma função de ativação (a da camada).
    """

    def __init__(self, n_entradas, n_neuronios, nome_ativacao, usar_bias, gerador):
        """
        n_entradas:    quantos neurônios tem a camada anterior
        n_neuronios:   quantos neurônios tem esta camada
        nome_ativacao: "sigmoid", "relu", "tanh" ou "linear"
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

        # Pesos iniciais Xavier/He (ver inicializacao.py); bias começa em 0.
        # A linha i da matriz vira os pesos do neurônio i.
        pesos = pesos_iniciais(n_entradas, n_neuronios, nome_ativacao, gerador)
        self.neuronios = [
            Neuronio(pesos[i].copy(), 0.0, nome_ativacao, usar_bias)
            for i in range(n_neuronios)
        ]

    def forward(self, entradas):
        """Passa um exemplo pela camada: cada neurônio calcula sua saída.

        entradas: saídas da camada anterior, forma (n_entradas,).
        Devolve as saídas da camada, forma (n_neuronios,).
        """
        return np.array([neuronio.ativar(entradas) for neuronio in self.neuronios])

    def backward(self, dE_da):
        """Calcula o delta de cada neurônio e devolve ∂E/∂a da camada anterior.

        dE_da: ∂E/∂a de cada neurônio desta camada, forma (n_neuronios,).

        A saída a_j da camada anterior chega em todos os neurônios i daqui
        multiplicada por w_ij, então ∂E/∂a_j = Σ_i δ_i · w_ij.
        Usa os pesos atuais: por isso a rede calcula os deltas de todas as
        camadas antes de atualizar qualquer peso.
        """
        for neuronio, derivada in zip(self.neuronios, dE_da):
            neuronio.calcular_delta(derivada)
        return self.deltas @ self.W

    def atualizar(self, taxa):
        """Atualiza pesos e bias de todos os neurônios (descida do gradiente)."""
        for neuronio in self.neuronios:
            neuronio.atualizar(taxa)

    @property
    def W(self):
        """Pesos da camada como matriz (n_neuronios, n_entradas). É uma cópia."""
        return np.array([neuronio.pesos for neuronio in self.neuronios])

    @property
    def b(self):
        """Vieses da camada, forma (n_neuronios,). É uma cópia."""
        return np.array([neuronio.bias for neuronio in self.neuronios])

    @property
    def deltas(self):
        """Deltas (∂E/∂z) do último exemplo, forma (n_neuronios,)."""
        return np.array([neuronio.delta for neuronio in self.neuronios])

    def numero_de_pesos(self):
        """Quantidade de pesos da camada: n_neuronios × n_entradas."""
        return self.n_neuronios * self.n_entradas
