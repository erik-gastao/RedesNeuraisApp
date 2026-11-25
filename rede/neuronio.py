"""Classe Neuronio: um neurônio artificial (seção 1.1 do enunciado)."""

import numpy as np

from rede.ativacoes import ATIVACOES


class Neuronio:
    """Um neurônio: guarda seus pesos e seu bias e calcula a = f(z).

        z = Σ wⱼ · xⱼ + b      (soma ponderada)
        a = f(z)               (saída, depois da ativação)

    Também guarda os valores que o treino precisa:
        entradas: valores que chegaram no último exemplo (x ou a da camada anterior)
        z, a:     soma ponderada e saída do último exemplo
        delta:    δ = ∂E/∂z, calculado na retropropagação
    """

    def __init__(self, pesos, bias, nome_ativacao, usar_bias):
        """
        pesos:         um peso por entrada (vetor numpy)
        bias:          valor inicial do viés
        nome_ativacao: "sigmoid", "relu", "tanh" ou "linear"
        usar_bias:     se False, o bias nunca é atualizado
        """
        self.pesos = np.array(pesos, dtype=float)
        self.bias = float(bias)
        self.nome_ativacao = nome_ativacao
        self.usar_bias = usar_bias
        self.ativacao, self.derivada_ativacao = ATIVACOES[nome_ativacao]

        self.entradas = None
        self.z = None
        self.a = None
        self.delta = None

    def ativar(self, entradas):
        """Propagação para frente: calcula z e a para um exemplo e devolve a."""
        self.entradas = np.asarray(entradas, dtype=float)
        self.z = float(self.pesos @ self.entradas + self.bias)
        self.a = float(self.ativacao(np.array(self.z)))
        return self.a

    def calcular_delta(self, dE_da):
        """Regra da cadeia: δ = ∂E/∂a · f'(z).

        dE_da: quanto o erro muda quando a saída deste neurônio muda.
        Na saída vem da derivada do erro; nas ocultas, da camada seguinte.
        """
        self.delta = float(dE_da * self.derivada_ativacao(np.array(self.z), np.array(self.a)))
        return self.delta

    def atualizar(self, taxa):
        """Descida do gradiente: w = w − η · δ · x  e  b = b − η · δ.

        ∂E/∂wⱼ = δ · xⱼ (a entrada j multiplica o peso j na soma z);
        ∂E/∂b = δ (o bias soma direto em z).
        """
        self.pesos -= taxa * self.delta * self.entradas
        if self.usar_bias:
            self.bias -= taxa * self.delta
