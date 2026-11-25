"""Função de perda (erro) da rede: erro quadrático médio (MSE)."""

import numpy as np


def mse(Y, Y_hat):
    """Erro quadrático médio: E = (1/N) · Σ (y − ŷ)².

    Y:     saídas esperadas, forma (N, n_saidas)
    Y_hat: saídas da rede (ŷ), mesma forma
    Devolve um número: quanto menor, mais perto a rede está do esperado.

    A soma é sobre os neurônios de saída e a média é sobre as N amostras.
    Com 1 neurônio de saída é exatamente a fórmula do professor.
    """
    return float(np.mean(np.sum((Y - Y_hat) ** 2, axis=1)))


def derivada_mse(Y, Y_hat):
    """Derivada do erro em relação à saída da rede: ∂E/∂ŷ = (2/N) · (ŷ − y).

    Devolve uma matriz com a mesma forma de Y_hat. É o ponto de partida do
    backpropagation: diz, para cada saída, em que direção o erro cresce.
    """
    N = Y.shape[0]
    return (2 / N) * (Y_hat - Y)
