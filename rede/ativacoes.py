"""Funções de ativação e suas derivadas.

Convenção usada em todo o projeto:
    z = soma ponderada que chega no neurônio (antes da ativação)
    a = saída do neurônio (depois da ativação), ou seja, a = f(z)

Todas as funções recebem e devolvem numpy.ndarray.
Todas as derivadas têm a mesma assinatura, derivada(z, a), mesmo que cada
uma use só um dos dois. Assim a camada pode chamar qualquer uma do mesmo jeito.
"""

import numpy as np


# ---------------------------------------------------------------- sigmoid

def sigmoid(z):
    """Sigmoid: f(z) = 1 / (1 + e^(-z)). Resultado sempre entre 0 e 1."""
    # Limita z para evitar overflow no cálculo de e^(-z) com valores muito grandes.
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def derivada_sigmoid(z, a):
    """Derivada da sigmoid: f'(z) = a · (1 − a), onde a = sigmoid(z)."""
    return a * (1 - a)


# ------------------------------------------------------------------- relu

def relu(z):
    """ReLU: f(z) = max(0, z). Zera os negativos e mantém os positivos."""
    return np.maximum(0, z)


def derivada_relu(z, a):
    """Derivada da ReLU: f'(z) = 1 se z > 0, senão 0."""
    return np.where(z > 0, 1.0, 0.0)


# ------------------------------------------------------------------- tanh

def tanh(z):
    """Tangente hiperbólica: f(z) = tanh(z). Resultado entre -1 e 1."""
    return np.tanh(z)


def derivada_tanh(z, a):
    """Derivada da tanh: f'(z) = 1 − a², onde a = tanh(z)."""
    return 1 - a ** 2


# ----------------------------------------------------------------- linear

def linear(z):
    """Linear (identidade): f(z) = z. A saída não é limitada.

    Usada na saída de problemas de regressão: como as saídas esperadas não
    são normalizadas, a rede precisa conseguir produzir qualquer valor
    (ex.: produtividade de 2500 kg/ha). É o neurônio ADALINE da aula.
    """
    return np.asarray(z, dtype=float)


def derivada_linear(z, a):
    """Derivada da linear: f'(z) = 1."""
    return np.ones_like(np.asarray(z, dtype=float))


# ----------------------------------------------------------------- degrau

def degrau(z):
    """Degrau: f(z) = 1 se z ≥ 0, senão 0.

    NÃO é usado no treino: a derivada do degrau é zero em quase todo ponto,
    então o backpropagation não teria como ajustar os pesos.
    """
    return np.where(z >= 0, 1.0, 0.0)


def classificar(y_hat, limiar=0.5):
    """Transforma a saída contínua da rede (ŷ, entre 0 e 1) em classe 0 ou 1.

    É o degrau aplicado em (ŷ − limiar): devolve 1 se ŷ ≥ 0,5, senão 0.
    A camada de saída usa sigmoid no treino; o degrau só entra aqui, na hora
    de gerar a classe final (avaliação e previsão).
    """
    return degrau(y_hat - limiar)


def maior_saida(y_hat):
    """Classe final quando há várias saídas, uma por classe (ex.: Iris).

    O neurônio de maior ŷ em cada amostra vira 1 e os demais viram 0
    (o "argmax"). Assim a rede sempre escolhe exatamente uma classe.

    Com o degrau em cada saída separada, a rede poderia responder 0 0 0
    (nenhuma classe) ou 1 1 0 (duas classes), o que não faz sentido quando
    cada amostra pertence a uma classe só.

    y_hat: forma (N, n_saidas).
    """
    classes = np.zeros_like(y_hat)
    linhas = np.arange(y_hat.shape[0])
    classes[linhas, np.argmax(y_hat, axis=1)] = 1
    return classes


# Dicionário para escolher a ativação pelo nome (usado pelo menu e pela camada).
# Cada item é o par (função, derivada).
ATIVACOES = {
    "sigmoid": (sigmoid, derivada_sigmoid),
    "relu": (relu, derivada_relu),
    "tanh": (tanh, derivada_tanh),
    "linear": (linear, derivada_linear),
}
