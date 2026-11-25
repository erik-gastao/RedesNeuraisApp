"""Inicialização dos pesos sinápticos (seção 1.5 do enunciado).

Os pesos começam aleatórios, sorteados de forma uniforme entre −limite e
+limite. O limite depende da função de ativação da camada:

    Xavier/Glorot (sigmoid, tanh, linear): limite = √(6 / (n_entradas + n_neuronios))
    He (ReLU):                             limite = √(6 / n_entradas)

Por que não sortear sempre entre −1 e 1: com muitas entradas, a soma z
fica grande, a sigmoid/tanh saturam (derivada ≈ 0) e a rede não aprende.
Os limites acima mantêm z numa faixa boa. A ReLU zera metade dos valores,
por isso o He usa um limite maior para compensar.

Os vieses começam em 0 (são criados pela camada).
"""

import numpy as np

# Qual inicialização cada ativação usa.
METODO_POR_ATIVACAO = {
    "sigmoid": "xavier",
    "tanh": "xavier",
    "linear": "xavier",
    "relu": "he",
}


def pesos_iniciais(n_entradas, n_neuronios, nome_ativacao, gerador):
    """Matriz de pesos iniciais, forma (n_neuronios, n_entradas).

    A linha i guarda os pesos que chegam no neurônio i.
    gerador: np.random.Generator (com a mesma seed, os mesmos pesos).
    """
    if METODO_POR_ATIVACAO[nome_ativacao] == "he":
        limite = np.sqrt(6 / n_entradas)
    else:
        limite = np.sqrt(6 / (n_entradas + n_neuronios))
    return gerador.uniform(-limite, limite, size=(n_neuronios, n_entradas))
