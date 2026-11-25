"""Treinamento da rede com SGD (seção 1.10 do enunciado).

Stochastic Gradient Descent: os pesos e vieses são atualizados depois de
CADA exemplo de treino (e não uma vez por época com a base inteira).
Ao fim de cada época, mede o MSE do treino e da validação.
"""

import numpy as np

from rede.perda import mse


def treinar(rede, X, Y, epocas, taxa, X_validacao=None, Y_validacao=None,
            seed=None, ao_fim_da_epoca=None):
    """Treina a rede e devolve o histórico de erros.

    X, Y:     conjunto de treino, formas (N, n_entradas) e (N, n_saidas)
    epocas:   quantas vezes o conjunto de treino inteiro passa pela rede
    taxa:     taxa de aprendizado (η)
    X_validacao, Y_validacao: conjunto de validação (opcional). Só é usado
              para medir o erro: nunca atualiza pesos.
    seed:     semente da ordem dos exemplos (a ordem é sorteada a cada época)
    ao_fim_da_epoca: função opcional chamada como
              f(epoca, erro_treino, erro_validacao) — erro_validacao é None
              sem conjunto de validação. A época começa em 1.

    Devolve {"treino": [MSE por época], "validacao": [MSE por época]}.
    Os erros são medidos com os pesos do fim de cada época.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    Y = _como_matriz(Y)
    if Y.shape != (X.shape[0], rede.n_saidas):
        raise ValueError(
            f"Y deveria ter forma ({X.shape[0]}, {rede.n_saidas}), "
            f"mas tem {Y.shape}."
        )

    tem_validacao = X_validacao is not None and len(X_validacao) > 0
    if tem_validacao:
        X_validacao = np.atleast_2d(np.asarray(X_validacao, dtype=float))
        Y_validacao = _como_matriz(Y_validacao)

    gerador = np.random.default_rng(seed)
    historico = {"treino": [], "validacao": []}
    for epoca in range(1, epocas + 1):
        # SGD: um exemplo por vez, em ordem embaralhada.
        for i in gerador.permutation(X.shape[0]):
            rede.treinar_exemplo(X[i], Y[i], taxa)

        erro_treino = mse(Y, rede.forward(X))
        historico["treino"].append(erro_treino)
        erro_validacao = None
        if tem_validacao:
            erro_validacao = mse(Y_validacao, rede.forward(X_validacao))
            historico["validacao"].append(erro_validacao)

        if ao_fim_da_epoca is not None:
            ao_fim_da_epoca(epoca, erro_treino, erro_validacao)

    return historico


def _como_matriz(Y):
    """Vetor simples de saídas → uma coluna (caso de 1 neurônio de saída)."""
    Y = np.asarray(Y, dtype=float)
    return Y.reshape(-1, 1) if Y.ndim == 1 else Y
