"""Classe RedeNeural: uma MLP montada a partir de várias Camadas."""

import numpy as np

from rede.ativacoes import classificar as aplicar_degrau
from rede.ativacoes import maior_saida
from rede.camada import Camada
from rede.perda import derivada_mse


class RedeNeural:
    """Rede MLP: entrada → camadas ocultas (0 ou mais) → saída.

    A camada de entrada não é um objeto Camada: ela só repassa os valores,
    sem pesos e sem função de ativação.
    """

    def __init__(self, n_entradas, ocultas, n_saidas, usar_bias, seed=None,
                 ativacao_saida="sigmoid"):
        """
        n_entradas:     neurônios da camada de entrada (colunas de X)
        ocultas:        lista de (n_neuronios, nome_ativacao), uma por camada
                        oculta. Lista vazia = sem oculta (perceptron simples).
        n_saidas:       neurônios da camada de saída
        usar_bias:      vale para todas as camadas
        seed:           semente para repetir o experimento (None = aleatória)
        ativacao_saida: ativação da camada de saída. "sigmoid" para
                        classificação (ŷ entre 0 e 1); "linear" para
                        regressão (saídas esperadas não normalizadas).
        """
        self.n_entradas = n_entradas
        self.n_saidas = n_saidas
        self.usar_bias = usar_bias
        self.ativacao_saida = ativacao_saida

        # Um único gerador para todas as camadas: com a mesma seed, a rede
        # inteira nasce com os mesmos pesos.
        gerador = np.random.default_rng(seed)

        # Encadeia as camadas: as entradas de cada uma são os neurônios
        # da anterior.
        self.camadas = []
        n_anterior = n_entradas
        for n_neuronios, nome_ativacao in ocultas:
            self.camadas.append(
                Camada(n_anterior, n_neuronios, nome_ativacao, usar_bias, gerador)
            )
            n_anterior = n_neuronios

        self.camadas.append(
            Camada(n_anterior, n_saidas, ativacao_saida, usar_bias, gerador)
        )

    def forward(self, X):
        """Propagação direta, um exemplo por vez, camada por camada.

        X: forma (N, n_entradas). Uma amostra só, forma (n_entradas,),
           também é aceita e tratada como N = 1.
        Devolve Ŷ, forma (N, n_saidas). Os neurônios ficam com os valores
        (entradas, z, a) do último exemplo.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.shape[1] != self.n_entradas:
            raise ValueError(
                f"A rede espera {self.n_entradas} entradas por amostra, "
                f"mas recebeu {X.shape[1]}."
            )
        return np.array([self._propagar(x) for x in X])

    def _propagar(self, x):
        """Um exemplo: a saída de uma camada é a entrada da próxima."""
        for camada in self.camadas:
            x = camada.forward(x)
        return x

    def retropropagar(self, y):
        """Backpropagation do último exemplo passado pelo forward.

        Calcula o delta de todos os neurônios, da saída para a entrada.
        Nenhum peso muda aqui: os deltas das ocultas usam os mesmos pesos
        do forward (seção 1.8 do enunciado).
        """
        y = np.asarray(y, dtype=float).reshape(1, -1)
        y_hat = np.array([[neuronio.a for neuronio in self.camadas[-1].neuronios]])
        # Na saída, ∂E/∂ŷ = 2(ŷ − y) (MSE de um exemplo, N = 1).
        dE_da = derivada_mse(y, y_hat)[0]
        for camada in reversed(self.camadas):
            dE_da = camada.backward(dE_da)

    def atualizar(self, taxa):
        """Atualiza pesos e vieses de todas as camadas (depois de todos os deltas)."""
        for camada in self.camadas:
            camada.atualizar(taxa)

    def treinar_exemplo(self, x, y, taxa):
        """Um passo do SGD: forward → deltas → atualização, para um exemplo."""
        self.forward(x)
        self.retropropagar(y)
        self.atualizar(taxa)

    def prever(self, X):
        """Saída contínua da rede (ŷ), só com a propagação direta."""
        return self.forward(X)

    def classificar(self, X):
        """Classe final, a partir da saída contínua ŷ.

        - 1 neurônio de saída (XOR, AND): degrau, 1 se ŷ ≥ 0,5, senão 0.
        - Vários neurônios de saída, um por classe (Iris): o de maior ŷ
          vira 1 e os outros 0, para sempre sair exatamente uma classe.
        """
        y_hat = self.prever(X)
        if self.n_saidas == 1:
            return aplicar_degrau(y_hat)
        return maior_saida(y_hat)

    def resumo(self):
        """Linhas de texto descrevendo as camadas e a quantidade de pesos.

        Devolve uma lista de strings (quem imprime é o main.py).
        """
        linhas = [f"Entrada: {_neuronios(self.n_entradas)} (sem ativação)"]

        total_pesos = 0
        ultima = len(self.camadas) - 1
        for i, camada in enumerate(self.camadas):
            if i == ultima:
                titulo = "Saída"
                regra = "degrau" if self.n_saidas == 1 else "maior ŷ"
                if camada.nome_ativacao == "sigmoid":
                    ativacao = f"sigmoid no treino / {regra} na classificação"
                else:
                    ativacao = f"{camada.nome_ativacao} — {regra} na classificação"
            else:
                titulo = f"Oculta {i + 1}"
                ativacao = camada.nome_ativacao

            pesos = camada.numero_de_pesos()
            total_pesos += pesos
            linha = (
                f"{titulo}: {_neuronios(camada.n_neuronios)}, {ativacao} — "
                f"{camada.n_entradas} × {camada.n_neuronios} = {pesos} pesos"
            )
            if self.usar_bias:
                linha += f" (+{camada.n_neuronios} bias)"
            linhas.append(linha)

        linhas.append(f"Total de pesos: {total_pesos}")
        return linhas


def _neuronios(quantidade):
    """Texto '1 neurônio' ou 'N neurônios'."""
    return f"{quantidade} neurônio" if quantidade == 1 else f"{quantidade} neurônios"
