"""Classe RedeNeural: uma MLP montada a partir de várias Camadas."""

import numpy as np

from rede.ativacoes import classificar as aplicar_degrau
from rede.ativacoes import maior_saida
from rede.camada import Camada
from rede.perda import derivada_mse, mse


class RedeNeural:
    """Rede MLP: entrada → camadas ocultas (0 ou mais) → saída.

    A camada de entrada não é um objeto Camada: ela só repassa os valores,
    sem pesos e sem função de ativação.
    """

    # Ativação da camada de saída, fixa no código.
    # O degrau não tem derivada útil, então não dá para treinar com ele.
    # Por isso a saída usa sigmoid no treino, e o degrau (1 se ŷ ≥ 0,5,
    # senão 0) é aplicado só em classificar(), para gerar a classe final.
    ATIVACAO_SAIDA = "sigmoid"

    def __init__(self, n_entradas, ocultas, n_saidas, usar_bias, seed=None):
        """
        n_entradas: neurônios da camada de entrada (colunas de X)
        ocultas:    lista de (n_neuronios, nome_ativacao), uma por camada
                    oculta. Lista vazia = sem oculta (perceptron simples).
        n_saidas:   neurônios da camada de saída
        usar_bias:  vale para todas as camadas
        seed:       semente para repetir o experimento (None = aleatória)
        """
        self.n_entradas = n_entradas
        self.n_saidas = n_saidas
        self.usar_bias = usar_bias

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
            Camada(n_anterior, n_saidas, self.ATIVACAO_SAIDA, usar_bias, gerador)
        )

    def forward(self, X):
        """Propaga as entradas pela rede, camada por camada.

        X: forma (N, n_entradas). Uma amostra só, forma (n_entradas,),
           também é aceita e tratada como N = 1.
        Devolve Ŷ, forma (N, n_saidas).
        """
        A = np.atleast_2d(np.asarray(X, dtype=float))
        if A.shape[1] != self.n_entradas:
            raise ValueError(
                f"A rede espera {self.n_entradas} entradas por amostra, "
                f"mas recebeu {A.shape[1]}."
            )

        # A saída de uma camada é a entrada da próxima.
        for camada in self.camadas:
            A = camada.forward(A)
        return A

    def treinar(self, X, Y, epocas, taxa, ao_fim_da_epoca=None):
        """Treina a rede com gradiente descendente usando a base inteira.

        X:      entradas, forma (N, n_entradas)
        Y:      saídas esperadas, forma (N, n_saidas)
        epocas: quantas vezes a base inteira passa pela rede
        taxa:   taxa de aprendizado (η)
        ao_fim_da_epoca: função opcional chamada como f(epoca, erro) ao fim
                de cada época (o main.py usa para mostrar o progresso, já
                que esta pasta não imprime nada). A época começa em 1.

        Devolve a lista com o erro (MSE) de cada época. O erro guardado é o
        calculado ANTES do ajuste de pesos daquela época.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        Y = np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            # Vetor simples de saídas → uma coluna (caso de 1 neurônio de saída).
            Y = Y.reshape(-1, 1)
        if Y.shape != (X.shape[0], self.n_saidas):
            raise ValueError(
                f"Y deveria ter forma ({X.shape[0]}, {self.n_saidas}), "
                f"mas tem {Y.shape}."
            )

        historico = []
        for epoca in range(1, epocas + 1):
            # 1. Propagação para frente: calcula a saída da rede.
            Y_hat = self.forward(X)

            # 2. Mede o erro.
            erro = mse(Y, Y_hat)
            historico.append(erro)

            # 3. Derivada do erro em relação à saída: (2/N) · (ŷ − y).
            dA = derivada_mse(Y, Y_hat)

            # 4. Backpropagation: da saída para a entrada. Cada camada calcula
            #    seus gradientes e devolve o dA da camada anterior.
            for camada in reversed(self.camadas):
                dA = camada.backward(dA)

            # 5. Só depois de todos os gradientes prontos, ajusta os pesos.
            for camada in self.camadas:
                camada.atualizar(taxa)

            if ao_fim_da_epoca is not None:
                ao_fim_da_epoca(epoca, erro)

        return historico

    def prever(self, X):
        """Saída contínua da rede (ŷ entre 0 e 1, vinda da sigmoid)."""
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
                ativacao = f"{camada.nome_ativacao} no treino / {regra} na classificação"
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
