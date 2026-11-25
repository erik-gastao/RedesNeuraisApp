"""Classe Experimento: o fluxo completo do programa, sem input() nem print().

    base → dividir em treino/validação/teste → normalizar entradas (opcional)
         → treinar (SGD) → avaliar → prever / salvar

Usada pelo terminal (main.py) e pela interface gráfica, para as duas
fazerem exatamente as mesmas contas. Quem conversa com o usuário é quem
chama esta classe.
"""

from pathlib import Path

import numpy as np

from dados import (
    aplicar_normalizacao,
    carregar_csv,
    e_classificacao,
    normalizar,
    normalizar_por_lotes,
    separar_treino_validacao_teste,
)
from persistencia import carregar_rede, salvar_rede
from rede.perda import mse
from rede.rede_neural import RedeNeural
from rede.treinamento import treinar as treinar_rede

PERCENTUAIS_PADRAO = (70, 15, 15)


class Experimento:
    """Uma rede, a base em que ela treina e tudo o que foi feito com elas.

    Uso em fases (o terminal pergunta cada coisa numa hora):
        exp = Experimento(rede, X, Y, classes, "xor.csv", seed)
        exp.dividir((70, 15, 15))
        exp.normalizar_entradas()          # opcional
        exp.treinar(1000, 0.1)
        exp.avaliar()
    ou tudo de uma vez: Experimento.da_base(caminho, ...).
    """

    def __init__(self, rede, X, Y, classes, nome_base, seed=None):
        """
        rede:      RedeNeural já montada (entradas e saídas iguais às da base)
        X, Y:      base inteira, formas (N, n_entradas) e (N, n_saidas)
        classes:   nomes das classes (base com classe em texto) ou None
        nome_base: nome do arquivo, para títulos e para o nome ao salvar
        seed:      semente da divisão da base e da ordem dos exemplos no treino
        """
        if X.shape[1] != rede.n_entradas or Y.shape[1] != rede.n_saidas:
            raise ValueError(
                f"A base tem {X.shape[1]} entrada(s) e {Y.shape[1]} saída(s), mas a "
                f"rede tem {rede.n_entradas} e {rede.n_saidas}."
            )
        self._comecar(rede, classes, nome_base, seed)
        self.X, self.Y = X, Y
        self.classificacao = e_classificacao(Y, classes)

    def _comecar(self, rede, classes, nome_base, seed):
        """Atributos comuns a um experimento novo e a uma rede carregada."""
        self.rede = rede
        self.classes = classes
        self.nome_base = nome_base
        self.seed = seed
        self.X = self.Y = None
        self.separou = False
        self.X_treino = self.Y_treino = self.X_treino_original = None
        self.X_validacao = self.Y_validacao = self.X_validacao_original = None
        self.X_teste = self.Y_teste = self.X_teste_original = None
        self.normalizacao = None
        self.tamanho_lote = None
        self.historico = {"treino": [], "validacao": []}

    @classmethod
    def da_base(cls, caminho, ocultas, ativacao_saida, usar_bias, seed, n_saidas=1,
                percentuais=PERCENTUAIS_PADRAO, normalizar=True, tamanho_lote=None):
        """Carrega a base, monta a rede com as entradas/saídas dela, divide e normaliza.

        n_saidas: colunas de saída numéricas (com classe em texto, tem que ser
                  o número de classes).
        """
        X, Y, classes = carregar_csv(caminho, n_saidas)
        rede = RedeNeural(X.shape[1], ocultas, Y.shape[1], usar_bias, seed, ativacao_saida)
        experimento = cls(rede, X, Y, classes, Path(caminho).name, seed)
        experimento.dividir(percentuais)
        if normalizar:
            experimento.normalizar_entradas(tamanho_lote)
        return experimento

    @classmethod
    def carregar(cls, caminho):
        """Rede salva em JSON, sem base: serve só para prever (modo previsão)."""
        rede, normalizacao, classes = carregar_rede(caminho)
        experimento = cls.__new__(cls)
        experimento._comecar(rede, classes, Path(caminho).name, None)
        experimento.normalizacao = normalizacao
        # Sem a base, é classificação quando há nomes de classes ou saída sigmoid.
        experimento.classificacao = classes is not None or rede.ativacao_saida == "sigmoid"
        return experimento

    # ------------------------------------------------------------ preparo

    def dividir(self, percentuais=PERCENTUAIS_PADRAO):
        """Divide a base em treino, validação e teste (ValueError se inválido).

        Base com menos de 10 linhas não é dividida: treino e teste são a base
        inteira e não há validação (seria igual ao treino).
        """
        X_tr, Y_tr, X_va, Y_va, X_te, Y_te, separou = separar_treino_validacao_teste(
            self.X, self.Y, percentuais, seed=self.seed
        )
        if not separou:
            X_va = Y_va = None
        self.separou = separou
        self.X_treino = self.X_treino_original = X_tr
        self.Y_treino = Y_tr
        self.X_validacao = self.X_validacao_original = X_va
        self.Y_validacao = Y_va
        self.X_teste = self.X_teste_original = X_te
        self.Y_teste = Y_te
        self.normalizacao = None
        self.tamanho_lote = None

    def normalizar_entradas(self, tamanho_lote=None):
        """Min-max das entradas com o mínimo e o máximo do treino (seção 1.9).

        A mesma escala vale para validação, teste e entradas digitadas, e é
        salva com a rede. As saídas esperadas nunca são normalizadas.
        Com `tamanho_lote` menor que o treino, só o treino é normalizado lote
        a lote (opção extra deste projeto, fora do padrão do enunciado).
        """
        X_treino, minimos, maximos = normalizar(self.X_treino_original)
        self.normalizacao = {"funcao": "min-max", "minimos": minimos.tolist(),
                             "maximos": maximos.tolist()}
        self.tamanho_lote = None
        if tamanho_lote is not None and tamanho_lote < len(X_treino):
            X_treino = normalizar_por_lotes(self.X_treino_original, tamanho_lote)
            self.tamanho_lote = tamanho_lote
        self.X_treino = X_treino
        self.X_teste = aplicar_normalizacao(self.X_teste_original, minimos, maximos)
        if self.X_validacao_original is not None:
            self.X_validacao = aplicar_normalizacao(self.X_validacao_original, minimos, maximos)

    # ------------------------------------------------------- treino e uso

    def treinar(self, epocas, taxa, ao_fim_da_epoca=None):
        """Treina com SGD e acrescenta os erros ao histórico.

        ao_fim_da_epoca: f(epoca, erro_treino, erro_validacao), como em
        rede.treinamento.treinar. Devolve só os erros deste treino.
        """
        novo = treinar_rede(
            self.rede, self.X_treino, self.Y_treino, epocas, taxa,
            X_validacao=self.X_validacao, Y_validacao=self.Y_validacao,
            seed=self.seed, ao_fim_da_epoca=ao_fim_da_epoca,
        )
        self.historico["treino"] += novo["treino"]
        self.historico["validacao"] += novo["validacao"]
        return novo

    def avaliar(self):
        """MSE dos três conjuntos e os resultados do teste, linha a linha.

        Devolve um dicionário com mse_treino, mse_validacao, mse_teste (None
        quando o conjunto não existe), classificacao, acuracia e acertos (None
        na regressão) e linhas: uma por amostra de teste, com entrada (escala
        original), esperado e y_hat; mais classificado e acertou na
        classificação, ou erro (ŷ − esperado) na regressão.
        """
        resultado = {
            "mse_treino": mse(self.Y_treino, self.rede.prever(self.X_treino)),
            "mse_validacao": None,
            "mse_teste": None,
            "classificacao": self.classificacao,
            "acuracia": None,
            "acertos": None,
            "linhas": [],
        }
        if self.X_validacao is not None and len(self.X_validacao):
            resultado["mse_validacao"] = mse(self.Y_validacao, self.rede.prever(self.X_validacao))
        if len(self.Y_teste) == 0:
            return resultado

        Y_hat = self.rede.prever(self.X_teste)
        resultado["mse_teste"] = mse(self.Y_teste, Y_hat)
        if self.classificacao:
            classificado = self.rede.classificar(self.X_teste)
            acertou = np.all(classificado == self.Y_teste, axis=1)
            resultado["acuracia"] = float(np.mean(acertou))
            resultado["acertos"] = int(acertou.sum())
            for i in range(len(self.Y_teste)):
                resultado["linhas"].append({
                    "entrada": self.X_teste_original[i], "esperado": self.Y_teste[i],
                    "y_hat": Y_hat[i], "classificado": classificado[i],
                    "acertou": bool(acertou[i]),
                })
        else:
            for i in range(len(self.Y_teste)):
                resultado["linhas"].append({
                    "entrada": self.X_teste_original[i], "esperado": self.Y_teste[i],
                    "y_hat": Y_hat[i], "erro": Y_hat[i] - self.Y_teste[i],
                })
        return resultado

    def prever_entrada(self, valores):
        """Previsão (seção 1.12): só a propagação direta, nenhum peso muda.

        valores: entrada na escala original da base. Recebe a mesma
        normalização do treino, se houve. Devolve (ŷ, classe), com a classe em
        texto na classificação e None na regressão.
        """
        x = np.asarray(valores, dtype=float).reshape(1, -1)
        if x.shape[1] != self.rede.n_entradas:
            raise ValueError(
                f"A rede espera {self.rede.n_entradas} valor(es), mas recebeu {x.shape[1]}."
            )
        if self.normalizacao is not None:
            x = aplicar_normalizacao(x, np.array(self.normalizacao["minimos"]),
                                     np.array(self.normalizacao["maximos"]))
        y_hat = self.rede.prever(x)[0]
        classe = nome_da_classe(self.rede.classificar(x)[0], self.classes) if self.classificacao else None
        return y_hat, classe

    def salvar(self, caminho):
        """Salva a rede, a normalização e as classes em JSON (seção 1.11)."""
        return salvar_rede(caminho, self.rede, self.normalizacao, self.classes)

    def reiniciar(self):
        """Volta a rede aos pesos iniciais (mesma semente) e zera o histórico."""
        ocultas = [(c.n_neuronios, c.nome_ativacao) for c in self.rede.camadas[:-1]]
        self.rede = RedeNeural(self.rede.n_entradas, ocultas, self.rede.n_saidas,
                               self.rede.usar_bias, self.seed, self.rede.ativacao_saida)
        self.historico = {"treino": [], "validacao": []}


def nome_da_classe(valores, classes):
    """Saída 0/1 → texto. Com nomes de classes, o nome da classe marcada com 1
    (ex.: 'Iris-setosa'); sem nomes, os números (ex.: '1')."""
    if classes is None:
        return " ".join(f"{valor:g}" for valor in valores)
    return classes[int(np.argmax(valores))]
