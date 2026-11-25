"""Gravador do player: roda o treino SGD real e grava os passos da animação.

"Gravar primeiro, tocar depois": o player só desenha o passo i da lista,
então voltar, pular e parar são simples. Nada é recalculado na interface —
os números vêm do que os neurônios guardam (entradas, z, a, delta) durante o
treino de verdade.

O laço é o mesmo de rede/treinamento.treinar (mesma semente, mesma ordem dos
exemplos), então a rede termina com exatamente os mesmos pesos.
"""

from dataclasses import dataclass

import numpy as np

from interface import formulas
from interface.passos import Passo
from rede.ativacoes import ATIVACOES
from rede.perda import derivada_mse, mse

NIVEIS = ("ligacao", "neuronio", "camada")


@dataclass
class Gravacao:
    """Resultado do gravador.

    passos:  lista de Passo, na ordem em que o player toca
    resumos: um por época: {"epoca", "pesos" [(W, b)], "erro_treino",
             "erro_validacao" (ou None), "acuracia" (do treino; None na regressão)}
    """

    passos: list
    resumos: list
    epocas_detalhadas: list
    exemplo_em_foco: int
    nivel: str
    taxa: float


def gravar(experimento, epocas, taxa, epocas_detalhadas, nivel="ligacao",
           exemplo_em_foco=0, ao_fim_da_epoca=None):
    """Treina a rede do experimento com SGD e grava a animação.

    epocas_detalhadas: épocas com passos (números e/ou "fim" = última)
    nivel:             "ligacao", "neuronio" ou "camada" (detalhe dos passos)
    exemplo_em_foco:   índice da amostra do treino mostrada em detalhe; os
                       outros exemplos das épocas detalhadas viram passos compactos
    ao_fim_da_epoca:   função opcional f(epoca), para barra de progresso

    Também acrescenta os erros ao histórico do experimento, como
    experimento.treinar faria.
    """
    if nivel not in NIVEIS:
        raise ValueError(f"Nível desconhecido: '{nivel}'. Opções: {', '.join(NIVEIS)}.")
    X, Y = experimento.X_treino, experimento.Y_treino
    N = len(X)
    if not 0 <= exemplo_em_foco < N:
        raise ValueError(f"O exemplo em foco precisa estar entre 0 e {N - 1}.")
    detalhadas = sorted({epocas if e == "fim" else int(e) for e in epocas_detalhadas
                         if e == "fim" or 1 <= int(e) <= epocas})

    rede = experimento.rede
    passos = []
    if experimento.normalizacao is not None:
        passos += _passos_normalizacao(experimento, exemplo_em_foco, _copiar_pesos(rede))

    tem_validacao = experimento.X_validacao is not None and len(experimento.X_validacao) > 0
    gerador = np.random.default_rng(experimento.seed)
    resumos = []
    for epoca in range(1, epocas + 1):
        detalhada = epoca in detalhadas
        for posicao, i in enumerate(gerador.permutation(N)):
            if not detalhada:
                rede.treinar_exemplo(X[i], Y[i], taxa)
            elif i == exemplo_em_foco:
                registro = _registrar_exemplo(rede, X[i], Y[i], taxa)
                passos += _passos_exemplo(registro, nivel, epoca, posicao, int(i), taxa)
            else:
                rede.treinar_exemplo(X[i], Y[i], taxa)
                passos.append(_passo_compacto(rede, X[i], Y[i], epoca, posicao, int(i)))

        erro_treino = mse(Y, rede.forward(X))
        erro_validacao = None
        if tem_validacao:
            erro_validacao = mse(experimento.Y_validacao, rede.forward(experimento.X_validacao))
            experimento.historico["validacao"].append(erro_validacao)
        experimento.historico["treino"].append(erro_treino)
        acuracia = None
        if experimento.classificacao:
            acuracia = float(np.mean(np.all(rede.classificar(X) == Y, axis=1)))
        resumos.append({"epoca": epoca, "pesos": _copiar_pesos(rede), "erro_treino": erro_treino,
                        "erro_validacao": erro_validacao, "acuracia": acuracia})
        if ao_fim_da_epoca is not None:
            ao_fim_da_epoca(epoca)

    return Gravacao(passos, resumos, detalhadas, exemplo_em_foco, nivel, taxa)


def passos_previsao(rede, x, nivel="ligacao"):
    """Passos só de forward para uma entrada (já normalizada, se for o caso).

    Usado em "Prever entrada": nenhum peso muda.
    """
    pesos = _copiar_pesos(rede)
    rede.forward(x)
    return _passos_forward(_ler_camadas(rede), pesos, nivel, 0, None, np.asarray(x, dtype=float))


# ------------------------------------------------------------- registro

def _copiar_pesos(rede):
    return [(camada.W, camada.b) for camada in rede.camadas]


def _ler_camadas(rede):
    """Copia o que cada camada guardou no último forward."""
    return [
        {
            "ativacao": camada.nome_ativacao,
            "entradas": camada.neuronios[0].entradas.copy(),
            "z": [n.z for n in camada.neuronios],
            "a": [n.a for n in camada.neuronios],
        }
        for camada in rede.camadas
    ]


def _registrar_exemplo(rede, x, y, taxa):
    """Um passo de SGD em partes, copiando o estado entre elas."""
    pesos_antes = _copiar_pesos(rede)
    rede.forward(x)
    camadas = _ler_camadas(rede)
    rede.retropropagar(y)
    for registro, camada in zip(camadas, rede.camadas):
        registro["delta"] = [n.delta for n in camada.neuronios]
    rede.atualizar(taxa)
    return {"x": np.asarray(x, dtype=float), "y": np.asarray(y, dtype=float),
            "camadas": camadas, "pesos_antes": pesos_antes, "pesos_depois": _copiar_pesos(rede)}


# ---------------------------------------------------------------- passos

def _passo(epoca, exemplo, fase, tipo, camada, neuronio, origem, valores, textos, texto, pesos):
    formula, formula_numeros = textos
    return Passo(epoca, exemplo, fase, tipo, camada, neuronio, origem, valores,
                 formula, formula_numeros, texto, pesos)


def _passos_normalizacao(experimento, foco, pesos):
    minimos = experimento.normalizacao["minimos"]
    maximos = experimento.normalizacao["maximos"]
    texto = "Cada entrada vai para a faixa de 0 a 1 com o mínimo e o máximo do treino."
    if experimento.tamanho_lote is not None:
        texto = (f"Normalização por lotes de {experimento.tamanho_lote}: este exemplo "
                 "usa a escala do lote dele (a fórmula mostra a do treino inteiro).")
    passos = []
    for j, original in enumerate(experimento.X_treino_original[foco]):
        normalizado = float(experimento.X_treino[foco][j])
        valores = {"original": float(original), "minimo": minimos[j], "maximo": maximos[j],
                   "normalizado": normalizado}
        textos = formulas.formula_normalizacao(j, original, minimos[j], maximos[j], normalizado)
        passos.append(_passo(0, None, "normalizacao", "entrada", 0, j, None, valores,
                             textos, texto, pesos))
    return passos


def _passos_exemplo(registro, nivel, epoca, exemplo, indice, taxa):
    pesos = registro["pesos_antes"]
    passos = _passos_forward(registro["camadas"], pesos, nivel, epoca, exemplo, registro["x"], indice)

    y, y_hat = registro["y"], np.array(registro["camadas"][-1]["a"])
    dE_dy_hat = derivada_mse(y.reshape(1, -1), y_hat.reshape(1, -1))[0]
    erro = mse(y.reshape(1, -1), y_hat.reshape(1, -1))
    n_camadas = len(registro["camadas"])
    passos.append(_passo(
        epoca, exemplo, "erro", "erro_saida", n_camadas, None, None,
        {"y": y.tolist(), "y_hat": y_hat.tolist(), "dE_dy_hat": dE_dy_hat.tolist(), "erro": erro},
        formulas.formula_erro(y, y_hat, erro),
        "A saída é comparada com o esperado. ∂E/∂ŷ = 2(ŷ − y) começa a retropropagação.",
        pesos,
    ))

    passos += _passos_retropropagacao(registro, nivel, epoca, exemplo, dE_dy_hat)
    passos += _passos_ajuste(registro, epoca, exemplo, taxa)
    return passos


def _passos_forward(camadas, pesos, nivel, epoca, exemplo, x, indice=None):
    valores = {"x": x.tolist()}
    if indice is not None:
        valores["indice"] = indice
    passos = [_passo(epoca, exemplo, "forward", "entrada_rede", 0, None, None, valores,
                     formulas.formula_entrada(x),
                     "A amostra entra na rede: a camada de entrada só repassa os valores.", pesos)]

    for numero_camada, registro in enumerate(camadas, start=1):
        W, b = pesos[numero_camada - 1]
        entradas, nome = registro["entradas"], registro["ativacao"]
        if nivel == "camada":
            passos.append(_passo(
                epoca, exemplo, "forward", "camada_forward", numero_camada, None, None,
                {"z": list(registro["z"]), "a": list(registro["a"])},
                formulas.formula_camada(nome, registro["a"]),
                f"A camada {numero_camada} calcula a saída de todos os seus neurônios.", pesos,
            ))
            continue

        for n, (z, a) in enumerate(zip(registro["z"], registro["a"])):
            if nivel == "neuronio":
                passos.append(_passo(
                    epoca, exemplo, "forward", "ativacao", numero_camada, n, None,
                    {"z": z, "a": a, "b": float(b[n])},
                    formulas.formula_neuronio(n, nome, W[n], entradas, b[n], z, a),
                    "O neurônio soma as entradas pesadas, soma o bias e aplica a ativação.",
                    pesos,
                ))
                continue

            produtos = W[n] * entradas
            for j in range(len(entradas)):
                passos.append(_passo(
                    epoca, exemplo, "forward", "ligacao", numero_camada, n, j,
                    {"w": float(W[n, j]), "a": float(entradas[j]), "produto": float(produtos[j])},
                    formulas.formula_ligacao(j, W[n, j], entradas[j]),
                    "Cada ligação multiplica a saída do neurônio anterior pelo seu peso.", pesos,
                ))
            soma = float(produtos.sum())
            passos.append(_passo(
                epoca, exemplo, "forward", "soma", numero_camada, n, None, {"soma": soma},
                formulas.formula_soma(n, produtos, soma),
                "O neurônio soma as entradas pesadas.", pesos,
            ))
            passos.append(_passo(
                epoca, exemplo, "forward", "bias", numero_camada, n, None,
                {"soma": soma, "b": float(b[n]), "z": z},
                formulas.formula_bias(n, soma, b[n], z),
                "Soma o bias e obtém z, a soma ponderada.", pesos,
            ))
            passos.append(_passo(
                epoca, exemplo, "forward", "ativacao", numero_camada, n, None, {"z": z, "a": a},
                formulas.formula_ativacao(n, nome, z, a),
                f"A ativação {nome} transforma z na saída a do neurônio.", pesos,
            ))
    return passos


def _passos_retropropagacao(registro, nivel, epoca, exemplo, dE_dy_hat):
    pesos = registro["pesos_antes"]
    camadas = registro["camadas"]
    passos = []
    dE_da = np.asarray(dE_dy_hat, dtype=float)
    for numero_camada in range(len(camadas), 0, -1):
        registro_camada = camadas[numero_camada - 1]
        deltas = np.array(registro_camada["delta"])
        derivada = ATIVACOES[registro_camada["ativacao"]][1]
        saida = numero_camada == len(camadas)

        if nivel == "camada":
            passos.append(_passo(
                epoca, exemplo, "retropropagacao", "camada_delta", numero_camada, None, None,
                {"delta": deltas.tolist()}, formulas.formula_camada_delta(deltas),
                f"A camada {numero_camada} calcula o delta (δ = ∂E/∂z) de todos os neurônios.",
                pesos,
            ))
        else:
            for n, (z, a) in enumerate(zip(registro_camada["z"], registro_camada["a"])):
                f_linha = float(derivada(np.array(z), np.array(a)))
                valores = {"dE_da": float(dE_da[n]), "derivada": f_linha, "delta": float(deltas[n])}
                if saida:
                    textos = formulas.formula_delta_saida(n, dE_da[n], f_linha, deltas[n])
                    texto = "Na saída, δ = (quanto o erro muda com ŷ) × f'(z)."
                else:
                    textos = formulas.formula_delta_oculta(n, dE_da[n], f_linha, deltas[n])
                    texto = ("Na oculta, o δ vem da camada seguinte pelos pesos de antes "
                             "do ajuste: δ = f'(z) · Σ δ·w.")
                passos.append(_passo(epoca, exemplo, "retropropagacao", "delta",
                                     numero_camada, n, None, valores, textos, texto, pesos))
                if nivel == "ligacao":
                    for j, entrada in enumerate(registro_camada["entradas"]):
                        gradiente = float(deltas[n] * entrada)
                        passos.append(_passo(
                            epoca, exemplo, "retropropagacao", "gradiente", numero_camada, n, j,
                            {"delta": float(deltas[n]), "a": float(entrada), "gradiente": gradiente},
                            formulas.formula_gradiente(j, deltas[n], entrada, gradiente),
                            "O gradiente do peso é o δ do neurônio vezes a entrada da ligação.",
                            pesos,
                        ))
        # ∂E/∂a da camada anterior, como Camada.backward: Σ δ·w (pesos de antes).
        dE_da = deltas @ pesos[numero_camada - 1][0]
    return passos


def _passos_ajuste(registro, epoca, exemplo, taxa):
    antes, depois = registro["pesos_antes"], registro["pesos_depois"]
    atuais = list(antes)
    passos = []
    for indice, camada in enumerate(registro["camadas"]):
        atuais = list(atuais)
        atuais[indice] = depois[indice]
        valores = {
            "W_antes": antes[indice][0], "W_depois": depois[indice][0],
            "b_antes": antes[indice][1], "b_depois": depois[indice][1],
            "deltas": np.array(camada["delta"]), "entradas": camada["entradas"], "taxa": taxa,
        }
        passos.append(_passo(
            epoca, exemplo, "atualizacao", "ajuste", indice + 1, None, None, valores,
            formulas.formula_ajuste(taxa),
            "Com todos os deltas prontos, os pesos da camada andam contra o gradiente.", atuais,
        ))
    return passos


def _passo_compacto(rede, x, y, epoca, exemplo, indice):
    """Exemplo fora de foco: já treinado; os neurônios guardam o ŷ de antes do ajuste."""
    y = np.asarray(y, dtype=float)
    y_hat = np.array([n.a for n in rede.camadas[-1].neuronios])
    erro = mse(y.reshape(1, -1), y_hat.reshape(1, -1))
    return _passo(
        epoca, exemplo, "exemplo", "exemplo_compacto", None, None, None,
        {"indice": indice, "x": np.asarray(x, dtype=float).tolist(), "y": y.tolist(),
         "y_hat": y_hat.tolist(), "erro": erro},
        formulas.formula_compacto(y, y_hat, erro),
        f"Exemplo {indice + 1}: forward, retropropagação e ajuste de uma vez (SGD).",
        _copiar_pesos(rede),
    )
