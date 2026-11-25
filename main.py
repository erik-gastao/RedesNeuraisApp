"""Construtor de Redes Neurais — programa de terminal.

Este é o único arquivo que conversa com o usuário (input e print).
A rede em si fica na pasta rede/ e a preparação da base em dados.py.

Rodar com:  python main.py
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from dados import (
    acuracia,
    aplicar_normalizacao,
    carregar_csv,
    desnormalizar,
    listar_bases,
    normalizar,
    normalizar_por_lotes,
    separar_treino_teste,
)
from graficos import grafico_do_erro
from rede.ativacoes import ATIVACOES
from rede.perda import mse
from rede.rede_neural import RedeNeural

# Pastas cujos arquivos aparecem no menu de bases (ficam ao lado deste arquivo).
PASTA_DO_PROJETO = Path(__file__).resolve().parent
PASTAS_DE_BASES = [PASTA_DO_PROJETO / "bases", PASTA_DO_PROJETO / "Datasets"]

# Pasta onde os gráficos do erro são salvos.
PASTA_DE_GRAFICOS = PASTA_DO_PROJETO / "graficos"

# Máximo de linhas mostradas na tabela de avaliação (bases grandes).
MAXIMO_LINHAS_TABELA = 20


# ------------------------------------------------------------- perguntas
# Todas repetem a pergunta até receber um valor válido: o programa nunca
# quebra por causa de algo digitado errado.

def perguntar_inteiro(msg, minimo, padrao=None, maximo=None):
    """Pede um inteiro ≥ minimo (e ≤ maximo, se dado).

    Com `padrao`, Enter vazio devolve o padrão.
    """
    while True:
        texto = input(msg).strip()
        if texto == "" and padrao is not None:
            return padrao
        try:
            valor = int(texto)
        except ValueError:
            print("  Valor inválido: digite um número inteiro.")
            continue
        if valor < minimo:
            print(f"  Valor inválido: o mínimo é {minimo}.")
            continue
        if maximo is not None and valor > maximo:
            print(f"  Valor inválido: o máximo é {maximo}.")
            continue
        return valor


def perguntar_float(msg, maior_que=None, padrao=None):
    """Pede um número real. Aceita vírgula ou ponto como separador decimal.

    Com `maior_que`, o valor precisa ser maior que esse limite.
    Com `padrao`, Enter vazio devolve o padrão.
    """
    while True:
        texto = input(msg).strip().replace(",", ".")
        if texto == "" and padrao is not None:
            return padrao
        try:
            valor = float(texto)
        except ValueError:
            print("  Valor inválido: digite um número (ex.: 0.5).")
            continue
        if not np.isfinite(valor):
            print("  Valor inválido: digite um número (ex.: 0.5).")
            continue
        if maior_que is not None and valor <= maior_que:
            print(f"  Valor inválido: precisa ser maior que {maior_que}.")
            continue
        return valor


def perguntar_sn(msg):
    """Pede s ou n. Devolve True para s, False para n."""
    while True:
        texto = input(msg).strip().lower()
        if texto in ("s", "sim"):
            return True
        if texto in ("n", "nao", "não"):
            return False
        print("  Resposta inválida: digite s ou n.")


def perguntar_opcao(msg, opcoes):
    """Pede uma das opções da lista (texto). Devolve a opção escolhida."""
    while True:
        texto = input(msg).strip().lower()
        if texto in opcoes:
            return texto
        print(f"  Opção inválida: escolha entre {', '.join(opcoes)}.")


def perguntar_semente():
    """Pede a semente aleatória. Enter vazio devolve None (aleatória)."""
    while True:
        texto = input("Semente aleatória (Enter = aleatória): ").strip()
        if texto == "":
            return None
        try:
            valor = int(texto)
        except ValueError:
            print("  Valor inválido: digite um inteiro ou só Enter.")
            continue
        if valor < 0:
            print("  Valor inválido: a semente não pode ser negativa.")
            continue
        return valor


# ------------------------------------------------------- montagem da rede

def montar_rede():
    """Faz as perguntas da arquitetura e devolve (rede, semente)."""
    n_entradas = perguntar_inteiro("Quantos neurônios na camada de entrada? ", 1)
    n_ocultas = perguntar_inteiro("Quantas camadas ocultas? ", 0)

    nomes_ativacoes = list(ATIVACOES)
    ocultas = []
    for i in range(1, n_ocultas + 1):
        neuronios = perguntar_inteiro(f"  Camada oculta {i} — quantos neurônios? ", 1)
        ativacao = perguntar_opcao(
            f"  Camada oculta {i} — ativação ({'/'.join(nomes_ativacoes)})? ",
            nomes_ativacoes,
        )
        ocultas.append((neuronios, ativacao))

    n_saidas = perguntar_inteiro("Quantos neurônios na camada de saída? ", 1)
    usar_bias = perguntar_sn("Usar bias? (s/n) ")
    semente = perguntar_semente()

    rede = RedeNeural(n_entradas, ocultas, n_saidas, usar_bias, semente)

    print()
    print("Rede montada:")
    for linha in rede.resumo():
        print(f"  {linha}")
    print()
    return rede, semente


# ---------------------------------------------------------- base de dados

def carregar_base(rede):
    """Mostra as bases disponíveis e carrega a escolhida.

    O usuário pode digitar o número de uma base da lista ou o caminho de
    qualquer outro arquivo. Repete até carregar uma base compatível com a
    rede.

    Devolve (X, Y, classes, nome_da_base), ou None se o usuário desistir
    com Enter vazio. `classes` são os nomes das classes quando a base traz
    a classe em texto (ex.: Iris); senão é None.
    """
    bases = listar_bases(PASTAS_DE_BASES)
    if bases:
        print("Bases disponíveis:")
        for i, arquivo in enumerate(bases, start=1):
            print(f"  {i}) {arquivo.name}  ({arquivo.parent.name}/)")

    while True:
        texto = input(
            "Base de dados — número da lista ou caminho do arquivo "
            "(Enter = cancelar e montar outra rede): "
        ).strip().strip('"')
        if texto == "":
            return None

        if texto.isdigit():
            if not bases:
                print("  Nenhuma base na lista: digite o caminho de um arquivo.")
                continue
            if not 1 <= int(texto) <= len(bases):
                print(f"  Opção inválida: escolha um número de 1 a {len(bases)}, "
                      "ou digite o caminho de um arquivo.")
                continue
            caminho = bases[int(texto) - 1]
        else:
            caminho = texto

        try:
            X, Y, classes = carregar_csv(caminho, rede.n_saidas)
        except (OSError, ValueError) as erro:
            print(f"  Não foi possível carregar a base: {erro}")
            continue

        # Colunas do arquivo = entradas + saídas. Com a classe em texto
        # (Iris), a saída é 1 coluna no arquivo, mesmo virando várias em Y.
        if X.shape[1] != rede.n_entradas:
            colunas_saida = 1 if classes is not None else rede.n_saidas
            print(
                f"  Base incompatível: a rede tem {rede.n_entradas} entrada(s) e "
                f"{rede.n_saidas} saída(s), então a base precisa de "
                f"{rede.n_entradas + colunas_saida} colunas, mas tem "
                f"{X.shape[1] + colunas_saida}."
            )
            continue

        print(f"  Base carregada: {X.shape[0]} amostras.")
        if classes is not None:
            print(f"  Classes (uma por neurônio de saída): {', '.join(classes)}")
        if not np.all((Y == 0) | (Y == 1)):
            print(
                "  Atenção: a saída esperada tem valores diferentes de 0 e 1. "
                "A rede classifica só em 0 ou 1, então a acurácia pode ficar baixa."
            )
        return X, Y, classes, Path(caminho).name


# ----------------------------------------------------------------- treino

def treinar(rede, X, Y, epocas, taxa):
    """Treina mostrando o erro a cada ~10% das épocas.

    Devolve a lista com o erro de cada época (usada no gráfico).
    """
    intervalo = max(1, epocas // 10)
    largura = len(str(epocas))

    # A pasta rede/ não imprime nada: ela chama esta função ao fim de cada
    # época e é aqui que o progresso aparece.
    def mostrar_progresso(epoca, erro):
        if epoca % intervalo == 0 or epoca == epocas:
            print(f"  Época {epoca:>{largura}}/{epocas} — erro (MSE): {erro:.4f}")

    print()
    print("Treinando...")
    return rede.treinar(X, Y, epocas, taxa, ao_fim_da_epoca=mostrar_progresso)


# -------------------------------------------------------------- avaliação

def formatar_valores(valores):
    """Vetor de números → texto curto, ex.: '0 1' ou '5.1 3.5'."""
    return " ".join(f"{valor:g}" for valor in valores)


def formatar_porcentagem(fracao):
    """0.75 → '75,00%' (vírgula decimal, como no português)."""
    return f"{fracao * 100:.2f}%".replace(".", ",")


def formatar_saida(valores, classes):
    """Saída 0/1 → texto. Com nomes de classes, mostra o nome da classe
    marcada com 1 (ex.: 'Iris-setosa'); sem nomes, mostra os números."""
    if classes is None:
        return formatar_valores(valores)
    return classes[int(np.argmax(valores))]


def avaliar(rede, X_norm, X_original, Y_norm, Y_original, escala_y, titulo, classes):
    """Mostra erro, acurácia e a tabela entrada | esperado | ŷ | classe | ok?.

    X_norm e Y_norm estão na escala [0, 1], a mesma do treino: o erro e a
    acurácia são calculados nela. X_original e Y_original aparecem na
    tabela, e ŷ é desnormalizado com escala_y = (minimos, maximos) de Y.
    """
    Y_hat = rede.prever(X_norm)
    classificado = rede.classificar(X_norm)
    acertou = np.all(classificado == Y_norm, axis=1)
    Y_hat_original = desnormalizar(Y_hat, *escala_y)
    classificado_original = desnormalizar(classificado, *escala_y)

    print()
    print(f"Avaliação ({titulo}):")
    print(f"  Erro final (MSE): {mse(Y_norm, Y_hat):.4f}")
    print(
        f"  Acurácia: {formatar_porcentagem(acuracia(Y_norm, classificado))} "
        f"({int(acertou.sum())}/{len(acertou)})"
    )

    n_linhas = min(len(Y_norm), MAXIMO_LINHAS_TABELA)
    linhas = [("entrada", "esperado", "ŷ", "classe", "ok?")]
    for i in range(n_linhas):
        linhas.append((
            formatar_valores(X_original[i]),
            formatar_saida(Y_original[i], classes),
            " ".join(f"{valor:.4f}" for valor in Y_hat_original[i]),
            formatar_saida(classificado_original[i], classes),
            "sim" if acertou[i] else "NÃO",
        ))

    # Largura de cada coluna = maior texto dela.
    larguras = [max(len(linha[c]) for linha in linhas) for c in range(5)]
    print()
    for linha in linhas:
        print("  " + " | ".join(texto.ljust(larguras[c]) for c, texto in enumerate(linha)))
    if len(Y_norm) > n_linhas:
        print(f"  ... (mostrando {n_linhas} de {len(Y_norm)} amostras)")


# ------------------------------------------------------------- menu final

def testar_entrada_digitada(rede, escala_x, escala_y, classes):
    """Pede uma entrada, passa pela rede e mostra ŷ e a classe.

    escala_x e escala_y são os (minimos, maximos) do treino: a entrada é
    normalizada com escala_x e ŷ volta para a escala original com escala_y.
    """
    valores = [
        perguntar_float(f"  Valor da entrada {i}: ")
        for i in range(1, rede.n_entradas + 1)
    ]
    # Mesma escala usada no treino.
    x = aplicar_normalizacao(np.array([valores]), *escala_x)
    y_hat = desnormalizar(rede.prever(x), *escala_y)[0]
    classe = desnormalizar(rede.classificar(x), *escala_y)[0]
    print(f"  ŷ = {' '.join(f'{valor:.4f}' for valor in y_hat)}"
          f"  →  classe: {formatar_saida(classe, classes)}")


def mostrar_grafico(historico, nome_da_base, taxa):
    """Salva o gráfico do erro em graficos/ e abre a janela com ele."""
    nome_arquivo = f"erro_{Path(nome_da_base).stem}_{datetime.now():%Y%m%d_%H%M%S}.png"
    titulo = f"Erro por época — {nome_da_base} (η = {taxa:g}, {len(historico)} épocas)"
    print("  Abrindo o gráfico. Feche a janela dele para voltar ao menu.")
    try:
        caminho = grafico_do_erro(historico, PASTA_DE_GRAFICOS / nome_arquivo, titulo)
    except ImportError:
        print("  O gráfico precisa do matplotlib. Instale com: pip install matplotlib")
        return
    print(f"  Gráfico salvo em: {caminho}")


def perguntar_lotes(X_treino_original, X_treino):
    """Pergunta se as entradas do treino devem ser normalizadas por lotes.

    Enter mantém a normalização da base inteira (X_treino, já pronto).
    Com um tamanho menor, cada lote de registros é normalizado com o
    mínimo e o máximo dele mesmo (ver dados.normalizar_por_lotes).
    Devolve o X do treino que a rede vai usar.
    """
    N = len(X_treino_original)
    tamanho = perguntar_inteiro(
        f"Tamanho do lote para normalizar (2 a {N}, Enter = base inteira): ",
        2, padrao=N, maximo=N,
    )
    if tamanho == N:
        return X_treino
    # Arredonda para cima; uma sobra de 1 registro entra no lote anterior.
    n_lotes = -(-N // tamanho) - (1 if N % tamanho == 1 else 0)
    print(
        f"  Entradas do treino normalizadas em {n_lotes} lote(s) de "
        f"~{tamanho} registros, cada um com sua própria escala. O teste e "
        "as entradas digitadas usam a escala da base de treino inteira."
    )
    return normalizar_por_lotes(X_treino_original, tamanho)


def sessao_da_rede():
    """Monta, treina e avalia uma rede; depois mostra o menu final.

    Devolve True se o usuário quer criar outra rede, False para sair.
    """
    rede, semente = montar_rede()

    base = carregar_base(rede)
    if base is None:
        return True
    X, Y, classes, nome_da_base = base

    # Separa primeiro, com os valores originais (a tabela mostra eles).
    X_treino_original, Y_treino_original, X_teste_original, Y_teste_original, separou = (
        separar_treino_teste(X, Y, seed=semente)
    )

    # Normaliza entradas E saídas para [0, 1] (bases que já são 0/1 não
    # mudam; as classes em one-hot também não). Mínimos e máximos vêm só do
    # treino; o teste e as entradas digitadas usam a mesma escala, como
    # dados que a rede nunca viu. ŷ é desnormalizado para ser mostrado.
    X_treino, *escala_x = normalizar(X_treino_original)
    X_teste = aplicar_normalizacao(X_teste_original, *escala_x)
    Y_treino, *escala_y = normalizar(Y_treino_original)
    Y_teste = aplicar_normalizacao(Y_teste_original, *escala_y)

    if separou:
        titulo = f"{len(Y_teste)} amostras de teste, não usadas no treino"
        print(
            f"  Base separada: {len(Y_treino)} amostras para treino e "
            f"{len(Y_teste)} para teste."
        )
        X_treino = perguntar_lotes(X_treino_original, X_treino)
    else:
        titulo = "base inteira"
        print(
            "  Base pequena (menos de 10 amostras): não foi separada. "
            "O treino e o teste usam a base inteira."
        )

    taxa = perguntar_float("Taxa de aprendizado (η) (Enter = 0.5): ", maior_que=0, padrao=0.5)
    epocas = perguntar_inteiro("Número de épocas (Enter = 5000): ", 1, padrao=5000)

    # Erro de cada época, desde o início (o "treinar mais" continua a lista).
    historico = treinar(rede, X_treino, Y_treino, epocas, taxa)
    avaliar(rede, X_teste, X_teste_original, Y_teste, Y_teste_original,
            escala_y, titulo, classes)

    while True:
        print()
        print("Menu:")
        print("  1) Testar uma entrada digitada")
        print("  2) Treinar mais épocas")
        print("  3) Criar outra rede")
        print("  4) Ver gráfico do erro por época")
        print("  0) Sair")
        opcao = perguntar_opcao("Opção: ", ["1", "2", "3", "4", "0"])

        if opcao == "1":
            testar_entrada_digitada(rede, escala_x, escala_y, classes)
        elif opcao == "2":
            mais = perguntar_inteiro("Quantas épocas a mais? ", 1)
            historico += treinar(rede, X_treino, Y_treino, mais, taxa)
            avaliar(rede, X_teste, X_teste_original, Y_teste, Y_teste_original,
                    escala_y, titulo, classes)
        elif opcao == "4":
            mostrar_grafico(historico, nome_da_base, taxa)
        elif opcao == "3":
            return True
        else:
            return False


def main():
    # Em terminais que não conhecem algum símbolo (ex.: ŷ), troca por "?"
    # em vez de quebrar o programa.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    print("=== Construtor de Redes Neurais ===")
    try:
        if perguntar_sn("Quer criar uma rede neural? (s/n) "):
            while sessao_da_rede():
                print()
                print("--- Nova rede ---")
    except (EOFError, KeyboardInterrupt):
        # Ctrl+C ou fim da entrada: sai sem mostrar traceback.
        print()
    print("Até mais!")


if __name__ == "__main__":
    main()
