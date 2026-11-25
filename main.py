"""Construtor de Redes Neurais — programa de terminal.

Este é o único arquivo que conversa com o usuário (input e print).
A rede em si fica na pasta rede/, a preparação da base em dados.py e o
salvar/carregar em persistencia.py.

Rodar com:  python main.py
"""

import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from dados import (
    acuracia,
    aplicar_normalizacao,
    carregar_csv,
    e_classificacao,
    listar_bases,
    normalizar,
    normalizar_por_lotes,
    separar_treino_validacao_teste,
)
from graficos import grafico_do_erro
from persistencia import carregar_rede, salvar_rede
from rede.ativacoes import ATIVACOES
from rede.perda import mse
from rede.rede_neural import RedeNeural
from rede.treinamento import treinar as treinar_rede

# Pastas cujos arquivos aparecem no menu de bases (ficam ao lado deste arquivo).
PASTA_DO_PROJETO = Path(__file__).resolve().parent
PASTAS_DE_BASES = [PASTA_DO_PROJETO / "bases", PASTA_DO_PROJETO / "Datasets"]

# Pasta onde os gráficos do erro são salvos.
PASTA_DE_GRAFICOS = PASTA_DO_PROJETO / "graficos"

# Pasta padrão para salvar redes treinadas (.json).
PASTA_DE_REDES = PASTA_DO_PROJETO / "redes"

# Máximo de linhas mostradas na tabela de avaliação (bases grandes).
MAXIMO_LINHAS_TABELA = 20

# Valores usados quando o usuário só aperta Enter.
PERCENTUAIS_PADRAO = (70, 15, 15)
TAXA_PADRAO = 0.1
EPOCAS_PADRAO = 1000


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


def perguntar_opcao(msg, opcoes, padrao=None):
    """Pede uma das opções da lista (texto). Devolve a opção escolhida.

    Com `padrao`, Enter vazio devolve o padrão.
    """
    while True:
        texto = input(msg).strip().lower()
        if texto == "" and padrao is not None:
            return padrao
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
    lista = "/".join(nomes_ativacoes)
    ocultas = []
    for i in range(1, n_ocultas + 1):
        neuronios = perguntar_inteiro(f"  Camada oculta {i} — quantos neurônios? ", 1)
        ativacao = perguntar_opcao(f"  Camada oculta {i} — ativação ({lista})? ", nomes_ativacoes)
        ocultas.append((neuronios, ativacao))

    n_saidas = perguntar_inteiro("Quantos neurônios na camada de saída? ", 1)
    ativacao_saida = perguntar_opcao(
        f"  Camada de saída — ativação ({lista}) (Enter = sigmoid)? ",
        nomes_ativacoes, padrao="sigmoid",
    )
    usar_bias = perguntar_sn("Usar bias? (s/n) ")
    semente = perguntar_semente()

    rede = RedeNeural(n_entradas, ocultas, n_saidas, usar_bias, semente, ativacao_saida)

    print()
    print("Rede montada:")
    mostrar_resumo(rede)
    return rede, semente


def mostrar_resumo(rede):
    for linha in rede.resumo():
        print(f"  {linha}")
    print()


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
        avisar_alcance_da_saida(Y, rede.ativacao_saida)
        return X, Y, classes, Path(caminho).name


# Valores que cada ativação consegue produzir: (mínimo, máximo, texto).
ALCANCE_DAS_ATIVACOES = {
    "sigmoid": (0.0, 1.0, "entre 0 e 1"),
    "tanh": (-1.0, 1.0, "entre -1 e 1"),
    "relu": (0.0, np.inf, "maiores ou iguais a 0"),
}


def avisar_alcance_da_saida(Y, ativacao_saida):
    """Avisa se a saída esperada passa do que a ativação da saída consegue produzir.

    As saídas esperadas não são normalizadas, então uma sigmoid (0 a 1) nunca
    chegaria em 2500 kg/ha, por exemplo. A linear não tem limite.
    """
    if ativacao_saida not in ALCANCE_DAS_ATIVACOES:
        return
    minimo, maximo, texto = ALCANCE_DAS_ATIVACOES[ativacao_saida]
    if Y.min() < minimo or Y.max() > maximo:
        print(
            f"  Atenção: a saída esperada vai de {Y.min():g} a {Y.max():g} e não é "
            f"normalizada, mas a ativação {ativacao_saida} só produz valores {texto}. "
            "Use a ativação linear na saída."
        )


def perguntar_divisao(X, Y, semente):
    """Pergunta os percentuais e divide em treino, validação e teste.

    Aceita "70/15/15", "70 15 15" ou "70,15,15". Com "/", ";" ou espaços
    separando, a vírgula vale como decimal ("70,5/14,5/15"). Repete até a
    divisão ser válida (3 números, sem negativos, somando 100, treino > 0).
    """
    padrao = "/".join(str(p) for p in PERCENTUAIS_PADRAO)
    while True:
        texto = input(f"Divisão treino/validação/teste em % (Enter = {padrao}): ").strip()
        if texto == "":
            percentuais = PERCENTUAIS_PADRAO
        else:
            if re.search(r"[/;\s]", texto):
                partes = re.split(r"[/;\s]+", texto.replace(",", "."))
            else:
                partes = texto.split(",")
            try:
                percentuais = [float(parte) for parte in partes]
            except ValueError:
                print("  Valor inválido: digite 3 números somando 100 (ex.: 70/15/15).")
                continue
        try:
            return separar_treino_validacao_teste(X, Y, percentuais, seed=semente)
        except ValueError as erro:
            print(f"  Valor inválido: {erro}")


def perguntar_lotes(X_treino_original, X_treino):
    """Pergunta se as entradas do treino devem ser normalizadas por lotes.

    Enter mantém a normalização da base de treino inteira (X_treino, já
    pronto), que é o padrão do enunciado. Com um tamanho menor, cada lote de
    registros é normalizado com o mínimo e o máximo dele mesmo (ver
    dados.normalizar_por_lotes) — opção extra deste projeto.
    Devolve o X do treino que a rede vai usar.
    """
    N = len(X_treino_original)
    tamanho = perguntar_inteiro(
        f"Tamanho do lote para normalizar (2 a {N}, Enter = treino inteiro, o padrão): ",
        2, padrao=N, maximo=N,
    )
    if tamanho == N:
        return X_treino
    # Arredonda para cima; uma sobra de 1 registro entra no lote anterior.
    n_lotes = -(-N // tamanho) - (1 if N % tamanho == 1 else 0)
    print(
        f"  Entradas do treino normalizadas em {n_lotes} lote(s) de "
        f"~{tamanho} registros, cada um com sua própria escala. A validação, "
        "o teste e as entradas digitadas usam a escala do treino inteiro."
    )
    print(
        "  Atenção: isso foge do padrão do enunciado (uma escala só, calculada "
        "no treino e aplicada aos três conjuntos)."
    )
    return normalizar_por_lotes(X_treino_original, tamanho)


# ----------------------------------------------------------------- treino

def treinar(rede, conjuntos, epocas, taxa, semente):
    """Treina com SGD mostrando o erro a cada ~10% das épocas.

    Devolve {"treino": [...], "validacao": [...]} com o MSE de cada época.
    """
    intervalo = max(1, epocas // 10)
    largura = len(str(epocas))

    # A pasta rede/ não imprime nada: ela chama esta função ao fim de cada
    # época e é aqui que o progresso aparece.
    def mostrar_progresso(epoca, erro_treino, erro_validacao):
        if epoca % intervalo == 0 or epoca == epocas:
            linha = f"  Época {epoca:>{largura}}/{epocas} — erro treino (MSE): {erro_treino:.4f}"
            if erro_validacao is not None:
                linha += f" | validação: {erro_validacao:.4f}"
            print(linha)

    print()
    print("Treinando...")
    return treinar_rede(
        rede, conjuntos["X_treino"], conjuntos["Y_treino"], epocas, taxa,
        X_validacao=conjuntos["X_validacao"], Y_validacao=conjuntos["Y_validacao"],
        seed=semente, ao_fim_da_epoca=mostrar_progresso,
    )


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


def avaliar(rede, conjuntos, titulo, classes, classificacao):
    """Mostra o MSE dos três conjuntos e a tabela do conjunto de teste.

    Classificação: acurácia e colunas entrada | esperado | ŷ | classe | ok?.
    Regressão: colunas entrada | esperado | ŷ | erro (ŷ − esperado).
    """
    partes = [f"MSE treino: {mse(conjuntos['Y_treino'], rede.prever(conjuntos['X_treino'])):.4f}"]
    if conjuntos["X_validacao"] is not None and len(conjuntos["X_validacao"]):
        erro = mse(conjuntos["Y_validacao"], rede.prever(conjuntos["X_validacao"]))
        partes.append(f"validação: {erro:.4f}")

    X_teste, Y_teste = conjuntos["X_teste"], conjuntos["Y_teste"]
    print()
    print(f"Avaliação ({titulo}):")
    if len(Y_teste) == 0:
        print("  " + " | ".join(partes))
        print("  Sem conjunto de teste (0%).")
        return

    Y_hat = rede.prever(X_teste)
    partes.append(f"teste: {mse(Y_teste, Y_hat):.4f}")
    print("  " + " | ".join(partes))

    if classificacao:
        classificado = rede.classificar(X_teste)
        acertou = np.all(classificado == Y_teste, axis=1)
        print(
            f"  Acurácia: {formatar_porcentagem(acuracia(Y_teste, classificado))} "
            f"({int(acertou.sum())}/{len(acertou)})"
        )
        cabecalho = ("entrada", "esperado", "ŷ", "classe", "ok?")
    else:
        cabecalho = ("entrada", "esperado", "ŷ", "erro")

    n_linhas = min(len(Y_teste), MAXIMO_LINHAS_TABELA)
    linhas = [cabecalho]
    for i in range(n_linhas):
        y_hat = " ".join(f"{valor:.4f}" for valor in Y_hat[i])
        if classificacao:
            linhas.append((
                formatar_valores(conjuntos["X_teste_original"][i]),
                formatar_saida(Y_teste[i], classes),
                y_hat,
                formatar_saida(classificado[i], classes),
                "sim" if acertou[i] else "NÃO",
            ))
        else:
            linhas.append((
                formatar_valores(conjuntos["X_teste_original"][i]),
                formatar_valores(Y_teste[i]),
                y_hat,
                " ".join(f"{valor:+.4f}" for valor in Y_hat[i] - Y_teste[i]),
            ))

    # Largura de cada coluna = maior texto dela.
    larguras = [max(len(linha[c]) for linha in linhas) for c in range(len(cabecalho))]
    print()
    for linha in linhas:
        print("  " + " | ".join(texto.ljust(larguras[c]) for c, texto in enumerate(linha)))
    if len(Y_teste) > n_linhas:
        print(f"  ... (mostrando {n_linhas} de {len(Y_teste)} amostras)")


# ------------------------------------------------------------- menu final

def testar_entrada_digitada(rede, normalizacao, classes, mostrar_classe):
    """Previsão (seção 1.12): pede uma entrada, passa pela rede e mostra ŷ.

    Só faz a propagação direta (nenhum peso muda). Se o treino usou
    entradas normalizadas, a entrada digitada recebe a mesma normalização
    (mesmos mínimos e máximos do treino).
    """
    valores = [
        perguntar_float(f"  Valor da entrada {i}: ")
        for i in range(1, rede.n_entradas + 1)
    ]
    x = np.array([valores])
    if normalizacao is not None:
        x = aplicar_normalizacao(x, np.array(normalizacao["minimos"]),
                                 np.array(normalizacao["maximos"]))
    y_hat = rede.prever(x)[0]
    linha = f"  ŷ = {' '.join(f'{valor:.4f}' for valor in y_hat)}"
    if mostrar_classe:
        linha += f"  →  classe: {formatar_saida(rede.classificar(x)[0], classes)}"
    print(linha)


def mostrar_grafico(historico, nome_da_base, taxa):
    """Salva o gráfico do erro em graficos/ e abre a janela com ele."""
    nome_arquivo = f"erro_{Path(nome_da_base).stem}_{datetime.now():%Y%m%d_%H%M%S}.png"
    epocas = len(historico["treino"])
    titulo = f"Erro por época — {nome_da_base} (η = {taxa:g}, {epocas} épocas)"
    print("  Abrindo o gráfico. Feche a janela dele para voltar ao menu.")
    try:
        caminho = grafico_do_erro(
            historico["treino"], PASTA_DE_GRAFICOS / nome_arquivo, titulo,
            historico_validacao=historico["validacao"] or None,
        )
    except ImportError:
        print("  O gráfico precisa do matplotlib. Instale com: pip install matplotlib")
        return
    print(f"  Gráfico salvo em: {caminho}")


def salvar(rede, normalizacao, classes, nome_da_base):
    """Pergunta o arquivo e salva a rede em JSON (seção 1.11)."""
    padrao = PASTA_DE_REDES / f"{Path(nome_da_base).stem}.json"
    texto = input(f"Arquivo para salvar (Enter = {padrao}): ").strip().strip('"')
    try:
        caminho = salvar_rede(texto or padrao, rede, normalizacao, classes)
    except OSError as erro:
        print(f"  Não foi possível salvar: {erro}")
        return
    print(f"  Rede salva em: {caminho}")


def sessao_da_rede():
    """Monta, treina e avalia uma rede; depois mostra o menu final.

    Devolve True se o usuário quer criar outra rede, False para sair.
    """
    rede, semente = montar_rede()

    base = carregar_base(rede)
    if base is None:
        return True
    X, Y, classes, nome_da_base = base
    classificacao = e_classificacao(Y, classes)

    # Divide primeiro, com os valores originais (a tabela mostra eles).
    X_treino, Y_treino, X_validacao, Y_validacao, X_teste, Y_teste, separou = (
        perguntar_divisao(X, Y, semente) if len(X) >= 10
        else separar_treino_validacao_teste(X, Y, seed=semente)
    )
    if separou:
        titulo = f"{len(Y_teste)} amostras de teste, não usadas no treino"
        print(
            f"  Base separada: {len(Y_treino)} para treino, {len(Y_validacao)} para "
            f"validação e {len(Y_teste)} para teste."
        )
    else:
        # Validação igual ao treino não acrescenta nada: fica sem validação.
        titulo = "base inteira"
        X_validacao = Y_validacao = None
        print(
            "  Base pequena (menos de 10 amostras): não foi separada. "
            "O treino e o teste usam a base inteira."
        )
    X_teste_original = X_teste

    # Normalização opcional, só das entradas. Mínimos e máximos vêm só do
    # treino e valem para validação, teste e entradas digitadas. As saídas
    # esperadas nunca são normalizadas.
    normalizacao = None
    if perguntar_sn("Normalizar as entradas (min-max, escala do treino)? (s/n) "):
        X_treino_original = X_treino
        X_treino, minimos, maximos = normalizar(X_treino)
        normalizacao = {"funcao": "min-max", "minimos": minimos.tolist(),
                        "maximos": maximos.tolist()}
        X_teste = aplicar_normalizacao(X_teste, minimos, maximos)
        if X_validacao is not None:
            X_validacao = aplicar_normalizacao(X_validacao, minimos, maximos)
        if separou:
            X_treino = perguntar_lotes(X_treino_original, X_treino)

    conjuntos = {
        "X_treino": X_treino, "Y_treino": Y_treino,
        "X_validacao": X_validacao, "Y_validacao": Y_validacao,
        "X_teste": X_teste, "Y_teste": Y_teste, "X_teste_original": X_teste_original,
    }

    taxa = perguntar_float(f"Taxa de aprendizado (η) (Enter = {TAXA_PADRAO}): ",
                           maior_que=0, padrao=TAXA_PADRAO)
    epocas = perguntar_inteiro(f"Número de épocas (Enter = {EPOCAS_PADRAO}): ",
                               1, padrao=EPOCAS_PADRAO)

    # Erros de cada época, desde o início (o "treinar mais" continua as listas).
    historico = treinar(rede, conjuntos, epocas, taxa, semente)
    avaliar(rede, conjuntos, titulo, classes, classificacao)

    while True:
        print()
        print("Menu:")
        print("  1) Testar uma entrada digitada")
        print("  2) Treinar mais épocas")
        print("  3) Criar outra rede")
        print("  4) Ver gráfico do erro por época")
        print("  5) Salvar a rede")
        print("  0) Sair")
        opcao = perguntar_opcao("Opção: ", ["1", "2", "3", "4", "5", "0"])

        if opcao == "1":
            testar_entrada_digitada(rede, normalizacao, classes, classificacao)
        elif opcao == "2":
            mais = perguntar_inteiro("Quantas épocas a mais? ", 1)
            novo = treinar(rede, conjuntos, mais, taxa, semente)
            historico["treino"] += novo["treino"]
            historico["validacao"] += novo["validacao"]
            avaliar(rede, conjuntos, titulo, classes, classificacao)
        elif opcao == "3":
            return True
        elif opcao == "4":
            mostrar_grafico(historico, nome_da_base, taxa)
        elif opcao == "5":
            salvar(rede, normalizacao, classes, nome_da_base)
        else:
            return False


def sessao_rede_salva():
    """Carrega uma rede de um arquivo .json e permite fazer previsões.

    Devolve True se o usuário cancelou (volta ao início), False para sair.
    """
    while True:
        texto = input("Arquivo da rede (.json) (Enter = voltar ao início): ").strip().strip('"')
        if texto == "":
            return True
        try:
            rede, normalizacao, classes = carregar_rede(texto)
        except (OSError, ValueError) as erro:
            print(f"  Não foi possível carregar a rede: {erro}")
            continue
        break

    print()
    print("Rede carregada:")
    mostrar_resumo(rede)
    if normalizacao is not None:
        print("  As entradas digitadas serão normalizadas com a escala salva.")
    # Sem a base, a classe aparece quando há nomes de classes ou saída sigmoid.
    mostrar_classe = classes is not None or rede.ativacao_saida == "sigmoid"

    while True:
        print()
        print("Menu:")
        print("  1) Testar uma entrada digitada")
        print("  0) Sair")
        if perguntar_opcao("Opção: ", ["1", "0"]) == "0":
            return False
        testar_entrada_digitada(rede, normalizacao, classes, mostrar_classe)


def main():
    # Em terminais que não conhecem algum símbolo (ex.: ŷ), troca por "?"
    # em vez de quebrar o programa.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    print("=== Construtor de Redes Neurais ===")
    try:
        while True:
            if perguntar_sn("Quer criar uma rede neural? (s/n) "):
                while sessao_da_rede():
                    print()
                    print("--- Nova rede ---")
                break
            if not perguntar_sn("Quer carregar uma rede salva? (s/n) "):
                break
            # Enter no nome do arquivo volta para a primeira pergunta.
            if not sessao_rede_salva():
                break
            print()
    except (EOFError, KeyboardInterrupt):
        # Ctrl+C ou fim da entrada: sai sem mostrar traceback.
        print()
    print("Até mais!")


if __name__ == "__main__":
    main()
