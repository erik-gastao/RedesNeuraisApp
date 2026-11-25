"""Construtor de Redes Neurais — programa de terminal.

Este é o único arquivo que conversa com o usuário (input e print).
As contas (dividir, normalizar, treinar, avaliar, prever, salvar) ficam
em experimento.py, que a interface gráfica também usa; a rede em si fica
na pasta rede/.

Rodar com:  python main.py
"""

import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from dados import carregar_csv, listar_bases
from experimento import PERCENTUAIS_PADRAO, Experimento, nome_da_classe
from graficos import grafico_do_erro
from rede.ativacoes import ATIVACOES
from rede.rede_neural import RedeNeural

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


def perguntar_divisao(experimento):
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
            experimento.dividir(percentuais)
            return
        except ValueError as erro:
            print(f"  Valor inválido: {erro}")


def perguntar_lotes(N):
    """Pergunta se as entradas do treino devem ser normalizadas por lotes.

    Enter mantém a normalização do treino inteiro, que é o padrão do
    enunciado (devolve None). Com um tamanho menor, cada lote de registros é
    normalizado com o mínimo e o máximo dele mesmo (ver
    dados.normalizar_por_lotes) — opção extra deste projeto.
    """
    tamanho = perguntar_inteiro(
        f"Tamanho do lote para normalizar (2 a {N}, Enter = treino inteiro, o padrão): ",
        2, padrao=N, maximo=N,
    )
    if tamanho == N:
        return None
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
    return tamanho


# ----------------------------------------------------------------- treino

def treinar(experimento, epocas, taxa):
    """Treina com SGD mostrando o erro a cada ~10% das épocas."""
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
    experimento.treinar(epocas, taxa, ao_fim_da_epoca=mostrar_progresso)


# -------------------------------------------------------------- avaliação

def formatar_valores(valores):
    """Vetor de números → texto curto, ex.: '0 1' ou '5.1 3.5'."""
    return " ".join(f"{valor:g}" for valor in valores)


def formatar_porcentagem(fracao):
    """0.75 → '75,00%' (vírgula decimal, como no português)."""
    return f"{fracao * 100:.2f}%".replace(".", ",")


def avaliar(experimento, titulo):
    """Mostra o MSE dos três conjuntos e a tabela do conjunto de teste.

    Classificação: acurácia e colunas entrada | esperado | ŷ | classe | ok?.
    Regressão: colunas entrada | esperado | ŷ | erro (ŷ − esperado).
    """
    resultado = experimento.avaliar()
    partes = [f"MSE treino: {resultado['mse_treino']:.4f}"]
    if resultado["mse_validacao"] is not None:
        partes.append(f"validação: {resultado['mse_validacao']:.4f}")

    print()
    print(f"Avaliação ({titulo}):")
    if resultado["mse_teste"] is None:
        print("  " + " | ".join(partes))
        print("  Sem conjunto de teste (0%).")
        return
    partes.append(f"teste: {resultado['mse_teste']:.4f}")
    print("  " + " | ".join(partes))

    classes = experimento.classes
    total = len(resultado["linhas"])
    if resultado["classificacao"]:
        print(
            f"  Acurácia: {formatar_porcentagem(resultado['acuracia'])} "
            f"({resultado['acertos']}/{total})"
        )
        cabecalho = ("entrada", "esperado", "ŷ", "classe", "ok?")
    else:
        cabecalho = ("entrada", "esperado", "ŷ", "erro")

    n_linhas = min(total, MAXIMO_LINHAS_TABELA)
    linhas = [cabecalho]
    for linha in resultado["linhas"][:n_linhas]:
        y_hat = " ".join(f"{valor:.4f}" for valor in linha["y_hat"])
        if resultado["classificacao"]:
            linhas.append((
                formatar_valores(linha["entrada"]),
                nome_da_classe(linha["esperado"], classes),
                y_hat,
                nome_da_classe(linha["classificado"], classes),
                "sim" if linha["acertou"] else "NÃO",
            ))
        else:
            linhas.append((
                formatar_valores(linha["entrada"]),
                formatar_valores(linha["esperado"]),
                y_hat,
                " ".join(f"{valor:+.4f}" for valor in linha["erro"]),
            ))

    # Largura de cada coluna = maior texto dela.
    larguras = [max(len(linha[c]) for linha in linhas) for c in range(len(cabecalho))]
    print()
    for linha in linhas:
        print("  " + " | ".join(texto.ljust(larguras[c]) for c, texto in enumerate(linha)))
    if total > n_linhas:
        print(f"  ... (mostrando {n_linhas} de {total} amostras)")


# ------------------------------------------------------------- menu final

def testar_entrada_digitada(experimento):
    """Previsão (seção 1.12): pede uma entrada, passa pela rede e mostra ŷ.

    Só faz a propagação direta (nenhum peso muda). Se o treino usou
    entradas normalizadas, a entrada digitada recebe a mesma normalização
    (mesmos mínimos e máximos do treino).
    """
    valores = [
        perguntar_float(f"  Valor da entrada {i}: ")
        for i in range(1, experimento.rede.n_entradas + 1)
    ]
    y_hat, classe = experimento.prever_entrada(valores)
    linha = f"  ŷ = {' '.join(f'{valor:.4f}' for valor in y_hat)}"
    if classe is not None:
        linha += f"  →  classe: {classe}"
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


def salvar(experimento):
    """Pergunta o arquivo e salva a rede em JSON (seção 1.11)."""
    padrao = PASTA_DE_REDES / f"{Path(experimento.nome_base).stem}.json"
    texto = input(f"Arquivo para salvar (Enter = {padrao}): ").strip().strip('"')
    try:
        caminho = experimento.salvar(texto or padrao)
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
    experimento = Experimento(rede, X, Y, classes, nome_da_base, semente)

    # Divide primeiro, com os valores originais (a tabela mostra eles).
    if len(X) >= 10:
        perguntar_divisao(experimento)
    else:
        experimento.dividir()
    if experimento.separou:
        titulo = f"{len(experimento.Y_teste)} amostras de teste, não usadas no treino"
        print(
            f"  Base separada: {len(experimento.Y_treino)} para treino, "
            f"{len(experimento.Y_validacao)} para validação e "
            f"{len(experimento.Y_teste)} para teste."
        )
    else:
        # Validação igual ao treino não acrescenta nada: fica sem validação.
        titulo = "base inteira"
        print(
            "  Base pequena (menos de 10 amostras): não foi separada. "
            "O treino e o teste usam a base inteira."
        )

    # Normalização opcional, só das entradas. Mínimos e máximos vêm só do
    # treino e valem para validação, teste e entradas digitadas. As saídas
    # esperadas nunca são normalizadas.
    if perguntar_sn("Normalizar as entradas (min-max, escala do treino)? (s/n) "):
        tamanho_lote = perguntar_lotes(len(experimento.X_treino)) if experimento.separou else None
        experimento.normalizar_entradas(tamanho_lote)

    taxa = perguntar_float(f"Taxa de aprendizado (η) (Enter = {TAXA_PADRAO}): ",
                           maior_que=0, padrao=TAXA_PADRAO)
    epocas = perguntar_inteiro(f"Número de épocas (Enter = {EPOCAS_PADRAO}): ",
                               1, padrao=EPOCAS_PADRAO)

    # O experimento guarda os erros de cada época desde o início.
    treinar(experimento, epocas, taxa)
    avaliar(experimento, titulo)

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
            testar_entrada_digitada(experimento)
        elif opcao == "2":
            mais = perguntar_inteiro("Quantas épocas a mais? ", 1)
            treinar(experimento, mais, taxa)
            avaliar(experimento, titulo)
        elif opcao == "3":
            return True
        elif opcao == "4":
            mostrar_grafico(experimento.historico, experimento.nome_base, taxa)
        elif opcao == "5":
            salvar(experimento)
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
            experimento = Experimento.carregar(texto)
        except (OSError, ValueError) as erro:
            print(f"  Não foi possível carregar a rede: {erro}")
            continue
        break

    print()
    print("Rede carregada:")
    mostrar_resumo(experimento.rede)
    if experimento.normalizacao is not None:
        print("  As entradas digitadas serão normalizadas com a escala salva.")

    while True:
        print()
        print("Menu:")
        print("  1) Testar uma entrada digitada")
        print("  0) Sair")
        if perguntar_opcao("Opção: ", ["1", "0"]) == "0":
            return False
        testar_entrada_digitada(experimento)


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
