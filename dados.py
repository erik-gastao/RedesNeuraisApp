"""Funções para preparar a base de dados: listar e carregar CSVs,
normalizar (inteira ou por lotes) e desnormalizar, separar treino/teste
e medir a acurácia.

Este arquivo não faz input() nem print(): problemas viram exceções com
mensagens claras (FileNotFoundError ou ValueError) e o main.py mostra ao
usuário.
"""

import csv
from pathlib import Path

import numpy as np

# Com menos linhas que isso (XOR, AND) a base não é separada em treino e teste.
MINIMO_PARA_SEPARAR = 10

# Extensões de arquivo mostradas no menu de bases.
EXTENSOES_DE_BASE = {".csv", ".data", ".txt"}


def listar_bases(pastas):
    """Arquivos de base (.csv, .data, .txt) das pastas dadas, em ordem.

    Pastas que não existem são ignoradas.
    """
    bases = []
    for pasta in pastas:
        pasta = Path(pasta)
        if pasta.is_dir():
            arquivos = [
                arquivo for arquivo in pasta.iterdir()
                if arquivo.is_file() and arquivo.suffix.lower() in EXTENSOES_DE_BASE
            ]
            bases.extend(sorted(arquivos, key=lambda arquivo: arquivo.name.lower()))
    return bases


def carregar_csv(caminho, n_saidas):
    """Lê um CSV separado por vírgula e devolve (X, Y, classes).

    caminho:  caminho do arquivo
    n_saidas: número de neurônios da camada de saída

    A saída esperada pode vir de dois jeitos:

    1. Números: as últimas n_saidas colunas viram Y e as demais viram X.
       Nesse caso `classes` é None.

    2. Texto na última coluna (ex.: Iris, com "Iris-setosa"): cada nome
       diferente é uma classe, e a coluna vira uma coluna 0/1 por classe
       (codificação one-hot). Com 3 classes:
           Iris-setosa     → 1 0 0
           Iris-versicolor → 0 1 0
           Iris-virginica  → 0 0 1
       A rede precisa de um neurônio de saída por classe. `classes` traz
       os nomes na ordem das colunas de Y (ordem alfabética).

    O cabeçalho é opcional (ver _e_cabecalho). Linhas em branco são puladas.
    """
    linhas = _ler_linhas(caminho)

    if _e_cabecalho(linhas):
        linhas = linhas[1:]
        if not linhas:
            raise ValueError("O arquivo só tem o cabeçalho, sem linhas de dados.")

    n_colunas = len(linhas[0][1])
    for numero, campos in linhas:
        if len(campos) != n_colunas:
            raise ValueError(
                f"Linha {numero}: tem {len(campos)} coluna(s), "
                f"mas as anteriores têm {n_colunas}."
            )

    ultima_coluna = [campos[-1] for _, campos in linhas]
    if all(_e_numero(valor) for valor in ultima_coluna):
        return _carregar_saida_numerica(linhas, n_colunas, n_saidas)
    return _carregar_saida_em_classes(linhas, n_colunas, n_saidas, ultima_coluna)


def _carregar_saida_numerica(linhas, n_colunas, n_saidas):
    """Caso 1: todas as colunas são números; as últimas n_saidas são Y."""
    if n_colunas <= n_saidas:
        raise ValueError(
            f"A base tem {n_colunas} coluna(s), mas a rede precisa de pelo "
            f"menos 1 de entrada mais {n_saidas} de saída."
        )
    tabela = np.array([_numeros(numero, campos) for numero, campos in linhas])
    return tabela[:, :-n_saidas], tabela[:, -n_saidas:], None


def _carregar_saida_em_classes(linhas, n_colunas, n_saidas, ultima_coluna):
    """Caso 2: a última coluna é o nome da classe; vira one-hot."""
    if n_colunas < 2:
        raise ValueError("A base precisa de pelo menos 1 coluna de entrada além da classe.")

    classes = sorted(set(ultima_coluna))
    if len(classes) != n_saidas:
        raise ValueError(
            f"A última coluna tem {len(classes)} classe(s) "
            f"({', '.join(classes)}), então a rede precisa de {len(classes)} "
            f"neurônio(s) de saída (um por classe), mas tem {n_saidas}."
        )

    X = np.array([_numeros(numero, campos[:-1]) for numero, campos in linhas])

    # One-hot: na linha de cada amostra, 1 na coluna da sua classe.
    Y = np.zeros((len(linhas), len(classes)))
    for i, nome in enumerate(ultima_coluna):
        Y[i, classes.index(nome)] = 1
    return X, Y, classes


def _ler_linhas(caminho):
    """Lê o arquivo e devolve [(número da linha, [valores])] sem linhas vazias."""
    try:
        with open(caminho, newline="", encoding="utf-8-sig") as arquivo:
            linhas = []
            for numero, campos in enumerate(csv.reader(arquivo), start=1):
                campos = [campo.strip() for campo in campos]
                if any(campos):
                    linhas.append((numero, campos))
    except FileNotFoundError:
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}") from None
    except IsADirectoryError:
        raise ValueError(f"O caminho é uma pasta, não um arquivo: {caminho}") from None
    except UnicodeDecodeError:
        raise ValueError(f"O arquivo não é um CSV de texto: {caminho}") from None

    if not linhas:
        raise ValueError(f"O arquivo está vazio: {caminho}")
    return linhas


def _e_cabecalho(linhas):
    """Decide se a primeira linha é um cabeçalho (nomes das colunas).

    É cabeçalho se tiver texto numa coluna que, na linha seguinte, é número.
    Exemplos:
        "x1,x2,y" seguido de "0,0,0"                  → cabeçalho
        "5.1,3.5,1.4,0.2,Iris-setosa" seguido de outra
        linha com texto no mesmo lugar                → dado (a classe é texto)
    Se o arquivo tiver uma linha só, é cabeçalho se tiver qualquer texto.
    """
    primeira = linhas[0][1]
    if len(linhas) == 1:
        return not all(_e_numero(valor) for valor in primeira)
    segunda = linhas[1][1]
    return any(
        not _e_numero(valor) and _e_numero(abaixo)
        for valor, abaixo in zip(primeira, segunda)
    )


def _numeros(numero_linha, campos):
    """Converte os campos de uma linha em números, ou explica o erro."""
    for campo in campos:
        if not _e_numero(campo):
            raise ValueError(f"Linha {numero_linha}: o valor '{campo}' não é um número.")
    return [float(campo) for campo in campos]


def _e_numero(texto):
    """True se o texto pode ser convertido para um número comum."""
    try:
        return bool(np.isfinite(float(texto)))
    except ValueError:
        return False


def normalizar(X):
    """Normalização min-max: leva cada coluna de X para o intervalo [0, 1].

        x_norm = (x − mínimo) / (máximo − mínimo)

    Devolve (X_norm, minimos, maximos). Os mínimos e máximos devem ser
    guardados para normalizar, do mesmo jeito, entradas que chegarem depois
    (ver aplicar_normalizacao).

    Por que normalizar: colunas com valores grandes (ex.: 5000) dominariam
    a soma ponderada e saturariam a sigmoid; com tudo entre 0 e 1 as
    entradas pesam de forma parecida.
    """
    minimos = X.min(axis=0)
    maximos = X.max(axis=0)
    return aplicar_normalizacao(X, minimos, maximos), minimos, maximos


def aplicar_normalizacao(X, minimos, maximos):
    """Normaliza X usando mínimos e máximos já calculados por normalizar().

    Usada para entradas novas (ex.: digitadas pelo usuário), que precisam
    da mesma escala usada no treino.
    """
    return (X - minimos) / _amplitude(minimos, maximos)


def desnormalizar(X_norm, minimos, maximos):
    """Desfaz a normalização: x = x_norm · (máximo − mínimo) + mínimo.

    Usada nas saídas: a rede treina e prevê Y na escala [0, 1], e esta
    função devolve ŷ na escala original da base para mostrar ao usuário.
    """
    return X_norm * _amplitude(minimos, maximos) + minimos


def _amplitude(minimos, maximos):
    """máximo − mínimo de cada coluna, trocando 0 por 1.

    Coluna constante (máximo = mínimo): evita divisão por zero. Dividindo
    por 1, a coluna normalizada fica 0, e desnormalizar a devolve inteira.
    """
    amplitude = maximos - minimos
    return np.where(amplitude == 0, 1.0, amplitude)


def normalizar_por_lotes(X, tamanho_lote):
    """Normalização min-max feita lote a lote.

    As linhas são divididas, na ordem em que estão, em lotes de
    `tamanho_lote` registros. Cada lote é normalizado com o mínimo e o
    máximo DELE MESMO (não da base inteira):

        lote 1 = linhas 0 .. t−1    → min/máx do lote 1
        lote 2 = linhas t .. 2t−1   → min/máx do lote 2
        ...

    Se o último lote ficar com 1 linha só, ela entra no lote anterior: uma
    linha sozinha tem mínimo = máximo e viraria toda 0.

    Devolve X_norm, com a mesma forma de X. Com tamanho_lote ≥ número de
    linhas, o resultado é igual ao de normalizar(X).
    """
    if tamanho_lote < 2:
        raise ValueError("O lote precisa ter pelo menos 2 registros.")

    N = X.shape[0]
    inicios = list(range(0, N, tamanho_lote))
    if len(inicios) > 1 and N - inicios[-1] < 2:
        inicios.pop()
    fins = inicios[1:] + [N]

    X_norm = np.empty(X.shape, dtype=float)
    for inicio, fim in zip(inicios, fins):
        X_norm[inicio:fim], _, _ = normalizar(X[inicio:fim])
    return X_norm


def separar_treino_teste(X, Y, proporcao_teste=0.2, seed=None):
    """Embaralha as amostras e separa uma parte para teste.

    Devolve (X_treino, Y_treino, X_teste, Y_teste, separou).

    O teste usa amostras que a rede não viu no treino: mede se ela aprendeu
    o padrão em vez de decorar a base. Embaralhar é essencial em bases
    ordenadas por classe (como a Iris): sem isso o teste teria uma classe só.

    Se a base tiver menos de 10 linhas (XOR, AND), não separa: treino e
    teste são a base inteira e `separou` vem False, para o main.py avisar
    o usuário.
    """
    N = X.shape[0]
    if N < MINIMO_PARA_SEPARAR:
        return X, Y, X, Y, False

    # Embaralha os índices das linhas (X e Y continuam alinhados).
    gerador = np.random.default_rng(seed)
    indices = gerador.permutation(N)

    n_teste = max(1, round(N * proporcao_teste))
    indices_teste = indices[:n_teste]
    indices_treino = indices[n_teste:]
    return X[indices_treino], Y[indices_treino], X[indices_teste], Y[indices_teste], True


def acuracia(Y, Y_classificado):
    """Fração de amostras classificadas corretamente (entre 0 e 1).

    Uma amostra só conta como acerto se TODAS as saídas baterem.
    """
    acertos = np.all(Y == Y_classificado, axis=1)
    return float(np.mean(acertos))
