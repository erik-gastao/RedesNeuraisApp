"""Salvar e carregar uma rede treinada em JSON (seção 1.11 do enunciado).

Formato do arquivo (versão 1), descrito também no README:

    {
      "formato": "rede-mlp", "versao": 1,
      "n_entradas": 2, "usar_bias": true,
      "camadas": [                      ← ocultas na ordem; a última é a saída
        {"neuronios": 4, "ativacao": "relu",
         "pesos": [[w11, w12], ...],    ← uma lista por neurônio, um peso por entrada
         "vieses": [b1, ...]},          ← um viés por neurônio
        ...
      ],
      "normalizacao": {"funcao": "min-max", "minimos": [...], "maximos": [...]} ou null,
      "classes": ["Iris-setosa", ...] ou null
    }

Os números são gravados com a precisão completa do Python, então a rede
carregada dá exatamente as mesmas saídas da original.
"""

import json
from pathlib import Path

import numpy as np

from rede.rede_neural import RedeNeural

FORMATO = "rede-mlp"
VERSAO = 1


def salvar_rede(caminho, rede, normalizacao=None, classes=None):
    """Grava a rede (arquitetura, pesos, vieses), a normalização e as classes.

    normalizacao: {"funcao": "min-max", "minimos": [...], "maximos": [...]}
                  ou None se as entradas não foram normalizadas.
    classes:      nomes das classes (base com classe em texto) ou None.
    Cria a pasta se faltar. Devolve o caminho do arquivo.
    """
    dados = {
        "formato": FORMATO,
        "versao": VERSAO,
        "n_entradas": rede.n_entradas,
        "usar_bias": rede.usar_bias,
        "camadas": [
            {
                "neuronios": camada.n_neuronios,
                "ativacao": camada.nome_ativacao,
                "pesos": [neuronio.pesos.tolist() for neuronio in camada.neuronios],
                "vieses": [neuronio.bias for neuronio in camada.neuronios],
            }
            for camada in rede.camadas
        ],
        "normalizacao": _normalizacao_para_json(normalizacao),
        "classes": list(classes) if classes is not None else None,
    }
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps(dados, ensure_ascii=False, indent=2), encoding="utf-8")
    return caminho


def carregar_rede(caminho):
    """Lê o arquivo e reconstrói a rede.

    Devolve (rede, normalizacao, classes). Levanta FileNotFoundError se o
    arquivo não existe e ValueError se ele não for uma rede válida.
    """
    try:
        texto = Path(caminho).read_text(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}") from None
    except (IsADirectoryError, PermissionError):
        raise ValueError(f"Não foi possível ler o arquivo: {caminho}") from None
    except UnicodeDecodeError:
        raise ValueError(f"O arquivo não é um arquivo de texto (JSON): {caminho}") from None
    try:
        dados = json.loads(texto)
    except json.JSONDecodeError:
        raise ValueError(f"O arquivo não é um JSON válido: {caminho}") from None

    if not isinstance(dados, dict) or dados.get("formato") != FORMATO:
        raise ValueError(f"O arquivo não é uma rede salva por este programa: {caminho}")

    try:
        return _reconstruir(dados)
    except (KeyError, TypeError, IndexError) as erro:
        raise ValueError(f"Arquivo de rede incompleto ou inválido (campo {erro}).") from None


def _reconstruir(dados):
    """Monta a RedeNeural com a arquitetura do arquivo e copia pesos e vieses."""
    camadas = dados["camadas"]
    if not isinstance(camadas, list) or not camadas:
        raise ValueError("O campo 'camadas' precisa ser uma lista com pelo menos uma camada.")
    if not _inteiro_positivo(dados["n_entradas"]):
        raise ValueError("A rede precisa de pelo menos 1 entrada (campo 'n_entradas').")
    for numero, camada in enumerate(camadas, start=1):
        if not _inteiro_positivo(camada["neuronios"]):
            raise ValueError(f"Camada {numero}: precisa de pelo menos 1 neurônio.")
    ocultas = [(c["neuronios"], c["ativacao"]) for c in camadas[:-1]]
    saida = camadas[-1]
    rede = RedeNeural(
        dados["n_entradas"], ocultas, saida["neuronios"], dados["usar_bias"],
        ativacao_saida=saida["ativacao"],
    )

    for numero, (camada, salva) in enumerate(zip(rede.camadas, camadas), start=1):
        pesos = np.array(salva["pesos"], dtype=float)
        vieses = np.array(salva["vieses"], dtype=float)
        if pesos.shape != (camada.n_neuronios, camada.n_entradas):
            raise ValueError(
                f"Camada {numero}: pesos com forma {pesos.shape}, "
                f"esperado ({camada.n_neuronios}, {camada.n_entradas})."
            )
        if vieses.shape != (camada.n_neuronios,):
            raise ValueError(f"Camada {numero}: esperado {camada.n_neuronios} vieses.")
        for neuronio, w, b in zip(camada.neuronios, pesos, vieses):
            neuronio.pesos = w.copy()
            neuronio.bias = float(b)

    normalizacao = dados.get("normalizacao")
    _conferir_normalizacao(normalizacao, rede.n_entradas)
    classes = dados.get("classes")
    if classes is not None and (not isinstance(classes, list) or len(classes) != rede.n_saidas):
        raise ValueError(
            f"O campo 'classes' precisa ter um nome por neurônio de saída ({rede.n_saidas})."
        )
    return rede, normalizacao, classes


def _inteiro_positivo(valor):
    return isinstance(valor, int) and not isinstance(valor, bool) and valor >= 1


def _conferir_normalizacao(normalizacao, n_entradas):
    """A normalização salva precisa ser min-max com um mínimo e um máximo por entrada."""
    if normalizacao is None:
        return
    if not isinstance(normalizacao, dict):
        raise ValueError("O campo 'normalizacao' precisa ser um objeto ou null.")
    funcao = normalizacao.get("funcao")
    if funcao != "min-max":
        raise ValueError(f"Normalização desconhecida: '{funcao}' (o programa usa só min-max).")
    for chave in ("minimos", "maximos"):
        valores = normalizacao.get(chave)
        if not isinstance(valores, list) or len(valores) != n_entradas:
            raise ValueError(
                f"A normalização precisa de '{chave}' com um valor para cada uma "
                f"das {n_entradas} entradas."
            )


def _normalizacao_para_json(normalizacao):
    """Converte os arrays numpy da normalização em listas."""
    if normalizacao is None:
        return None
    return {
        chave: valor.tolist() if isinstance(valor, np.ndarray) else valor
        for chave, valor in normalizacao.items()
    }
