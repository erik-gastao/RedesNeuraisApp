"""Formulário do hub: o que o usuário digitou e a validação (sem pygame).

O hub (tela_hub.py) guarda os textos dos campos num FormularioHub e chama
validar() a cada mudança: os erros aparecem embaixo de cada campo e o botão
"Gerar" só funciona sem erros. Os parâmetros validados vão direto para
Experimento.da_base e para o gravador.
"""

import re
from dataclasses import dataclass, field

from dados import aviso_alcance, tamanhos_da_divisao
from rede.ativacoes import ATIVACOES

NIVEIS = ("ligacao", "neuronio", "camada")


@dataclass
class FormularioHub:
    """Valores dos campos do hub, como texto (do jeito que foram digitados)."""

    base: str | None = None
    colunas_saida: str = "1"
    ocultas: list = field(default_factory=lambda: [["5", "sigmoid"]])
    ativacao_saida: str = "sigmoid"
    usar_bias: bool = True
    semente: str = "1"
    taxa: str = "0.1"
    epocas: str = "1000"
    divisao: list = field(default_factory=lambda: ["70", "15", "15"])
    normalizar: bool = True
    lote: str = ""
    nivel: str = "ligacao"
    epocas_detalhadas: str = "1, 2, 10, 100, fim"
    exemplo_em_foco: str = "1"


def _inteiro(texto, minimo, maximo=None):
    """Inteiro entre minimo e maximo, ou ValueError com a explicação."""
    try:
        valor = int(texto.strip())
    except ValueError:
        raise ValueError("Digite um número inteiro.") from None
    if valor < minimo:
        raise ValueError(f"O mínimo é {minimo}.")
    if maximo is not None and valor > maximo:
        raise ValueError(f"O máximo é {maximo}.")
    return valor


def numero_de_entradas_e_saidas(formulario, descricao):
    """(n_entradas, n_saidas) da base, ou ValueError se as colunas de saída não servem."""
    if descricao["classes"] is not None:
        return descricao["colunas"] - 1, len(descricao["classes"])
    n_saidas = _inteiro(formulario.colunas_saida, 1, descricao["colunas"] - 1)
    return descricao["colunas"] - n_saidas, n_saidas


def validar(formulario, descricao):
    """Confere todos os campos.

    descricao: o que dados.descrever_base devolveu para a base escolhida.
    Devolve (parametros, erros): parametros é None se houver algum erro;
    erros é {campo: mensagem}.
    """
    erros = {}
    if formulario.base is None or descricao is None:
        return None, {"base": "Escolha uma base."}

    def conferir(campo, funcao):
        try:
            return funcao()
        except ValueError as erro:
            erros[campo] = str(erro)
            return None

    entradas_saidas = conferir("colunas_saida", lambda: numero_de_entradas_e_saidas(formulario, descricao))

    def ocultas():
        camadas = []
        for numero, (neuronios, ativacao) in enumerate(formulario.ocultas, start=1):
            if ativacao not in ATIVACOES:
                raise ValueError(f"Camada {numero}: ativação desconhecida.")
            try:
                camadas.append((_inteiro(neuronios, 1), ativacao))
            except ValueError as erro:
                raise ValueError(f"Camada {numero}: {erro}") from None
        return camadas

    camadas = conferir("ocultas", ocultas)
    seed = conferir("semente", lambda: None if not formulario.semente.strip()
                    else _inteiro(formulario.semente, 0))

    def taxa():
        try:
            valor = float(formulario.taxa.strip().replace(",", "."))
        except ValueError:
            raise ValueError("Digite um número (ex.: 0.1).") from None
        if not valor > 0:
            raise ValueError("Precisa ser maior que 0.")
        return valor

    eta = conferir("taxa", taxa)
    epocas = conferir("epocas", lambda: _inteiro(formulario.epocas, 1))

    def divisao():
        try:
            percentuais = tuple(float(p.strip().replace(",", ".")) for p in formulario.divisao)
        except ValueError:
            raise ValueError("Os percentuais precisam ser números.") from None
        return percentuais, tamanhos_da_divisao(descricao["amostras"], percentuais)

    divisao_ok = conferir("divisao", divisao)
    n_treino = divisao_ok[1][0] if divisao_ok else descricao["amostras"]

    tamanho_lote = None
    if formulario.normalizar and formulario.lote.strip():
        tamanho_lote = conferir("lote", lambda: _inteiro(formulario.lote, 2, n_treino))

    def epocas_detalhadas():
        resultado = []
        for parte in re.split(r"[,\s]+", formulario.epocas_detalhadas.strip()):
            if not parte:
                continue
            if parte.lower() == "fim":
                resultado.append("fim")
                continue
            try:
                resultado.append(_inteiro(parte, 1, epocas))
            except ValueError as erro:
                raise ValueError(f"'{parte}': {erro} Use números de época ou 'fim'.") from None
        return resultado

    detalhadas = conferir("epocas_detalhadas", epocas_detalhadas)
    exemplo = conferir("exemplo_em_foco", lambda: _inteiro(formulario.exemplo_em_foco, 1, n_treino))
    if formulario.nivel not in NIVEIS:
        erros["nivel"] = "Escolha ligação, neurônio ou camada."

    if erros:
        return None, erros
    n_entradas, n_saidas = entradas_saidas
    return {
        "caminho": formulario.base,
        "n_entradas": n_entradas,
        "n_saidas": n_saidas,
        "ocultas": camadas,
        "ativacao_saida": formulario.ativacao_saida,
        "usar_bias": formulario.usar_bias,
        "seed": seed,
        "percentuais": divisao_ok[0],
        "normalizar": formulario.normalizar,
        "tamanho_lote": tamanho_lote,
        "taxa": eta,
        "epocas": epocas,
        "epocas_detalhadas": detalhadas,
        "nivel": formulario.nivel,
        "exemplo_em_foco": exemplo - 1,
    }, {}


def resumo_base(formulario, descricao):
    """Ex.: '150 amostras · 4 entradas · 3 classes'."""
    try:
        n_entradas, n_saidas = numero_de_entradas_e_saidas(formulario, descricao)
    except ValueError:
        n_entradas, n_saidas = descricao["colunas"] - 1, 1
    if descricao["classes"] is not None:
        saida = f"{len(descricao['classes'])} classes"
    else:
        saida = "saída numérica" if n_saidas == 1 else f"{n_saidas} saídas numéricas"
    return f"{descricao['amostras']} amostras · {n_entradas} entradas · {saida}"


def sugestao_ativacao_saida(descricao):
    """sigmoid se a saída é classe (texto ou 0/1); linear se é número (regressão)."""
    if descricao["classes"] is not None:
        return "sigmoid"
    minimo, maximo = descricao["minimos"][-1], descricao["maximos"][-1]
    return "sigmoid" if minimo >= 0 and maximo <= 1 else "linear"


def aviso_saida(formulario, descricao):
    """Aviso quando a saída esperada passa do alcance da ativação da saída."""
    if descricao["classes"] is not None:
        return aviso_alcance(0.0, 1.0, formulario.ativacao_saida)   # one-hot
    try:
        _, n_saidas = numero_de_entradas_e_saidas(formulario, descricao)
    except ValueError:
        return None
    minimo = min(descricao["minimos"][-n_saidas:])
    maximo = max(descricao["maximos"][-n_saidas:])
    return aviso_alcance(minimo, maximo, formulario.ativacao_saida)


def texto_divisao(formulario, descricao):
    """Ex.: '→ 125 / 27 / 27' (treino / validação / teste)."""
    try:
        percentuais = [float(p.replace(",", ".")) for p in formulario.divisao]
        treino, validacao, teste = tamanhos_da_divisao(descricao["amostras"], percentuais)
    except ValueError:
        return ""
    if validacao == 0 and teste == 0 and descricao["amostras"] < 10:
        return "base pequena: não é dividida"
    return f"→ {treino} / {validacao} / {teste}"


def previa(formulario, descricao):
    """Ex.: '4 → [5 sigmoid] → 3 sigmoid   (35 pesos + 8 bias)'; '' se inválido."""
    try:
        n_entradas, n_saidas = numero_de_entradas_e_saidas(formulario, descricao)
        camadas = [(_inteiro(n, 1), a) for n, a in formulario.ocultas]
    except ValueError:
        return ""
    partes = [str(n_entradas)] + [f"[{n} {a}]" for n, a in camadas]
    partes.append(f"{n_saidas} {formulario.ativacao_saida}")
    tamanhos = [n_entradas] + [n for n, _ in camadas] + [n_saidas]
    pesos = sum(a * b for a, b in zip(tamanhos, tamanhos[1:]))
    contagem = f"{pesos} pesos"
    if formulario.usar_bias:
        contagem += f" + {sum(tamanhos[1:])} bias"
    return f"{' → '.join(partes)}   ({contagem})"
