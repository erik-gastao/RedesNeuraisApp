"""O que é um passo do player e como montar o estado da rede num passo (sem pygame)."""

from dataclasses import dataclass, field

# Fases, na ordem em que aparecem. "exemplo" é o passo compacto de um
# exemplo que não está em foco (forward, δ e ajuste de uma vez).
FASES = ("normalizacao", "forward", "erro", "retropropagacao", "atualizacao", "exemplo")


@dataclass
class Passo:
    """Um passo da animação. Carrega tudo que precisa para ser desenhado sozinho.

    camada: 0 = entrada; k = k-ésima camada da rede (rede.camadas[k − 1]).
    neuronio, origem: índices começando em 0 (origem = neurônio da camada
    anterior, nos passos de ligação).
    pesos: [(W, b), ...] de cada camada valendo neste passo — os de antes do
    exemplo até o ajuste; depois do ajuste de uma camada, os novos dela.
    """

    epoca: int
    exemplo: int | None
    fase: str
    tipo: str
    camada: int | None
    neuronio: int | None
    origem: int | None
    valores: dict
    formula: str
    formula_numeros: str
    texto: str
    pesos: list = field(default=None, repr=False)


def _mesmo_grupo(a, b):
    """Passos do mesmo exemplo detalhado (cada passo compacto é um grupo sozinho)."""
    return (a.epoca == b.epoca and a.exemplo == b.exemplo
            and a.fase != "exemplo" and b.fase != "exemplo")


def estado_em(passos, i):
    """Estado da rede para desenhar o passo i.

    Junta as ativações e os deltas já calculados neste exemplo até o passo
    i (inclusive). Devolve {"pesos", "ativacoes", "deltas", "fase"}, com
    ativacoes e deltas como {(camada, neuronio): valor}.
    """
    atual = passos[i]
    inicio = i
    while inicio > 0 and _mesmo_grupo(passos[inicio - 1], atual):
        inicio -= 1

    ativacoes, deltas = {}, {}
    for passo in passos[inicio:i + 1]:
        v = passo.valores
        if passo.tipo == "entrada_rede":
            ativacoes.update({(0, j): x for j, x in enumerate(v["x"])})
        elif passo.tipo == "ativacao":
            ativacoes[(passo.camada, passo.neuronio)] = v["a"]
        elif passo.tipo == "camada_forward":
            ativacoes.update({(passo.camada, n): a for n, a in enumerate(v["a"])})
        elif passo.tipo == "delta":
            deltas[(passo.camada, passo.neuronio)] = v["delta"]
        elif passo.tipo == "camada_delta":
            deltas.update({(passo.camada, n): d for n, d in enumerate(v["delta"])})
    return {"pesos": atual.pesos, "ativacoes": ativacoes, "deltas": deltas, "fase": atual.fase}
