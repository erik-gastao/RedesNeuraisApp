"""Textos das fórmulas mostradas no painel do player (sem pygame).

Cada função devolve (formula, formula_numeros): a fórmula genérica e a
mesma fórmula com os números do passo. Os índices aparecem a partir de 1,
como no quadro (w₁, a₁, z₂...), e números negativos ficam entre parênteses.
"""

SUBSCRITOS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

# Como cada ativação aparece na fórmula.
NOMES = {"sigmoid": "σ", "tanh": "tanh", "relu": "ReLU", "linear": "linear"}


def sub(indice):
    """Índice começando em 0 → subscrito começando em 1 (0 → '₁')."""
    return str(indice + 1).translate(SUBSCRITOS)


def numero(valor, casas=2):
    """0.5 → '0.50'; −1.2 → '(−1.20)'."""
    texto = f"{abs(valor):.{casas}f}"
    if valor < 0 and float(texto) != 0:
        return f"(−{texto})"
    return texto


def lista(valores, casas=2):
    """[0.5, −1] → '(0.50, (−1.00))'."""
    return "(" + ", ".join(numero(v, casas) for v in valores) + ")"


def formula_entrada(x):
    return "x", f"x = {lista(x)}"


def formula_normalizacao(j, original, minimo, maximo, normalizado):
    return (
        "x' = (x − mín) / (máx − mín)",
        f"x'{sub(j)} = ({numero(original)} − {numero(minimo)}) / "
        f"({numero(maximo)} − {numero(minimo)}) = {numero(normalizado)}",
    )


def formula_ligacao(j, w, a):
    return "w·a", f"w{sub(j)}·a{sub(j)} = {numero(w)}·{numero(a)} = {numero(w * a)}"


def formula_soma(n, produtos, soma):
    termos = " + ".join(numero(p) for p in produtos)
    return "Σ w·a", f"Σ{sub(n)} = {termos} = {numero(soma)}"


def formula_bias(n, soma, b, z):
    return "z = Σ w·a + b", f"z{sub(n)} = {numero(soma)} + {numero(b)} = {numero(z)}"


def formula_ativacao(n, nome, z, a):
    f = NOMES[nome]
    return f"a = {f}(z)", f"a{sub(n)} = {f}({numero(z)}) = {numero(a)}"


def formula_neuronio(n, nome, pesos, entradas, b, z, a):
    f = NOMES[nome]
    termos = " + ".join(f"{numero(w)}·{numero(x)}" for w, x in zip(pesos, entradas))
    return (
        f"z = Σ w·a + b;  a = {f}(z)",
        f"z{sub(n)} = {termos} + {numero(b)} = {numero(z)};  "
        f"a{sub(n)} = {f}({numero(z)}) = {numero(a)}",
    )


def formula_camada(nome, saidas):
    return f"a = {NOMES[nome]}(W·a + b)", f"a = {lista(saidas)}"


def formula_erro(y, y_hat, erro):
    termos = " + ".join(f"({numero(yi)} − {numero(yh)})²" for yi, yh in zip(y, y_hat))
    return "E = Σ (y − ŷ)²", f"E = {termos} = {numero(erro, 4)}"


def formula_delta_saida(n, dE_da, derivada, delta):
    return (
        "δ = ∂E/∂ŷ · f'(z)",
        f"δ{sub(n)} = {numero(dE_da, 4)} · {numero(derivada, 4)} = {numero(delta, 4)}",
    )


def formula_delta_oculta(n, dE_da, derivada, delta):
    return (
        "δ = f'(z) · Σ δ·w",
        f"δ{sub(n)} = {numero(derivada, 4)} · {numero(dE_da, 4)} = {numero(delta, 4)}",
    )


def formula_camada_delta(deltas):
    return "δ = f'(z) · ∂E/∂a", f"δ = {lista(deltas, 4)}"


def formula_gradiente(j, delta, a, gradiente):
    return (
        "∂E/∂w = δ·a",
        f"∂E/∂w{sub(j)} = {numero(delta, 4)}·{numero(a)} = {numero(gradiente, 4)}",
    )


def formula_ajuste(taxa):
    return "w = w − η·δ·a;  b = b − η·δ", f"η = {taxa:g}"


def formula_compacto(y, y_hat, erro):
    return (
        "forward → δ → ajuste",
        f"ŷ = {lista(y_hat)}  ·  y = {lista(y)}  ·  E = {numero(erro, 4)}",
    )
