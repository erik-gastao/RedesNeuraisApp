"""Gráficos do treinamento (parte opcional do projeto).

Usa matplotlib só para desenhar: nenhum cálculo da rede passa por aqui.
O matplotlib é importado dentro da função, então o resto do programa
funciona mesmo sem ele instalado.
"""

import warnings
from pathlib import Path


def grafico_do_erro(historico, caminho, titulo="Erro por época", mostrar=True,
                    historico_validacao=None):
    """Desenha a curva do erro (MSE) ao longo das épocas.

    historico: lista com o erro do treino em cada época
    historico_validacao: lista com o erro da validação em cada época
               (opcional); quando dada, vira uma segunda curva
    caminho:   arquivo .png onde a figura é salva (a pasta é criada se faltar)
    titulo:    texto no topo do gráfico
    mostrar:   se True, também abre a janela do gráfico; o programa continua
               quando ela for fechada

    Como ler: o eixo horizontal é a época e o vertical é o erro. A curva
    descendo mostra a rede aprendendo; quando fica reta, mais épocas com a
    mesma configuração quase não ajudam.

    Levanta ImportError se o matplotlib não estiver instalado.
    """
    import matplotlib.pyplot as plt

    epocas = range(1, len(historico) + 1)

    figura, eixo = plt.subplots(figsize=(8, 4.5))
    eixo.plot(epocas, historico, color="tab:blue", linewidth=2, label="treino")
    if historico_validacao:
        # Validação subindo enquanto o treino cai = a rede está decorando.
        eixo.plot(epocas, historico_validacao, color="tab:orange", linewidth=2,
                  label="validação")
        eixo.legend()
    eixo.set_title(titulo)
    eixo.set_xlabel("Época")
    eixo.set_ylabel("Erro (MSE)")
    eixo.set_ylim(bottom=0)
    eixo.grid(True, alpha=0.3)

    # Marca o erro final, que é o número mostrado na avaliação do treino.
    eixo.annotate(
        f"erro final: {historico[-1]:.4f}",
        xy=(len(historico), historico[-1]),
        xytext=(-10, 25),
        textcoords="offset points",
        ha="right",
        arrowprops={"arrowstyle": "->", "color": "gray"},
    )

    figura.tight_layout()

    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(caminho, dpi=150)

    if mostrar:
        # Em ambiente sem tela (ex.: testes automáticos) o matplotlib só
        # avisa que não pode abrir janela; o arquivo já foi salvo.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            plt.show()
    plt.close(figura)
    return caminho
