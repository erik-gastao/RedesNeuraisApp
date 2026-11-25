"""Hub: escolher a base e montar a rede (seção 6.1 do PLANO_INTERFACE.md).

Os textos dos campos ficam num FormularioHub; a cada mudança, validar()
(config_hub.py) confere tudo, mostra os erros ao lado de cada campo e só
libera "Gerar" quando não há erro. A tela é refeita (construir) quando muda
o número de campos: base escolhida, camada adicionada/removida, janela
redimensionada.
"""

from pathlib import Path

import pygame
import pygame_gui
from pygame_gui.elements import UIButton, UIDropDownMenu, UILabel, UITextEntryLine

from dados import descrever_base, listar_bases
from experimento import Experimento
from interface.config_hub import (
    FormularioHub,
    aviso_saida,
    numero_de_entradas_e_saidas,
    previa,
    resumo_base,
    sugestao_ativacao_saida,
    texto_divisao,
    validar,
)
from rede.ativacoes import ATIVACOES

PASTA_DO_PROJETO = Path(__file__).resolve().parent.parent
PASTAS_DE_BASES = [PASTA_DO_PROJETO / "bases", PASTA_DO_PROJETO / "Datasets"]
ESCOLHA = "Escolha uma base…"
NOMES_NIVEIS = {"ligacao": "ligação", "neuronio": "neurônio", "camada": "camada"}
COR_FUNDO = (248, 248, 246)
ALTURA_LINHA = 40


class TelaHub:
    def __init__(self, app, formulario=None):
        self.app = app
        self.form = formulario or FormularioHub()
        self.descricao = None
        self.erro_base = ""
        self.erros = {}
        self.parametros = None
        self.elementos = []
        self.ao_mudar_texto = {}     # caixa de texto → função que guarda o texto
        self.ao_escolher = {}        # lista suspensa → função que guarda a opção
        self.ao_clicar = {}          # botão → função
        self.rotulos_erro = {}
        self.bases = {f"{p.name}  ({p.parent.name})": p for p in listar_bases(PASTAS_DE_BASES)}
        if self.form.base:
            self._descrever()
        self.construir()

    # ------------------------------------------------------------ criação

    def _novo(self, elemento):
        self.elementos.append(elemento)
        return elemento

    def _rotulo(self, x, y, largura, conteudo, estilo=None):
        return self._novo(UILabel(pygame.Rect(x, y, largura, 32), conteudo, self.app.manager,
                                  object_id=estilo))

    def _caixa(self, x, y, largura, valor, guardar):
        caixa = self._novo(UITextEntryLine(pygame.Rect(x, y, largura, 32), self.app.manager))
        caixa.set_text(valor)
        self.ao_mudar_texto[caixa] = guardar
        return caixa

    def _lista(self, x, y, largura, opcoes, escolhida, guardar):
        lista = self._novo(UIDropDownMenu(opcoes, escolhida, pygame.Rect(x, y, largura, 32),
                                          self.app.manager))
        self.ao_escolher[lista] = guardar
        return lista

    def _botao(self, x, y, largura, conteudo, acao):
        botao = self._novo(UIButton(pygame.Rect(x, y, largura, 32), conteudo, self.app.manager))
        self.ao_clicar[botao] = acao
        return botao

    def _erro(self, campo, x, y, largura=380):
        self.rotulos_erro[campo] = self._rotulo(x, y, largura, "", "#erro")

    def construir(self):
        """Cria todos os campos a partir do formulário atual."""
        self.fechar()
        self.ao_mudar_texto, self.ao_escolher, self.ao_clicar, self.rotulos_erro = {}, {}, {}, {}
        f = self.form
        x_rotulo, x_campo = 30, 200
        y = 16
        self._rotulo(x_rotulo, y, 600, "Construtor de Redes Neurais", "#titulo")
        y += 44

        # Base.
        self._rotulo(x_rotulo, y, 160, "BASE", "#secao")
        atual = next((nome for nome, p in self.bases.items()
                      if f.base and p.resolve() == Path(f.base).resolve()), ESCOLHA)
        self._lista(x_campo, y, 280, [ESCOLHA] + list(self.bases), atual, self.escolher_base_pelo_nome)
        self.rotulo_base = self._rotulo(x_campo + 295, y, 460, "")
        self._erro("base", x_campo + 295, y)
        y += ALTURA_LINHA
        self._rotulo(x_rotulo, y, 160, "Colunas de saída")
        caixa = self._caixa(x_campo, y, 60, f.colunas_saida, lambda t: setattr(f, "colunas_saida", t))
        if self.descricao is not None and self.descricao["classes"] is not None:
            caixa.disable()   # classe em texto: uma saída por classe, sem editar
        self._erro("colunas_saida", x_campo + 75, y)
        y += ALTURA_LINHA

        # Camadas ocultas.
        self._rotulo(x_rotulo, y, 160, "CAMADAS OCULTAS", "#secao")
        nomes = list(ATIVACOES)
        for indice, (neuronios, ativacao) in enumerate(f.ocultas):
            self._rotulo(x_campo, y, 80, f"Camada {indice + 1}:")
            self._caixa(x_campo + 85, y, 60, neuronios,
                        lambda t, i=indice: f.ocultas[i].__setitem__(0, t))
            self._lista(x_campo + 155, y, 120, nomes, ativacao,
                        lambda t, i=indice: f.ocultas[i].__setitem__(1, t))
            self._botao(x_campo + 285, y, 34, "−", lambda i=indice: self.remover_camada(i))
            y += ALTURA_LINHA
        self._botao(x_campo, y, 190, "+ adicionar camada", self.adicionar_camada)
        self._erro("ocultas", x_campo + 330, y - ALTURA_LINHA if f.ocultas else y)
        y += ALTURA_LINHA

        # Saída.
        self._rotulo(x_rotulo, y, 160, "SAÍDA", "#secao")
        self.rotulo_saida = self._rotulo(x_campo, y, 140, "")
        self._lista(x_campo + 145, y, 120, nomes, f.ativacao_saida,
                    lambda t: setattr(f, "ativacao_saida", t))
        y += ALTURA_LINHA - 8
        self.rotulo_aviso = self._rotulo(x_campo, y, 1050, "", "#aviso")
        y += ALTURA_LINHA - 4

        # Treino.
        self._rotulo(x_rotulo, y, 160, "TREINO", "#secao")
        self.botao_bias = self._botao(x_campo, y, 120, "", self.alternar_bias)
        self._rotulo(x_campo + 135, y, 70, "Semente")
        self._caixa(x_campo + 205, y, 60, f.semente, lambda t: setattr(f, "semente", t))
        self._rotulo(x_campo + 280, y, 70, "Taxa (η)")
        self._caixa(x_campo + 350, y, 90, f.taxa, lambda t: setattr(f, "taxa", t))
        self._rotulo(x_campo + 455, y, 60, "Épocas")
        self._caixa(x_campo + 515, y, 80, f.epocas, lambda t: setattr(f, "epocas", t))
        y += ALTURA_LINHA - 6
        for campo, deslocamento in (("semente", 135), ("taxa", 280), ("epocas", 455)):
            self._erro(campo, x_campo + deslocamento, y, 300)
        y += ALTURA_LINHA - 8

        # Divisão.
        self._rotulo(x_rotulo, y, 160, "DIVISÃO (%)", "#secao")
        for indice, nome in enumerate(("Treino", "Validação", "Teste")):
            x = x_campo + indice * 150
            self._rotulo(x, y, 80, nome)
            self._caixa(x + 80, y, 55, f.divisao[indice],
                        lambda t, i=indice: f.divisao.__setitem__(i, t))
        self.rotulo_divisao = self._rotulo(x_campo + 455, y, 260, "")
        self._erro("divisao", x_campo + 720, y)
        y += ALTURA_LINHA

        # Normalização.
        self._rotulo(x_rotulo, y, 160, "NORMALIZAÇÃO", "#secao")
        self.botao_normalizar = self._botao(x_campo, y, 300, "", self.alternar_normalizar)
        self._rotulo(x_campo + 315, y, 200, "Lote (vazio = treino inteiro)")
        caixa = self._caixa(x_campo + 520, y, 70, f.lote, lambda t: setattr(f, "lote", t))
        if not f.normalizar:
            caixa.disable()
        self._erro("lote", x_campo + 600, y)
        y += ALTURA_LINHA

        # Detalhe do player.
        self._rotulo(x_rotulo, y, 160, "DETALHE", "#secao")
        self.botoes_nivel = {}
        for indice, nivel in enumerate(NOMES_NIVEIS):
            self.botoes_nivel[nivel] = self._botao(x_campo + indice * 130, y, 120, "",
                                                   lambda n=nivel: self.escolher_nivel(n))
        y += ALTURA_LINHA
        self._rotulo(x_rotulo, y, 160, "Épocas detalhadas")
        self._caixa(x_campo, y, 200, f.epocas_detalhadas,
                    lambda t: setattr(f, "epocas_detalhadas", t))
        self._rotulo(x_campo + 215, y, 130, "Exemplo em foco")
        self._caixa(x_campo + 345, y, 60, f.exemplo_em_foco,
                    lambda t: setattr(f, "exemplo_em_foco", t))
        y += ALTURA_LINHA - 6
        self._erro("epocas_detalhadas", x_campo, y, 210)
        self._erro("exemplo_em_foco", x_campo + 215, y)
        y += ALTURA_LINHA

        # Prévia e ações.
        self.rotulo_previa = self._rotulo(x_rotulo, y, 900, "", "#previa")
        y += ALTURA_LINHA
        self.botao_gerar = self._botao(x_rotulo, y, 220, "Gerar e abrir a rede", self.gerar)
        self.botao_carregar = self._botao(x_rotulo + 235, y, 180, "Carregar rede…", lambda: None)
        self.botao_carregar.disable()   # etapa 9 do plano
        self.rotulo_status = self._rotulo(x_rotulo + 430, y, 700, "", "#erro")
        self.validar()

    def fechar(self):
        for elemento in self.elementos:
            elemento.kill()
        self.elementos = []

    def redimensionar(self):
        self.construir()

    # ------------------------------------------------------------- ações

    def _descrever(self):
        try:
            self.descricao = descrever_base(self.form.base)
            self.erro_base = ""
        except (OSError, ValueError) as erro:
            self.descricao = None
            self.erro_base = f"Não foi possível ler a base: {erro}"

    def escolher_base_pelo_nome(self, nome):
        self.escolher_base(self.bases.get(nome))

    def escolher_base(self, caminho):
        self.form.base = str(Path(caminho).resolve()) if caminho else None
        self.descricao = None
        if caminho:
            self._descrever()
            if self.descricao is not None:
                self.form.ativacao_saida = sugestao_ativacao_saida(self.descricao)
        self.construir()

    def adicionar_camada(self):
        self.form.ocultas.append(["4", "sigmoid"])
        self.construir()

    def remover_camada(self, indice):
        self.form.ocultas.pop(indice)
        self.construir()

    def alternar_bias(self):
        self.form.usar_bias = not self.form.usar_bias
        self.validar()

    def alternar_normalizar(self):
        self.form.normalizar = not self.form.normalizar
        self.construir()

    def escolher_nivel(self, nivel):
        self.form.nivel = nivel
        self.validar()

    def validar(self):
        """Confere o formulário e atualiza erros, textos e o botão Gerar."""
        f = self.form
        self.parametros, self.erros = validar(f, self.descricao)
        if self.erro_base:
            self.erros["base"] = self.erro_base
        for campo, rotulo in self.rotulos_erro.items():
            rotulo.set_text(self.erros.get(campo, ""))

        if self.descricao is not None:
            self.rotulo_base.set_text(resumo_base(f, self.descricao))
            self.rotulo_divisao.set_text(texto_divisao(f, self.descricao))
            self.rotulo_aviso.set_text(aviso_saida(f, self.descricao) or "")
            self.rotulo_previa.set_text("Prévia:  " + (previa(f, self.descricao) or "—"))
            try:
                _, n_saidas = numero_de_entradas_e_saidas(f, self.descricao)
                self.rotulo_saida.set_text(f"{n_saidas} neurônio(s)")
            except ValueError:
                self.rotulo_saida.set_text("")
        else:
            for rotulo in (self.rotulo_base, self.rotulo_divisao, self.rotulo_aviso,
                           self.rotulo_saida):
                rotulo.set_text("")
            self.rotulo_previa.set_text("Prévia:  escolha uma base")

        self.botao_bias.set_text(f"Bias: {'sim' if f.usar_bias else 'não'}")
        self.botao_normalizar.set_text(
            f"Normalizar entradas (min-max): {'sim' if f.normalizar else 'não'}")
        for nivel, botao in self.botoes_nivel.items():
            marca = "●" if f.nivel == nivel else "○"
            botao.set_text(f"{marca} {NOMES_NIVEIS[nivel]}")
        if self.parametros is None:
            self.botao_gerar.disable()
        else:
            self.botao_gerar.enable()

    def gerar(self):
        """Monta o experimento e abre a rede (na etapa 5: grava e abre o player)."""
        if self.parametros is None:
            return
        p = self.parametros
        try:
            experimento = Experimento.da_base(
                p["caminho"], p["ocultas"], p["ativacao_saida"], p["usar_bias"], p["seed"],
                n_saidas=p["n_saidas"], percentuais=p["percentuais"],
                normalizar=p["normalizar"], tamanho_lote=p["tamanho_lote"],
            )
        except (OSError, ValueError) as erro:
            self.rotulo_status.set_text(f"Não foi possível montar a rede: {erro}")
            return
        from interface.tela_rede import TelaRede
        self.app.ultimo_formulario = self.form
        self.app.ultima_descricao = self.descricao
        self.app.ultima_previa = previa(self.form, self.descricao)
        self.app.trocar_tela(TelaRede(self.app, experimento, p))

    # ------------------------------------------------------------- eventos

    def tratar_evento(self, evento):
        if evento.type == pygame_gui.UI_TEXT_ENTRY_CHANGED and evento.ui_element in self.ao_mudar_texto:
            self.ao_mudar_texto[evento.ui_element](evento.text)
            self.validar()
        elif evento.type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED and evento.ui_element in self.ao_escolher:
            self.ao_escolher[evento.ui_element](evento.text)
            if evento.ui_element in self.elementos:   # a base refaz a tela; as outras não
                self.validar()
        elif evento.type == pygame_gui.UI_BUTTON_PRESSED and evento.ui_element in self.ao_clicar:
            self.ao_clicar[evento.ui_element]()

    def atualizar(self, dt):
        pass

    def desenhar(self, superficie):
        superficie.fill(COR_FUNDO)
