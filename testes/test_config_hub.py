"""Testes da validação do formulário do hub (interface/config_hub.py, sem pygame).

    python -m unittest discover testes
"""

import unittest
from pathlib import Path

from dados import descrever_base
from interface.config_hub import (
    FormularioHub,
    aviso_saida,
    previa,
    resumo_base,
    sugestao_ativacao_saida,
    texto_divisao,
    validar,
)

RAIZ = Path(__file__).resolve().parent.parent
XOR = str(RAIZ / "bases" / "xor.csv")
IRIS = str(RAIZ / "Datasets" / "bezdekIris.data")
DATASET = str(RAIZ / "Datasets" / "dataset.csv")


def formulario(base=IRIS, **campos):
    return FormularioHub(base=base, **campos)


class TesteBaseESaidas(unittest.TestCase):

    def test_iris_saida_vem_das_classes(self):
        descricao = descrever_base(IRIS)
        params, erros = validar(formulario(colunas_saida="7"), descricao)
        self.assertEqual(erros, {})
        self.assertEqual((params["n_entradas"], params["n_saidas"]), (4, 3))

    def test_base_numerica_usa_colunas_de_saida(self):
        descricao = descrever_base(DATASET)
        params, erros = validar(formulario(DATASET, ativacao_saida="linear"), descricao)
        self.assertEqual(erros, {})
        self.assertEqual((params["n_entradas"], params["n_saidas"]), (2, 1))

    def test_colunas_de_saida_demais_e_erro(self):
        _, erros = validar(formulario(DATASET, colunas_saida="3"), descrever_base(DATASET))
        self.assertIn("colunas_saida", erros)

    def test_sem_base_e_erro(self):
        params, erros = validar(FormularioHub(), None)
        self.assertIsNone(params)
        self.assertIn("base", erros)

    def test_resumo_da_base(self):
        self.assertEqual(resumo_base(formulario(), descrever_base(IRIS)),
                         "150 amostras · 4 entradas · 3 classes")
        self.assertEqual(resumo_base(formulario(DATASET), descrever_base(DATASET)),
                         "179 amostras · 2 entradas · saída numérica")

    def test_sugestao_da_ativacao_de_saida(self):
        self.assertEqual(sugestao_ativacao_saida(descrever_base(IRIS)), "sigmoid")
        self.assertEqual(sugestao_ativacao_saida(descrever_base(XOR)), "sigmoid")
        self.assertEqual(sugestao_ativacao_saida(descrever_base(DATASET)), "linear")

    def test_aviso_de_saida_numerica_com_sigmoid(self):
        aviso = aviso_saida(formulario(DATASET, ativacao_saida="sigmoid"), descrever_base(DATASET))
        self.assertIn("Use a ativação linear na saída.", aviso)
        self.assertIsNone(aviso_saida(formulario(DATASET, ativacao_saida="linear"),
                                      descrever_base(DATASET)))
        self.assertIsNone(aviso_saida(formulario(), descrever_base(IRIS)))


class TesteCampos(unittest.TestCase):

    def setUp(self):
        self.descricao = descrever_base(IRIS)

    def test_valores_padrao_sao_validos(self):
        params, erros = validar(formulario(), self.descricao)
        self.assertEqual(erros, {})
        self.assertEqual(params["ocultas"], [(5, "sigmoid")])
        self.assertEqual(params["seed"], 1)
        self.assertEqual(params["percentuais"], (70.0, 15.0, 15.0))
        self.assertEqual(params["epocas_detalhadas"], [1, 2, 10, 100, "fim"])
        self.assertEqual(params["exemplo_em_foco"], 0)   # o campo começa em 1
        self.assertIsNone(params["tamanho_lote"])

    def test_cada_campo_invalido_tem_sua_mensagem(self):
        casos = {
            "ocultas": dict(ocultas=[["0", "sigmoid"]]),
            "semente": dict(semente="-3"),
            "taxa": dict(taxa="zero"),
            "epocas": dict(epocas="0"),
            "divisao": dict(divisao=["70", "20", "20"]),
            "lote": dict(lote="1"),
            "epocas_detalhadas": dict(epocas_detalhadas="1, dois"),
            "exemplo_em_foco": dict(exemplo_em_foco="999"),
        }
        for campo, valores in casos.items():
            params, erros = validar(formulario(**valores), self.descricao)
            self.assertIsNone(params, campo)
            self.assertIn(campo, erros, campo)
            self.assertTrue(erros[campo], campo)

    def test_taxa_aceita_virgula_e_semente_vazia_e_aleatoria(self):
        params, _ = validar(formulario(taxa="0,05", semente=""), self.descricao)
        self.assertEqual(params["taxa"], 0.05)
        self.assertIsNone(params["seed"])

    def test_epoca_detalhada_alem_do_total_e_erro(self):
        _, erros = validar(formulario(epocas="50", epocas_detalhadas="1, 100"), self.descricao)
        self.assertIn("epocas_detalhadas", erros)

    def test_lote_so_conta_com_normalizacao(self):
        params, erros = validar(formulario(normalizar=False, lote="1"), self.descricao)
        self.assertEqual(erros, {})
        self.assertIsNone(params["tamanho_lote"])

    def test_divisao_mostra_os_tamanhos(self):
        self.assertEqual(texto_divisao(formulario(), self.descricao), "→ 106 / 22 / 22")
        self.assertEqual(texto_divisao(formulario(base=XOR), descrever_base(XOR)),
                         "base pequena: não é dividida")

    def test_previa_da_rede(self):
        self.assertEqual(previa(formulario(), self.descricao),
                         "4 → [5 sigmoid] → 3 sigmoid   (35 pesos + 8 bias)")
        self.assertEqual(previa(formulario(usar_bias=False, ocultas=[]), self.descricao),
                         "4 → 3 sigmoid   (12 pesos)")


if __name__ == "__main__":
    unittest.main()
