"""Testes do persistencia.py: salvar e carregar a rede em JSON (seção 1.11)."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from persistencia import carregar_rede, salvar_rede
from rede.rede_neural import RedeNeural
from rede.treinamento import treinar


class TestePersistencia(unittest.TestCase):

    def setUp(self):
        self.pasta = tempfile.TemporaryDirectory()
        self.caminho = Path(self.pasta.name) / "sub" / "rede.json"
        gerador = np.random.default_rng(0)
        self.X = gerador.uniform(0, 1, size=(10, 3))
        Y = gerador.uniform(0, 1, size=(10, 2))
        self.rede = RedeNeural(3, [(4, "relu"), (3, "tanh")], 2, usar_bias=True, seed=1,
                               ativacao_saida="linear")
        treinar(self.rede, self.X, Y, 5, 0.1, seed=0)

    def tearDown(self):
        self.pasta.cleanup()

    def test_salvar_e_carregar_da_saidas_identicas(self):
        salvar_rede(self.caminho, self.rede)
        carregada, _, _ = carregar_rede(self.caminho)
        np.testing.assert_array_equal(self.rede.forward(self.X), carregada.forward(self.X))

    def test_guarda_arquitetura_ativacoes_e_bias(self):
        salvar_rede(self.caminho, self.rede)
        carregada, _, _ = carregar_rede(self.caminho)
        self.assertEqual(carregada.n_entradas, 3)
        self.assertEqual(carregada.n_saidas, 2)
        self.assertTrue(carregada.usar_bias)
        self.assertEqual(carregada.ativacao_saida, "linear")
        self.assertEqual(
            [(c.n_neuronios, c.nome_ativacao) for c in carregada.camadas],
            [(4, "relu"), (3, "tanh"), (2, "linear")],
        )
        for original, copia in zip(self.rede.camadas, carregada.camadas):
            np.testing.assert_array_equal(original.W, copia.W)
            np.testing.assert_array_equal(original.b, copia.b)

    def test_arquivo_e_json_legivel(self):
        salvar_rede(self.caminho, self.rede)
        dados = json.loads(self.caminho.read_text(encoding="utf-8"))
        self.assertEqual(dados["formato"], "rede-mlp")
        self.assertEqual(dados["versao"], 1)
        self.assertEqual(len(dados["camadas"]), 3)
        self.assertEqual(len(dados["camadas"][0]["pesos"]), 4)      # um por neurônio
        self.assertEqual(len(dados["camadas"][0]["pesos"][0]), 3)   # um por entrada

    def test_guarda_normalizacao_e_classes(self):
        normalizacao = {"funcao": "min-max", "minimos": [0.0, 1.0, 2.0], "maximos": [5.0, 6.0, 7.0]}
        salvar_rede(self.caminho, self.rede, normalizacao, ["a", "b"])
        _, norm, classes = carregar_rede(self.caminho)
        self.assertEqual(norm, normalizacao)
        self.assertEqual(classes, ["a", "b"])

    def test_aceita_normalizacao_com_arrays_numpy(self):
        normalizacao = {"funcao": "min-max", "minimos": np.array([0.0, 1.0, 2.0]),
                        "maximos": np.array([5.0, 6.0, 7.0])}
        salvar_rede(self.caminho, self.rede, normalizacao)
        _, norm, _ = carregar_rede(self.caminho)
        self.assertEqual(norm["minimos"], [0.0, 1.0, 2.0])

    def test_sem_normalizacao_volta_none(self):
        salvar_rede(self.caminho, self.rede)
        _, norm, classes = carregar_rede(self.caminho)
        self.assertIsNone(norm)
        self.assertIsNone(classes)

    def test_arquivo_inexistente(self):
        with self.assertRaises(FileNotFoundError):
            carregar_rede(Path(self.pasta.name) / "nao_existe.json")

    def test_json_corrompido_da_valueerror(self):
        self.caminho.parent.mkdir(parents=True)
        self.caminho.write_text("{ isso não é json", encoding="utf-8")
        with self.assertRaises(ValueError):
            carregar_rede(self.caminho)

    def test_campo_faltando_da_valueerror(self):
        salvar_rede(self.caminho, self.rede)
        dados = json.loads(self.caminho.read_text(encoding="utf-8"))
        del dados["camadas"][1]["vieses"]
        self.caminho.write_text(json.dumps(dados), encoding="utf-8")
        with self.assertRaises(ValueError) as erro:
            carregar_rede(self.caminho)
        self.assertIn("vieses", str(erro.exception))

    def test_formato_errado_da_valueerror(self):
        self.caminho.parent.mkdir(parents=True)
        self.caminho.write_text(json.dumps({"formato": "outra-coisa"}), encoding="utf-8")
        with self.assertRaises(ValueError):
            carregar_rede(self.caminho)

    def gravar_alterado(self, alterar, normalizacao=None, classes=None):
        """Salva a rede, aplica `alterar` ao dicionário do JSON e regrava."""
        salvar_rede(self.caminho, self.rede, normalizacao, classes)
        dados = json.loads(self.caminho.read_text(encoding="utf-8"))
        alterar(dados)
        self.caminho.write_text(json.dumps(dados), encoding="utf-8")

    def assertArquivoInvalido(self, *trechos):
        with self.assertRaises(ValueError) as erro:
            carregar_rede(self.caminho)
        for trecho in trechos:
            self.assertIn(trecho, str(erro.exception))

    NORMALIZACAO = {"funcao": "min-max", "minimos": [0.0, 0.0, 0.0], "maximos": [1.0, 1.0, 1.0]}

    def test_normalizacao_sem_minimos_da_valueerror(self):
        self.gravar_alterado(lambda d: d["normalizacao"].pop("minimos"), self.NORMALIZACAO)
        self.assertArquivoInvalido("normaliza")

    def test_normalizacao_com_tamanho_errado_da_valueerror(self):
        def alterar(d):
            d["normalizacao"]["minimos"] = [5.0]
            d["normalizacao"]["maximos"] = [10.0]
        self.gravar_alterado(alterar, self.NORMALIZACAO)
        self.assertArquivoInvalido("normaliza", "3")

    def test_normalizacao_com_funcao_desconhecida_da_valueerror(self):
        self.gravar_alterado(lambda d: d["normalizacao"].update(funcao="z-score"), self.NORMALIZACAO)
        self.assertArquivoInvalido("z-score")

    def test_classes_com_tamanho_errado_da_valueerror(self):
        self.gravar_alterado(lambda d: d.update(classes=["a"]), classes=["a", "b"])
        self.assertArquivoInvalido("classes")

    def test_n_entradas_zero_da_valueerror(self):
        self.gravar_alterado(lambda d: d.update(n_entradas=0))
        self.assertArquivoInvalido("entradas")

    def test_camada_sem_neuronios_da_valueerror(self):
        self.gravar_alterado(lambda d: d["camadas"][0].update(neuronios=0))
        self.assertArquivoInvalido("neurônio")

    def test_camadas_que_nao_sao_lista_da_valueerror(self):
        self.gravar_alterado(lambda d: d.update(camadas=5))
        with self.assertRaises(ValueError) as erro:
            carregar_rede(self.caminho)
        self.assertNotIn("slice", str(erro.exception))
        self.assertIn("camadas", str(erro.exception))

    def test_arquivo_binario_da_valueerror_em_portugues(self):
        self.caminho.parent.mkdir(parents=True)
        self.caminho.write_bytes(b"\xff\xfe\x00\x81binario")
        self.assertArquivoInvalido("não é um arquivo de texto")

    def test_pesos_com_forma_errada_da_valueerror(self):
        salvar_rede(self.caminho, self.rede)
        dados = json.loads(self.caminho.read_text(encoding="utf-8"))
        dados["camadas"][0]["pesos"][0].append(9.9)   # 4 pesos num neurônio de 3 entradas
        self.caminho.write_text(json.dumps(dados), encoding="utf-8")
        with self.assertRaises(ValueError):
            carregar_rede(self.caminho)


if __name__ == "__main__":
    unittest.main()
