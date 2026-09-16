"""Testes puros (sem torch) — rodam no runner hospedado do GitHub."""

from pytorch_gpu_sandbox.formatting import linha, secao, titulo


def test_linha_repete_caractere_ate_o_tamanho(capsys):
    linha("=", 10)
    assert capsys.readouterr().out == "=" * 10 + "\n"


def test_titulo_centraliza_texto_entre_duas_linhas(capsys):
    titulo("OI")
    saida = capsys.readouterr().out.splitlines()
    assert saida[0] == "=" * 70
    assert saida[1] == "OI".center(70)
    assert saida[2] == "=" * 70


def test_secao_imprime_texto_entre_tracejados(capsys):
    secao("Bloco")
    saida = capsys.readouterr().out.splitlines()
    assert saida[-2] == "Bloco"
    assert saida[-1] == "-" * 70
