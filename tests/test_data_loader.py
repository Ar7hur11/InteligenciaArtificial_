"""Testes básicos para o data_loader."""
import sys
sys.path.append("src")
from pathlib import Path
from data_loader import contar_imagens


def test_contar_imagens_pasta_vazia(tmp_path):
    resultado = contar_imagens(tmp_path)
    assert resultado == {}


def test_contar_imagens_com_jpg(tmp_path):
    (tmp_path / "img1.jpg").touch()
    (tmp_path / "img2.jpg").touch()
    resultado = contar_imagens(tmp_path)
    assert resultado.get(".jpg") == 2
