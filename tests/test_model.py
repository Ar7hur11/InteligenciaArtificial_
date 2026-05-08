"""Testes básicos para o carregamento do modelo."""
import sys
sys.path.append("src")


def test_modelos_disponiveis():
    from model import MODELOS_DISPONIVEIS
    assert "nano" in MODELOS_DISPONIVEIS
    assert "small" in MODELOS_DISPONIVEIS


def test_tamanho_invalido_usa_nano():
    from model import MODELOS_DISPONIVEIS
    resultado = MODELOS_DISPONIVEIS.get("inexistente", "yolov8n.pt")
    assert resultado == "yolov8n.pt"
