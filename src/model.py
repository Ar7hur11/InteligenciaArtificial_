"""
model.py
Carrega e configura o modelo YOLO para treinamento ou inferência.
"""
from ultralytics import YOLO
from pathlib import Path


MODELOS_DISPONIVEIS = {
    "nano":   "yolov8n.pt",   # mais leve, ideal para teste
    "small":  "yolov8s.pt",
    "medium": "yolov8m.pt",
    "large":  "yolov8l.pt",
    "xlarge": "yolov8x.pt",   # mais pesado, mais preciso
}


def carregar_modelo(tamanho: str = "nano", pesos: str = None) -> YOLO:
    """
    Carrega um modelo YOLO.
    - tamanho: 'nano', 'small', 'medium', 'large', 'xlarge'
    - pesos: caminho para um .pt salvo (para continuar treino ou inferência)
    """
    if pesos and Path(pesos).exists():
        print(f"Carregando modelo salvo: {pesos}")
        return YOLO(pesos)

    nome = MODELOS_DISPONIVEIS.get(tamanho, "yolov8n.pt")
    print(f"Carregando modelo pré-treinado: {nome}")
    return YOLO(nome)


def info_modelo(model: YOLO):
    """Exibe informações básicas do modelo."""
    model.info()
