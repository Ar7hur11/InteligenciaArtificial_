"""
train.py
Script principal de treinamento do modelo YOLO.
"""
from model import carregar_modelo
from pathlib import Path


# ── Configurações ──────────────────────────────────────────
DATA_YAML   = "data.yaml"
TAMANHO     = "nano"      # nano | small | medium | large | xlarge
EPOCAS      = 50
TAMANHO_IMG = 640
BATCH       = 16
NOME_TREINO = "treino_v1"
# ───────────────────────────────────────────────────────────


def treinar():
    model = carregar_modelo(tamanho=TAMANHO)

    resultados = model.train(
        data=DATA_YAML,
        epochs=EPOCAS,
        imgsz=TAMANHO_IMG,
        batch=BATCH,
        name=NOME_TREINO,
        project="runs/train",
        patience=10,       # early stopping
        save=True,
        plots=True,
    )

    print("\nTreinamento concluído!")
    print(f"Melhor modelo salvo em: runs/train/{NOME_TREINO}/weights/best.pt")
    return resultados


if __name__ == "__main__":
    treinar()
