"""
evaluate.py
Avalia o modelo treinado e exibe métricas detalhadas.
"""
from model import carregar_modelo


PESOS = "runs/train/treino_v1/weights/best.pt"
DATA_YAML = "data.yaml"


def avaliar(pesos: str = PESOS):
    model = carregar_modelo(pesos=pesos)

    metricas = model.val(data=DATA_YAML)

    print("\n── Métricas ──────────────────────")
    print(f"  mAP50:     {metricas.box.map50:.4f}")
    print(f"  mAP50-95:  {metricas.box.map:.4f}")
    print(f"  Precision: {metricas.box.mp:.4f}")
    print(f"  Recall:    {metricas.box.mr:.4f}")
    print("──────────────────────────────────\n")

    return metricas


if __name__ == "__main__":
    avaliar()
