"""
predict.py
Faz predições em imagens novas usando o modelo treinado.
"""
from model import carregar_modelo
from pathlib import Path
import sys


PESOS = "runs/train/treino_v1/weights/best.pt"
CONFIANCA = 0.5


def prever(caminho_imagem: str, pesos: str = PESOS, salvar: bool = True):
    """
    Roda o modelo em uma imagem e exibe/salva o resultado.
    - caminho_imagem: path para a imagem ou pasta de imagens
    - salvar: se True, salva o resultado em runs/predict/
    -teste
    """
    model = carregar_modelo(pesos=pesos)

    resultados = model.predict(
        source=caminho_imagem,
        conf=CONFIANCA,
        save=salvar,
        project="runs/predict",
    )

    for r in resultados:
        print(f"\nImagem: {r.path}")
        print(f"  Detecções: {len(r.boxes)}")
        for box in r.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            print(f"    Classe {cls} — confiança {conf:.2f}")

    return resultados


if __name__ == "__main__":
    imagem = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    prever(imagem)
