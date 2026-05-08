"""
data_loader.py
Responsável por carregar, verificar e preparar o dataset de imagens.
"""
import os
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt


DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed" / "images"


def contar_imagens(pasta: Path) -> dict:
    """Conta imagens por extensão em uma pasta."""
    extensoes = [".jpg", ".jpeg", ".png", ".bmp"]
    contagem = {}
    for ext in extensoes:
        arquivos = list(pasta.rglob(f"*{ext}"))
        if arquivos:
            contagem[ext] = len(arquivos)
    return contagem


def inspecionar_dataset(pasta: Path = RAW_DIR):
    """Mostra um resumo do dataset: total de imagens, resoluções, classes."""
    print(f"\n{'='*50}")
    print(f"  Inspecionando: {pasta}")
    print(f"{'='*50}")

    contagem = contar_imagens(pasta)
    total = sum(contagem.values())
    print(f"\nTotal de imagens: {total}")
    for ext, n in contagem.items():
        print(f"  {ext}: {n}")

    # Classes (subpastas)
    classes = [d.name for d in pasta.iterdir() if d.is_dir()]
    if classes:
        print(f"\nClasses encontradas ({len(classes)}):")
        for c in classes:
            n = len(list((pasta / c).rglob("*.jpg"))) + \
                len(list((pasta / c).rglob("*.png")))
            print(f"  {c}: {n} imagens")
    else:
        print("\nNenhuma subpasta encontrada (sem classes separadas por pasta).")

    print(f"{'='*50}\n")
    return classes


def mostrar_amostras(pasta: Path = RAW_DIR, n: int = 6):
    """Exibe uma grade com amostras do dataset."""
    imagens = list(pasta.rglob("*.jpg")) + list(pasta.rglob("*.png"))
    imagens = imagens[:n]

    if not imagens:
        print("Nenhuma imagem encontrada.")
        return

    fig, axes = plt.subplots(1, len(imagens), figsize=(3 * len(imagens), 3))
    if len(imagens) == 1:
        axes = [axes]

    for ax, caminho in zip(axes, imagens):
        img = Image.open(caminho)
        ax.imshow(img)
        ax.set_title(f"{img.size[0]}×{img.size[1]}", fontsize=9)
        ax.axis("off")

    plt.suptitle("Amostras do dataset", fontsize=12)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    inspecionar_dataset()
