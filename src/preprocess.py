import cv2
import os
import albumentations as A
from glob import glob
import shutil
import numpy as np

# Sempre aponta para a raiz do projeto (InteligenciaArtificial_), independente de onde o script é rodado
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(BASE_DIR, "data", "raw")
OUTPUT_IMG_DIR = os.path.join(BASE_DIR, "data", "processed", "images")
OUTPUT_LBL_DIR = os.path.join(BASE_DIR, "data", "labels")
TARGET_SIZE = (640, 640)

# Pré-processamento: apenas redimensiona mantendo proporção + padding cinza YOLO
# Augmentation (distorção, brilho) é responsabilidade do YOLO durante o treino
transform = A.Compose([
    A.LongestMaxSize(max_size=max(TARGET_SIZE), interpolation=cv2.INTER_AREA),
    A.PadIfNeeded(
        min_height=TARGET_SIZE[1],
        min_width=TARGET_SIZE[0],
        border_mode=cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    ),
])

# Mapeamento: possíveis nomes de pasta na raw -> nome do split no YOLO
SPLITS = {
    "train":      "train",
    "training":   "train",
    "val":        "val",
    "valid":      "val",
    "validation": "val",
    "test":       "test",
    "testing":    "test",
}

def processar_split(pasta_origem, split_destino):
    arquivos = glob(os.path.join(INPUT_DIR, pasta_origem, "**", "*.png"), recursive=True)

    if not arquivos:
        print(f"[AVISO] Nenhuma imagem encontrada em: {pasta_origem}")
        return

    img_target = os.path.join(OUTPUT_IMG_DIR, split_destino)
    lbl_target = os.path.join(OUTPUT_LBL_DIR, split_destino)
    os.makedirs(img_target, exist_ok=True)
    os.makedirs(lbl_target, exist_ok=True)

    print(f"\n--- Processando '{pasta_origem}' -> '{split_destino}' ({len(arquivos)} imagens) ---")

    for img_path in arquivos:
        print(f"Lendo: {os.path.basename(img_path)}")

        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)

        if img is None:
            print(f"!! ERRO AO LER IMAGEM: {img_path}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        augmented = transform(image=img_rgb)
        final_img = cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)

        nome_base = os.path.basename(img_path)
        caminho_final_img = os.path.join(img_target, nome_base)

        is_success, im_buf_arr = cv2.imencode(".png", final_img)
        if is_success:
            im_buf_arr.tofile(caminho_final_img)
            print(f"-> Salvo: {caminho_final_img}")

        txt_origem = img_path.replace(".png", ".txt")
        if os.path.exists(txt_origem):
            shutil.copy(txt_origem, os.path.join(lbl_target, nome_base.replace(".png", ".txt")))
            print(f"-> Label copiado.")

def processar():
    pastas_raw = [p for p in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, p))]

    if not pastas_raw:
        print(f"Nenhuma pasta encontrada em: {INPUT_DIR}")
        return

    processados = set()
    for pasta in pastas_raw:
        split = SPLITS.get(pasta.lower())
        if split and split not in processados:
            processar_split(pasta, split)
            processados.add(split)
        elif not split:
            print(f"[AVISO] Pasta '{pasta}' não reconhecida, ignorando.")

    print("\n=== Pré-processamento concluído! ===")

if __name__ == "__main__":
    processar()