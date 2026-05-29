import cv2
import os
import re
import albumentations as A
from glob import glob
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(BASE_DIR, "data", "raw")
OUTPUT_IMG_DIR = os.path.join(BASE_DIR, "data", "processed", "images")
OUTPUT_LBL_DIR = os.path.join(BASE_DIR, "data", "processed", "labels")
TARGET_SIZE = (640, 640)

# BboxParams faz albumentations transformar as coordenadas junto com a imagem
transform = A.Compose([
    A.LongestMaxSize(max_size=max(TARGET_SIZE), interpolation=cv2.INTER_AREA),
    A.PadIfNeeded(
        min_height=TARGET_SIZE[1],
        min_width=TARGET_SIZE[0],
        border_mode=cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    ),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=0.3))

SPLITS = {
    "train":      "train",
    "training":   "train",
    "val":        "val",
    "valid":      "val",
    "validation": "val",
    "test":       "test",
    "testing":    "test",
}


def parse_annotation(txt_path):
    """Lê o formato customizado do dataset e retorna lista de bboxes (x1,y1,x2,y2)."""
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception:
        return []

    bboxes = []
    for match in re.finditer(r'corners:\s*([\d,\s]+)', content):
        pairs = re.findall(r'(\d+),(\d+)', match.group(1))
        if len(pairs) < 2:
            continue
        xs = [int(p[0]) for p in pairs]
        ys = [int(p[1]) for p in pairs]
        bboxes.append((min(xs), min(ys), max(xs), max(ys)))

    return bboxes


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
        nome_base = os.path.basename(img_path)
        txt_path = img_path.replace(".png", ".txt")

        bboxes_raw = parse_annotation(txt_path) if os.path.exists(txt_path) else []
        if not bboxes_raw:
            print(f"[AVISO] Sem label válido: {nome_base}")
            continue

        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"!! ERRO AO LER IMAGEM: {img_path}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = img_rgb.shape[:2]

        # Garante que as coordenadas não saiam dos limites da imagem
        bboxes_validas = []
        for (x1, y1, x2, y2) in bboxes_raw:
            x1 = max(0, min(x1, w_orig - 1))
            y1 = max(0, min(y1, h_orig - 1))
            x2 = max(x1 + 1, min(x2, w_orig))
            y2 = max(y1 + 1, min(y2, h_orig))
            bboxes_validas.append([x1, y1, x2, y2])

        if not bboxes_validas:
            continue

        # Transforma imagem E coordenadas juntos
        augmented = transform(
            image=img_rgb,
            bboxes=bboxes_validas,
            class_labels=[0] * len(bboxes_validas),
        )

        final_img = cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)
        bboxes_transformadas = augmented['bboxes']

        # Salva imagem
        caminho_final_img = os.path.join(img_target, nome_base)
        is_success, im_buf_arr = cv2.imencode(".png", final_img)
        if is_success:
            im_buf_arr.tofile(caminho_final_img)

        # Salva label em formato YOLO
        if bboxes_transformadas:
            lbl_path = os.path.join(lbl_target, nome_base.replace(".png", ".txt"))
            with open(lbl_path, 'w') as f:
                for (x1, y1, x2, y2) in bboxes_transformadas:
                    x_center = (x1 + x2) / 2 / TARGET_SIZE[0]
                    y_center = (y1 + y2) / 2 / TARGET_SIZE[1]
                    width    = (x2 - x1) / TARGET_SIZE[0]
                    height   = (y2 - y1) / TARGET_SIZE[1]
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            print(f"-> {nome_base}: {len(bboxes_transformadas)} placa(s)")


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