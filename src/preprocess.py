import cv2
import os
import albumentations as A
from glob import glob
import shutil
import numpy as np

# Força o caminho absoluto para não ter erro de localização no Windows
BASE_DIR = os.path.abspath(os.getcwd())
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

def processar():
    # Focando apenas no teste atual: testing -> test
    pasta_origem = "testing"
    split_destino = "test"
    
    arquivos = glob(os.path.join(INPUT_DIR, pasta_origem, "**", "*.png"), recursive=True)
    
    img_target = os.path.join(OUTPUT_IMG_DIR, split_destino)
    lbl_target = os.path.join(OUTPUT_LBL_DIR, split_destino)
    
    # Cria as pastas na marra
    
    os.makedirs(img_target, exist_ok=True)
    os.makedirs(lbl_target, exist_ok=True)

    print(f"--- Iniciando Processamento ---")
    print(f"Arquivos encontrados: {len(arquivos)}")

    for img_path in arquivos:
        print(f"Lendo: {os.path.basename(img_path)}")
        
        # Leitura robusta para Windows (aceita acentos no caminho)
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        
        if img is None:
            print(f"!! ERRO AO LER IMAGEM: {img_path}")
            continue
        
        # Processamento
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        augmented = transform(image=img_rgb)
        final_img = cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)
        
        # Salvamento robusto para Windows
        nome_base = os.path.basename(img_path)
        caminho_final_img = os.path.join(img_target, nome_base)
        
        is_success, im_buf_arr = cv2.imencode(".png", final_img)
        if is_success:
            im_buf_arr.tofile(caminho_final_img)
            print(f"-> Salvo com sucesso em: {caminho_final_img}")
        
        # Copiar TXT
        txt_origem = img_path.replace(".png", ".txt")
        if os.path.exists(txt_origem):
            shutil.copy(txt_origem, os.path.join(lbl_target, nome_base.replace(".png", ".txt")))
            print(f"-> Label copiado.")

if __name__ == "__main__":
    processar()