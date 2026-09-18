"""
Módulo de Descarga y Verificación de Pesos del Modelo Meta SAM ViT-B
TFM UNIR - Cristian Alexis García Pumagualle
"""

import os
import sys
import urllib.request
import time

MODEL_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "models"))
MODEL_PATH = os.path.join(MODEL_DIR, "sam_vit_b_01ec64.pth")

def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration)) if duration > 0 else 0
    percent = min(int(count * block_size * 100 / total_size), 100)
    sys.stdout.write(f"\r[SAM MODEL DOWNLOAD] {percent}% ({progress_size / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB) - {speed} KB/s")
    sys.stdout.flush()

def ensure_sam_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 300000000:
        print(f"[SAM] Checkpoint existente y verificado: {MODEL_PATH} ({os.path.getsize(MODEL_PATH)/(1024*1024):.1f} MB)")
        return MODEL_PATH
        
    print(f"[SAM] Descargando pesos oficiales de Meta SAM ViT-B (357 MB) desde {MODEL_URL}...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH, reporthook)
        print(f"\n[SAM] Descarga finalizada exitosamente: {MODEL_PATH}")
    except Exception as e:
        print(f"\n[SAM ERROR] Error al descargar modelo SAM: {e}")
        raise
        
    return MODEL_PATH

if __name__ == "__main__":
    ensure_sam_model()
