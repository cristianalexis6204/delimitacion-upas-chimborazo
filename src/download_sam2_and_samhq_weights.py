"""
Módulo de Descarga y Verificación de Pesos SOTA: SAM-HQ y Meta SAM-2.1
TFM UNIR - Cristian Alexis García Pumagualle
"""

import os
import sys
import urllib.request
import time

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "models"))

CHECKPOINTS = {
    "sam_vit_b": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "filename": "sam_vit_b_01ec64.pth",
        "min_size": 350000000,
        "desc": "Meta SAM-1 ViT-B (Baseline v57)"
    },
    "sam_hq_vit_b": {
        "url": "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_b.pth",
        "filename": "sam_hq_vit_b.pth",
        "min_size": 350000000,
        "desc": "SysCV SAM-HQ ViT-B (High Quality Token)"
    },
    "sam2.1_hiera_base_plus": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
        "filename": "sam2.1_hiera_base_plus.pt",
        "min_size": 300000000,
        "desc": "Meta SAM-2.1 Hiera Base+ (SOTA Fast)"
    },
    "sam2.1_hiera_large": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
        "filename": "sam2.1_hiera_large.pt",
        "min_size": 800000000,
        "desc": "Meta SAM-2.1 Hiera Large (SOTA Max Accuracy)"
    }
}

def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration)) if duration > 0 else 0
    percent = min(int(count * block_size * 100 / (total_size + 1e-8)), 100)
    sys.stdout.write(f"\r[DESCARGA MODELO] {percent}% ({progress_size / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB) - {speed} KB/s")
    sys.stdout.flush()

def ensure_checkpoint(key):
    os.makedirs(MODEL_DIR, exist_ok=True)
    if key not in CHECKPOINTS:
        raise ValueError(f"Modelo desconocido: {key}")
        
    cfg = CHECKPOINTS[key]
    path = os.path.join(MODEL_DIR, cfg["filename"])
    
    if os.path.exists(path) and os.path.getsize(path) >= cfg["min_size"]:
        print(f"[{key.upper()}] Checkpoint verificado: {path} ({os.path.getsize(path)/(1024*1024):.1f} MB)")
        return path
        
    print(f"\n[{key.upper()}] Descargando {cfg['desc']} desde {cfg['url']}...")
    try:
        urllib.request.urlretrieve(cfg["url"], path, reporthook)
        print(f"\n[{key.upper()}] Descarga completada: {path}")
    except Exception as e:
        print(f"\n[{key.upper()} ERROR] Fallo al descargar: {e}")
        raise
        
    return path

def ensure_all_models():
    paths = {}
    for k in ["sam_vit_b", "sam_hq_vit_b", "sam2.1_hiera_base_plus"]:
        paths[k] = ensure_checkpoint(k)
    return paths

if __name__ == "__main__":
    ensure_all_models()
