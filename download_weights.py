"""
Script de descarga automática de pesos pre-entrenados del modelo Meta Segment Anything (SAM).
TFM UNIR - Autor: Cristian Alexis García Pumagualle
"""

import os
import sys
import urllib.request

SAM_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
DEFAULT_WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
CHECKPOINT_NAME = "sam_vit_b_01ec64.pth"

def download_sam_weights(target_dir=DEFAULT_WEIGHTS_DIR):
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, CHECKPOINT_NAME)

    if os.path.exists(target_path):
        size_mb = os.path.getsize(target_path) / (1024 * 1024)
        if size_mb > 350:
            print(f"[SAM Weights OK] Pesos ya disponibles en: {target_path} ({size_mb:.1f} MB)")
            return target_path

    print(f"[Descarga SAM] Descargando pesos oficiales de Meta AI (~375 MB)...")
    print(f"URL: {SAM_CHECKPOINT_URL}")
    print(f"Destino: {target_path}")

    def progress_bar(blocks_transferred, block_size, total_size):
        percent = 100.0 * (blocks_transferred * block_size) / total_size
        downloaded_mb = (blocks_transferred * block_size) / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        sys.stdout.write(f"\r  -> Progreso: [{percent:5.1f}%] {downloaded_mb:5.1f} / {total_mb:5.1f} MB")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(SAM_CHECKPOINT_URL, target_path, reporthook=progress_bar)
        print("\n[Descarga SAM Completada Exitosamente]")
        return target_path
    except Exception as e:
        print(f"\n[ERROR] No se pudo descargar automáticamente: {e}")
        print("Por favor descargue manualmente sam_vit_b_01ec64.pth y colóquelo en la carpeta 'weights/'.")
        return None

if __name__ == "__main__":
    download_sam_weights()
