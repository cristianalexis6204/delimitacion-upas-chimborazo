import sys
import os

print("[Runner Colab A100] Iniciando ejecucion remota de colab_a100_san_andres_v100_master.py (Kaizen v100 - CORONACION CENTENARIA LADM ISO 19152, 0.000000% Solape & Atlas en Duplas)...")
script_path = "/content/colab_a100_san_andres_v100_master.py"
if os.path.exists(script_path):
    with open(script_path, "r", encoding="utf-8") as f:
        code = f.read()
    exec(code, globals())
else:
    print(f"ERROR: {script_path} no existe en /content")
