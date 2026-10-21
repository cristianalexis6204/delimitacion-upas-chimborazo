"""
SYNC EXPERIMENT VISUALS (TFM UNIR)
Autor: Cristian Alexis García Pumagualle

Sincroniza y copia todas las láminas cartográficas y figuras a 300 DPI
directamente a la carpeta del experimento actual (v60.0.0).
"""

import os
import shutil

def sync_visuals():
    base_src = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.dirname(base_src)
    root_tfm = os.path.abspath(os.path.join(repo_dir, ".."))
    
    source_dirs = [
        os.path.join(repo_dir, "outputs"),
        os.path.join(root_tfm, "06_Codigo_y_Experimentos", "reportes", "figuras")
    ]
    
    target_dirs = [
        os.path.join(root_tfm, "06_Codigo_y_Experimentos", "experimentos", "v60.0.0"),
        os.path.join(root_tfm, "06_Codigo_y_Experimentos", "experimentos", "v60.0.0", "outputs")
    ]
    
    images_to_sync = [
        "mapa_politico_provincial_cantonal_parroquial_chimborazo.png",
        "vista_macro_san_andres_total_v60_0_0.png",
        "comparativa_san_andres_sector_valle_aluvial_v60_0_0.png",
        "comparativa_parroquial_calpi_v60_0_0.png",
        "comparativa_parroquial_lican_v60_0_0.png",
        "comparativa_parroquial_columbe_v60_0_0.png",
        "mosaico_delimitacion_4_parroquias_v60_0_0.png"
    ]
    
    for t_dir in target_dirs:
        os.makedirs(t_dir, exist_ok=True)
        
    copied = 0
    for img_name in images_to_sync:
        found_path = None
        for s_dir in source_dirs:
            cand = os.path.join(s_dir, img_name)
            if os.path.exists(cand):
                found_path = cand
                break
        
        if found_path:
            for t_dir in target_dirs:
                dest = os.path.join(t_dir, img_name)
                shutil.copy2(found_path, dest)
                print(f"[OK] Copiado {img_name} -> {dest}")
            copied += 1
        else:
            print(f"[ALERTA] No se encontró {img_name} en fuentes.")
            
    print(f"\nTotal imágenes sincronizadas: {copied}/{len(images_to_sync)}")

if __name__ == "__main__":
    sync_visuals()
