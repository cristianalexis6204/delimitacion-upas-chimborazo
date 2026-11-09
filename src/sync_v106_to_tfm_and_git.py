"""
================================================================================
SCRIPT DE SINCRONIZACIÓN ATÓMICA DE FIGURAS v106.0.0
================================================================================
Copia las 22 figuras canónicas desde:
  experimentos/v106.0.0_muestra_completa_e2e/figuras_tfm/
Hacia:
  1. 03_Fase_2_Elaboracion/latex/figuras/
  2. delimitacion-upas-chimborazo/outputs/
  3. 06_Codigo_y_Experimentos/reportes/figuras/
================================================================================
"""

import os
import shutil
import pathlib

SRC_DIR = pathlib.Path(__file__).resolve().parents[1] / "figuras_tfm"

TARGET_DIRS = [
    pathlib.Path(r"K:\Mi unidad\CURSOS Y CAPACITACIÓN PERSONAL\MASTER-INTELIGENCIA ARTIFICIAL\Aula Master IA\02_Segundo_Cuatrimestre\09_TFM\03_Fase_2_Elaboracion\latex\figuras"),
    pathlib.Path(r"K:\Mi unidad\CURSOS Y CAPACITACIÓN PERSONAL\MASTER-INTELIGENCIA ARTIFICIAL\Aula Master IA\02_Segundo_Cuatrimestre\09_TFM\delimitacion-upas-chimborazo\outputs"),
    pathlib.Path(r"K:\Mi unidad\CURSOS Y CAPACITACIÓN PERSONAL\MASTER-INTELIGENCIA ARTIFICIAL\Aula Master IA\02_Segundo_Cuatrimestre\09_TFM\06_Codigo_y_Experimentos\reportes\figuras")
]

def sync_figures():
    files = list(SRC_DIR.glob("*.png"))
    print(f"Sincronizando {len(files)} figuras canónicas desde {SRC_DIR}...")
    
    for target in TARGET_DIRS:
        target.mkdir(parents=True, exist_ok=True)
        print(f"\n[DESTINO] -> {target}")
        for f in files:
            dest_file = target / f.name
            shutil.copy2(f, dest_file)
            print(f"  + Copiado: {f.name} ({f.stat().st_size / 1024:.1f} KB)")
            
    # Aliases de compatibilidad hacia atrás para compilación sin fisuras
    latex_fig_dir = TARGET_DIRS[0]
    compat_map = {
        "super_resolucion.png": "super_resolucion.jpg",
        "ground_truth_multicanton_v106.png": "ground_truth_calpi.jpg",
        "atlas_ensamble_maestro_chimborazo.png": "atlas_ensamble_maestro_san_andres.jpg",
        "benchmark_comparativo_7_modelos_v106.png": "benchmark_comparativo_7_modelos_v58_0_0.png",
        "fase1_cielab_san_andres.png": "fase1_cielab.jpg",
        "fase2_restricciones_san_andres.png": "fase2_vias.jpg",
        "fase4_sam_hq_san_andres.png": "fase3_sam.jpg",
        "fase6_ladm_halfedge_san_andres.png": "fase4_ladm.jpg"
    }
    print("\nGenerando alias de compatibilidad en latex/figuras/...")
    for orig, alias in compat_map.items():
        src_path = latex_fig_dir / orig
        if src_path.exists():
            alias_path = latex_fig_dir / alias
            shutil.copy2(src_path, alias_path)
            print(f"  * Alias creado: {alias} <- {orig}")

    print("\n[ÉXITO] Sincronización atómica completada en todos los destinos.")

if __name__ == "__main__":
    sync_figures()
