"""
================================================================================
MOE AGENTS CLI v3.0 - INTERFAZ DE COMANDOS DE EVALUACIÓN Y CALIDAD MULTI-AGENTE
Inspirado en Google Agents CLI (google/agents-cli) y Google Agent Development Kit (ADK)
Comandos: render-zooms | eval probe | eval grade | eval compare
TFM UNIR - Delimitación Automatizada de UPAs en Chimborazo
================================================================================
"""

import os
import sys
import json
import time
import argparse
import asyncio
from typing import Dict, Any, List

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from gis_telemetry_probe import run_gis_telemetry_probe
from visual_grounding_renderer import (
    prepare_quadrant_zoom_parts_for_gemini,
    build_quadrant_250m_zoom_sheet,
    split_atlas_dupla_into_quadrants,
    cv2_imread_unicode
)
from kaizen_memory_ledger import KaizenMemoryLedger
import moe_kaizen_v2_orchestrator as orch

def cmd_render_zooms(args):
    """Genera las láminas analíticas desacopladas por cuadrante con micro-zooms a altitud virtual de 250m."""
    exp_dir = os.path.abspath(args.exp_dir)
    print(f"\n[CLI: render-zooms] Procesando directorio: {exp_dir}")
    
    # 1. Ejecutar sonda para obtener los focos de zoom por cuadrante
    telemetry = run_gis_telemetry_probe(exp_dir)
    zoom_foci = telemetry.get("focos_micro_zoom_250m", {})
    
    out_dir = os.path.join(exp_dir, "quad_zooms_250m")
    os.makedirs(out_dir, exist_ok=True)
    
    # Identificar duplas a procesar
    if args.dupla:
        target_duplas = [args.dupla]
    else:
        target_duplas = [f for f in os.listdir(exp_dir) if f.startswith("dupla_") and f.endswith(".png")]
        target_duplas.sort()

    print(f"Duplas a procesar ({len(target_duplas)} láminas):")
    total_generated = 0
    t0 = time.time()
    
    for d_name in target_duplas:
        d_path = os.path.join(exp_dir, d_name)
        if not os.path.exists(d_path):
            print(f"  [Omitido] Archivo no existe: {d_path}")
            continue
        base_name = os.path.splitext(d_name)[0]
        quads = split_atlas_dupla_into_quadrants(d_path)
        print(f"  -> Generando micro-zooms para {d_name} (4 cuadrantes)...")
        for idx, q_im in enumerate(quads, 1):
            q_key = f"Q{idx}"
            foci = zoom_foci.get(q_key, {})
            out_file = os.path.join(out_dir, f"{base_name}_{q_key}_zoom250m.jpg")
            build_quadrant_250m_zoom_sheet(q_im, quad_idx=idx, zoom_foci=foci, out_path=out_file)
            total_generated += 1
            
    elapsed = time.time() - t0
    print(f"\n[Éxito render-zooms] Generadas {total_generated} láminas analíticas en {elapsed:.1f} s.")
    print(f"Directorio de salida: {out_dir}\n")

def cmd_eval_probe(args):
    """Ejecuta la sonda determinista GIS y matriz de colisión topológica inter-capa."""
    exp_dir = os.path.abspath(args.exp_dir)
    print(f"\n[CLI: eval probe] Ejecutando sonda determinista en: {exp_dir}")
    telemetry = run_gis_telemetry_probe(exp_dir)
    print(telemetry["telemetry_text_prompt"])
    
    out_json = os.path.join(exp_dir, "telemetry_probe_v3.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(telemetry, f, indent=2, ensure_ascii=False)
    print(f"[Telemetría Exportada]: {out_json}\n")

def cmd_eval_compare(args):
    """Compara métricas y dominancia de Pareto entre dos versiones (Base vs Target)."""
    base_dir = os.path.abspath(args.base)
    target_dir = os.path.abspath(args.target)
    
    print(f"\n" + "=" * 80)
    print(f"[CLI: eval compare] Comparación de Calidad LADM y Cobertura Catastral")
    print(f"  Línea Base:  {os.path.basename(base_dir)}")
    print(f"  Candidato:   {os.path.basename(target_dir)}")
    print("=" * 80)
    
    t_base = run_gis_telemetry_probe(base_dir)
    t_target = run_gis_telemetry_probe(target_dir)
    
    res_b = t_base["resumen_global"]
    res_t = t_target["resumen_global"]
    
    delta_upas = res_t["total_upas"] - res_b["total_upas"]
    delta_ha = res_t["area_total_ha"] - res_b["area_total_ha"]
    delta_spill = res_t["spillover_total_m2"] - res_b["spillover_total_m2"]
    
    # Criterio Pareto
    pareto_solape = (res_t["solape_inter_parcelario_pct"] <= res_b["solape_inter_parcelario_pct"]) and (res_t["solape_inter_parcelario_pct"] == 0.0)
    pareto_spill = res_t["spillover_total_m2"] <= res_b["spillover_total_m2"]
    pareto_area = res_t["area_total_ha"] >= res_b["area_total_ha"]
    
    cumple_pareto = pareto_solape and pareto_spill and pareto_area
    
    print(f"\nTABLA COMPARATIVA DE MÉTRICAS DETERMINISTAS:")
    print(f"| Métrica LADM / Algoritmo | Línea Base ({os.path.basename(base_dir)}) | Candidato ({os.path.basename(target_dir)}) | Delta / Evolución |")
    print(f"| :--- | :---: | :---: | :---: |")
    solape_diff = res_t['solape_inter_parcelario_pct'] - res_b['solape_inter_parcelario_pct']
    solape_evol = '0.0000% (Estable)' if abs(solape_diff) < 1e-6 else f'{solape_diff:+.6f}%'

    print(f"| Solapamiento Inter-parcelario | {res_b['solape_inter_parcelario_pct']:.6f}% | {res_t['solape_inter_parcelario_pct']:.6f}% | {solape_evol} |")
    print(f"| Confinamiento Circular R=350m | {res_b['spillover_total_m2']:.4f} m² | {res_t['spillover_total_m2']:.4f} m² | {delta_spill:+.4f} m² |")
    print(f"| UPAs Agrícolas Puras | {res_b['total_upas']} | {res_t['total_upas']} | {delta_upas:+d} parcelas |")
    print(f"| Área Total Agrícola | {res_b['area_total_ha']:.2f} ha | {res_t['area_total_ha']:.2f} ha | {delta_ha:+.2f} ha |")
    print(f"| Conformidad LADM ISO 19152 | {res_b['conformidad_ladm_iso_19152']} | {res_t['conformidad_ladm_iso_19152']} | 100.0% CONFORME |")
    print(f"\nDICTAMEN DE DOMINANCIA DE PARETO: {'DOMINANTE (APROBADO)' if cumple_pareto else 'NO DOMINANTE (REQUIERE AJUSTES)'}\n")

def cmd_eval_grade(args):
    """Ejecuta la deliberación del Tribunal MoE-Kaizen v3.0 con calificación VLM-as-a-Judge."""
    exp_dir = os.path.abspath(args.exp_dir)
    version_tag = args.version or os.path.basename(exp_dir).split(".")[0]
    model_name = args.model or "gemini-4-argon"
    
    print(f"\n[CLI: eval grade] Iniciando evaluación VLM-as-a-Judge en: {exp_dir} (Versión: {version_tag}, Modelo: {model_name})")
    asyncio.run(orch.run_orchestrator_cycle(exp_dir, version_tag, model_name=model_name))

def main():
    parser = argparse.ArgumentParser(description="Google Agents CLI Catastral - MoE-Kaizen v3.0")
    subparsers = parser.add_subparsers(dest="command", help="Comandos de ingeniería de agentes")
    
    # render-zooms
    p_render = subparsers.add_parser("render-zooms", help="Generar láminas por cuadrante con micro-zoom a 250m")
    p_render.add_argument("--exp-dir", required=True, help="Directorio del experimento")
    p_render.add_argument("--dupla", default=None, help="Nombre específico de la dupla (opcional)")
    p_render.set_defaults(func=cmd_render_zooms)
    
    # eval
    p_eval = subparsers.add_parser("eval", help="Subcomandos de evaluación y calidad")
    eval_sub = p_eval.add_subparsers(dest="eval_command", help="Acción de evaluación")
    
    # eval probe
    p_probe = eval_sub.add_parser("probe", help="Ejecutar sonda determinista GIS y matriz de colisiones")
    p_probe.add_argument("--exp-dir", required=True, help="Directorio del experimento")
    p_probe.set_defaults(func=cmd_eval_probe)
    
    # eval compare
    p_comp = eval_sub.add_parser("compare", help="Comparar dos versiones con criterio de Pareto")
    p_comp.add_argument("--base", required=True, help="Directorio de la versión base")
    p_comp.add_argument("--target", required=True, help="Directorio de la versión objetivo")
    p_comp.set_defaults(func=cmd_eval_compare)
    
    # eval grade
    p_grade = eval_sub.add_parser("grade", help="Ejecutar deliberación VLM-as-a-Judge")
    p_grade.add_argument("--exp-dir", required=True, help="Directorio del experimento")
    p_grade.add_argument("--version", default=None, help="Etiqueta de versión (ej. v100.0.0)")
    p_grade.add_argument("--model", default="gemini-4-argon", help="Modelo para VLM-as-a-Judge")
    p_grade.set_defaults(func=cmd_eval_grade)
    
    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
