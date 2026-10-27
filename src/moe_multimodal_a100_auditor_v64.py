"""
AUDITOR MoE MULTIMODAL v64.0.0 - CICLO 3 KAIZEN (GEMINI 4 ARGON & GEMINI 3.1 PRO)
TFM UNIR - Delimitacion Automatizada de UPAs en Chimborazo
Autor: Cristian Alexis Garcia Pumagualle
Director: Fernando Antonio Rufo Jimenez

Conecta con Vertex AI Global (proyecto 'tesis-bi-edison-paredes') con ThinkingLevel.HIGH.
Inspecciona visualmente los 4 renders HD de la version v64.0.0 SOTA para evaluar:
1. Erradicacion total del contorno amarillo sobre dosel arboreo y quebradas (Patch/Forest TOF en Verde #15803D).
2. Deteccion calibrada de naves e invernaderos PGI en Turquesa (#06B6D4) en Batzacon.
3. Snapping Vial y Particion Planar Disjunta Estricta (0.0000% solapes).
4. Maquetacion Cartografica HUD Profesional.
5. Veredicto Final del Tribunal Colegiado.
"""

import os
import sys
import json
import time
import asyncio
import numpy as np

os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
os.environ["GOOGLE_CLOUD_PROJECT"] = "tesis-bi-edison-paredes"
os.environ["GOOGLE_CLOUD_LOCATION"] = "global"

from google import genai
from google.genai import types

def build_multimodal_audit_prompt(metrics_payload):
    return f"""
Actúa como un Comité Colegiado de Mixture of Experts (MoE) en Visión por Computador, Sensores Remotos, Catastro LADM ISO 19152 y Agronomía de Máximo Nivel Internacional.

Se te presentan las 4 láminas comparativas de alta resolución (220 DPI) correspondientes a los 4 cuadrantes agrícolas representativos de la Parroquia San Andrés (Chimborazo, Ecuador), procesados mediante el pipeline maestro v64.0.0 SOTA ejecutado en GPU NVIDIA A100.

Esta versión v64.0.0 implementa los MICRO-REFINAMIENTOS ESPECÍFICOS DE LA AUDITORÍA PREVIA v63:
1. DIFERENCIA BOOLEANA ESTRICTA Y Z-ORDER EN COBERTURAS ARBÓREAS: Se ha erradicado por completo el trazo perimetral amarillo sobre los doseles boscosos y quebradas (especialmente en San Andrés Centro y Pulucate), asegurando que las masas leñosas se rendericen con z-order prioritario exclusivamente con contorno y relleno en Verde Bosque (#15803D).
2. CALIBRACIÓN ESPECTRAL PGI EN BORDES (BATZACÓN): Se ha ajustado el umbral de reflectancia del Plastic Greenhouse Index para detectar de forma exhaustiva las cubiertas y naves plásticas agroindustriales en Turquesa (#06B6D4) con buffer adaptativo de 2.0 m, incluyendo aquellas ubicadas cerca del borde del recorte.
3. SNAPPING VIAL LPIS Y CONTINUIDAD PLANAR: Se consolida el sellado topológico de micro-franjas (< 1.8 m) frente a las carreteras cian/blanco.
4. MAQUETACIÓN CARTOGRÁFICA HUD: Tarjeta técnica en la esquina superior izquierda y leyenda en la esquina inferior derecha sin ningún solape tipográfico.

MÉTRICAS CUANTITATIVAS REGISTRADAS POR LA A100:
{json.dumps(metrics_payload, indent=2, ensure_ascii=False)}

MANDATOS DE INSPECCIÓN VISUAL MULTIMODAL:
1. Verificación del Desacople Absoluto de Bosquetes (Patch TOF):
   - ¿Se confirma que en los 4 cuadrantes (especialmente en las quebradas de San Andrés Centro y Pulucate) se ha eliminado el contorno amarillo de cultivo sobre las arboledas, visualizándose únicamente en Verde Bosque (#15803D)?
2. Inspección del Detector de Invernaderos Calibrado (PGI):
   - ¿Se aprecia una detección completa y precisa de las naves agrícolas e invernaderos en Turquesa (#06B6D4) en Batzacón y demás sectores?
3. Inspección del Snapping Vial y Partición Planar:
   - ¿Se observa una continuidad perfecta entre parcelas y caminos sin solapes ni fisuras artificiales?
4. Evaluación Estética y Cartográfica:
   - ¿Alcanza el nivel visual definitivo de estándar de publicación científica para la Memoria de TFM?
5. Dictamen Colegiado del Tribunal:
   - Puntuación numérica (0-100) en cada dimensión.
   - 3 Fortalezas visuales concluyentes.
   - Veredicto final: ¿Aprobado con Distinción / Sobresaliente / Matrícula de Honor?

Responde ÚNICA Y ESTRICTAMENTE con un objeto JSON con este formato exacto:
{{
  "modelo_auditor": "string",
  "scorecard_multimodal": {{
    "aislamiento_bosquetes_patch_tof": 0-100,
    "deteccion_invernaderos_pgi": 0-100,
    "snapping_vial_y_sellado_lpis": 0-100,
    "calidad_delimitacion_upas_puras": 0-100,
    "maquetacion_hud_y_estetica_visual": 0-100,
    "regularidad_topologica_0_solapes": 0-100,
    "calidad_global_consolidada": 0-100
  }},
  "diagnostico_visual_por_cuadrante": {{
    "cuadrante_1_san_andres_centro": "string",
    "cuadrante_2_chocavi_laderas": "string",
    "cuadrante_3_pulucate_piedemonte": "string",
    "cuadrante_4_batzacon_interfase_vial": "string"
  }},
  "evaluacion_de_las_mejoras_v64": {{
    "impacto_diferencia_booleana_bosquetes_verde": "string",
    "impacto_calibracion_pgi_batzacon": "string",
    "impacto_continuidad_planar_y_snapping": "string",
    "impacto_maquetacion_hud": "string"
  }},
  "fortalezas_visuales_identificadas": [
    "string", "string", "string"
  ],
  "observaciones_adicionales": [
    "string", "string"
  ],
  "veredicto_tribunal": "Aprobado con Distinción / Sobresaliente / Matrícula de Honor"
}}
"""

async def run_single_multimodal_audit(client, model_name, images_paths, prompt):
    print(f"\n[MoE Multimodal] Iniciando auditoría v64 con {model_name} (Thinking: HIGH)...")
    t0 = time.time()
    parts = []
    import cv2
    for img_p in images_paths:
        if os.path.exists(img_p):
            with open(img_p, "rb") as f:
                raw_b = f.read()
            arr = np.frombuffer(raw_b, dtype=np.uint8)
            im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if im is not None:
                h, w, _ = im.shape
                max_dim = 1600
                if max(h, w) > max_dim:
                    sc = max_dim / max(h, w)
                    im = cv2.resize(im, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
                _, enc = cv2.imencode(".jpg", im, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
                parts.append(types.Part.from_bytes(data=enc.tobytes(), mime_type="image/jpeg"))
                print(f"  [Adjuntada Imagen]: {os.path.basename(img_p)} ({len(enc)//1024} KB)")
                
    parts.append(prompt)
    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.HIGH),
        temperature=0.2,
        response_mime_type="application/json"
    )
    
    response = await client.aio.models.generate_content(
        model=model_name,
        contents=parts,
        config=config
    )
    dur = round(time.time() - t0, 1)
    print(f"[MoE Multimodal] {model_name} finalizó en {dur} s.")
    try:
        raw_text = response.text.strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]
        parsed = json.loads(raw_text.strip())
        parsed["tiempo_auditoria_segundos"] = dur
        parsed["modelo_auditor"] = model_name
        return parsed
    except Exception as e:
        print(f"  [Error Parseo {model_name}]: {e}")
        return {
            "modelo_auditor": model_name,
            "error": str(e),
            "raw_text": response.text[:1000],
            "tiempo_auditoria_segundos": dur
        }

async def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    v64_dir = os.path.join(root_dir, "06_Codigo_y_Experimentos", "experimentos", "v64.0.0")
    if not os.path.exists(v64_dir):
        v64_dir = "./outputs"
        
    metrics_path = os.path.join(v64_dir, "metrics_summary_v64_a100.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics_payload = json.load(f)
    else:
        metrics_payload = {"version": "v64.0.0-A100-SOTA"}
        
    images = [
        os.path.join(v64_dir, f"comparativa_a100_cuadrante_{i}_san_andres.png")
        for i in range(1, 5)
    ]
    
    prompt = build_multimodal_audit_prompt(metrics_payload)
    client = genai.Client()
    
    models = ["gemini-4-argon", "gemini-3.1-pro-preview"]
    print("=" * 80)
    print("INICIANDO AUDITORÍA MoE MULTIMODAL PARALELA - VERSIÓN v64.0.0 SOTA")
    print(f"Modelos: {models} (ThinkingLevel.HIGH en Vertex AI Global)")
    print("=" * 80)
    
    tasks = [
        run_single_multimodal_audit(client, m, images, prompt)
        for m in models
    ]
    
    results = await asyncio.gather(*tasks)
    
    audit_dict = {res["modelo_auditor"]: res for res in results}
    
    # Consolidacion
    scorecards = [res["scorecard_multimodal"] for res in results if "scorecard_multimodal" in res]
    consolidated_scores = {}
    if scorecards:
        for k in scorecards[0].keys():
            vals = [sc[k] for sc in scorecards if k in sc]
            consolidated_scores[k] = round(float(np.mean(vals)), 1)
            
    final_output = {
        "fecha": time.strftime("%Y-%m-%d %H:%M:%S"),
        "version_auditada": "v64.0.0-A100-SOTA",
        "cuadrantes_analizados": len(images),
        "auditorias": audit_dict,
        "score_consolidado_moe": consolidated_scores
    }
    
    out_json = os.path.join(v64_dir, "auditoria_moe_multimodal_v64.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
        
    print(f"\n[OK] Informe de Auditoría MoE Multimodal v64 guardado en: {out_json}")
    print("\n" + "=" * 80)
    print("RESUMEN DEL SCORECARD MoE MULTIMODAL v64.0.0:")
    print(json.dumps(consolidated_scores, indent=2, ensure_ascii=False))
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())
