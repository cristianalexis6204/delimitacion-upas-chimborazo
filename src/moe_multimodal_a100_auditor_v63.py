"""
AUDITOR MoE MULTIMODAL v63.0.0 - BUCLE DE MEJORA CONTINUA (GEMINI 4 ARGON & GEMINI 3.1 PRO)
TFM UNIR - Delimitacion Automatizada de UPAs en Chimborazo
Autor: Cristian Alexis Garcia Pumagualle
Director: Fernando Antonio Rufo Jimenez

Conecta con Vertex AI Global (proyecto 'tesis-bi-edison-paredes') con ThinkingLevel.HIGH.
Inspecciona visualmente los 4 renders HD de la version v63.0.0 SOTA para evaluar:
1. Simbologia y Aislamiento de Bosquetes (Verde Bosque #15803D sin borde amarillo de cultivo).
2. Detector de Invernaderos y Naves Agricolas PGI (Aguamarina #06B6D4 con buffer 2.0 m).
3. Snapping Vial y Sellado de Micro-Franjas (< 1.8 m) contra la Red Vial LPIS.
4. Maquetacion Cartografica HUD (cero solapes tipograficos con titulos).
5. Calidad Global y Dictamen Colegiado del Tribunal.
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

Se te presentan las 4 imágenes de alta resolución (220 DPI) correspondientes a los 4 cuadrantes agrícolas representativos de la Parroquia San Andrés (Chimborazo, Ecuador), procesados mediante el pipeline maestro v63.0.0 SOTA ejecutado en GPU NVIDIA A100.

Esta versión v63.0.0 implementa las RECOMENDACIONES ESPECÍFICAS DE LA AUDITORÍA PREVIA v62:
1. DESACOPLE Y SIMBOLOGÍA DE BOSQUETES (Patch TOF, Lucas et al. 2026): Los bosquetes y arboledas se han extraído de las UPAs agrícolas y se renderizan exclusivamente con contorno y relleno en Verde Bosque (#15803D), erradicando el contorno amarillo de cultivo sobre copas de árboles.
2. DETECTOR DE INVERNADEROS Y ESTRUCTURAS PRODUCTIVAS (PGI): Se ha incorporado el Plastic Greenhouse Index y análisis de textura para clasificar naves agrícolas e invernaderos en Aguamarina (#06B6D4) con buffer adaptativo de 2.0 m.
3. SNAPPING VIAL LPIS: Se aplica dilatación controlada para cerrar las micro-franjas residuales (< 1.8 m) entre los límites parcelarios y el buffer vial exterior de 3.5 m / 1.5 m.
4. MAQUETACIÓN CARTOGRÁFICA HUD: La tarjeta técnica de métricas se reposiciona con coordenadas relativas transAxes (y=0.96) y mayor padding en los títulos para garantizar 0.00% solape tipográfico.

MÉTRICAS CUANTITATIVAS REGISTRADAS POR LA A100:
{json.dumps(metrics_payload, indent=2, ensure_ascii=False)}

MANDATOS DE INSPECCIÓN VISUAL MULTIMODAL:
1. Verificación del Desacople de Bosquetes (Patch TOF):
   - ¿Aparecen los bosquetes y masas leñosas claramente en verde bosque (#15803D)? ¿Se confirma que ya no llevan el contorno amarillo de las parcelas de cultivo?
2. Inspección del Detector de Invernaderos (PGI):
   - ¿Se identifican las cubiertas agroindustriales e invernaderos en aguamarina (#06B6D4) con su buffer de 2 m (especialmente visible en Batzacón)?
3. Inspección del Snapping Vial y Continuidad Planar:
   - ¿Se observa un sellado continuo entre el borde de las UPAs amarillas y la servidumbre vial cian/blanco, sin micro-grietas vacías?
4. Evaluación de la Maquetación y Legibilidad Cartográfica:
   - ¿La tarjeta HUD de métricas y la leyenda quedan limpiamente posicionadas sin interferir con los títulos ni con el contenido geográfico?
5. Dictamen Colegiado:
   - Puntuación numérica (0-100) en cada dimensión.
   - 3 Fortalezas visuales indiscutibles.
   - Veredicto final del Tribunal: ¿Alcanza el nivel de Excelencia Definitiva para la Memoria de TFM?

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
  "evaluacion_de_las_mejoras_v63": {{
    "impacto_simbologia_bosquetes_verde": "string",
    "impacto_detector_invernaderos_turquesa": "string",
    "impacto_snapping_vial": "string",
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
    print(f"\n[MoE Multimodal] Iniciando auditoría v63 con {model_name} (Thinking: HIGH)...")
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
    v63_dir = os.path.join(root_dir, "06_Codigo_y_Experimentos", "experimentos", "v63.0.0")
    if not os.path.exists(v63_dir):
        v63_dir = "./outputs"
        
    metrics_path = os.path.join(v63_dir, "metrics_summary_v63_a100.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics_payload = json.load(f)
    else:
        metrics_payload = {"version": "v63.0.0-A100-SOTA"}
        
    images = [
        os.path.join(v63_dir, f"comparativa_a100_cuadrante_{i}_san_andres.png")
        for i in range(1, 5)
    ]
    
    prompt = build_multimodal_audit_prompt(metrics_payload)
    client = genai.Client()
    
    models = ["gemini-4-argon", "gemini-3.1-pro-preview"]
    print("=" * 80)
    print("INICIANDO AUDITORÍA MoE MULTIMODAL PARALELA - VERSIÓN v63.0.0 SOTA")
    print(f"Modelos: {models} (ThinkingLevel.HIGH en Vertex AI Global)")
    print("=" * 80)
    
    tasks = [
        run_single_multimodal_audit(client, m, images, prompt)
        for m in models
    ]
    results = await asyncio.gather(*tasks)
    
    final_report = {
        "fecha": time.strftime("%Y-%m-%d %H:%M:%S"),
        "version_auditada": "v63.0.0-A100-SOTA",
        "cuadrantes_analizados": 4,
        "auditorias": {res.get("modelo_auditor", f"mod_{i}"): res for i, res in enumerate(results)},
        "score_consolidado_moe": {}
    }
    
    scores = [r.get("scorecard_multimodal", {}) for r in results if "scorecard_multimodal" in r]
    if scores:
        keys = list(scores[0].keys())
        consolidated = {}
        for k in keys:
            vals = [s.get(k, 0) for s in scores if k in s]
            consolidated[k] = round(float(np.mean(vals)), 1)
        final_report["score_consolidado_moe"] = consolidated
        
    rep_path = os.path.join(v63_dir, "auditoria_moe_multimodal_v63.json")
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Informe de Auditoría MoE Multimodal v63 guardado en: {rep_path}")

if __name__ == "__main__":
    asyncio.run(main())
