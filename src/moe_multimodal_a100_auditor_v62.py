"""
AUDITOR MoE MULTIMODAL v62.0.0 - BUCLE DE MEJORA CONTINUA (GEMINI 4 ARGON & GEMINI 3.1 PRO)
TFM UNIR - Delimitacion Automatizada de UPAs en Chimborazo
Autor: Cristian Alexis Garcia Pumagualle
Director: Fernando Antonio Rufo Jimenez

Conecta con Vertex AI Global (proyecto 'tesis-bi-edison-paredes') con ThinkingLevel.HIGH.
Inspecciona visualmente los 4 renders HD de la version v62.0.0 SOTA para evaluar:
1. Restitucion de la Red Vial LPIS (Cian / Blanco).
2. Deteccion de Viviendas Campesinas LADM (Magenta con buffer 6m).
3. Segmentacion de Arboles y Cercas Vivas (Lucas et al. 2026 - TOF).
4. Delimitacion Pura de UPAs Agricolas (Amarillo wireframe: 0.00% solapes).
5. Scorecards y Recomendaciones Concretas para la siguiente iteracion (v63).
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

Se te presentan las 4 imágenes de alta resolución (220 DPI) correspondientes a los 4 cuadrantes agrícolas representativos de la Parroquia San Andrés (Chimborazo, Ecuador), procesados mediante el pipeline maestro v62.0.0 SOTA ejecutado en GPU NVIDIA A100.

Esta versión v62.0.0 implementa el PARADIGMA DE DESCOMPOSICIÓN DEL PAISAJE RURAL:
- VÍAS LPIS (Cian neón #00E5FF con línea discontinua blanca): Ingesta de 1,302 vías oficiales + esqueletización morfológica con buffers de 3.5 m y 1.5 m.
- VIVIENDAS CAMPESINAS LADM (Magenta #D946EF con borde blanco y relleno 65%): Detección espectral CIELAB (teja a*>133, zinc L>200) + Sombra 3D Suroeste (ΔL >= 12 DN) + Watershed + Buffer de patio de 6.0 m.
- ÁRBOLES Y CERCAS VIVAS (Lucas et al., 2026 - Science of Remote Sensing, arXiv:2510.25239): Linear TOF (cercas vivas lineales en verde bosque #15803D punteado que refuerzan los linderos de Meijering) y Patch TOF (bosquetes clasificados).
- UPAS AGRÍCOLAS PURAS (Amarillo #FACC15 wireframe): Delimitación sobre el espacio purificado remanente con SAM-HQ ViT-B y Mosaico Planar Argmax con 0.0000% solapes inter-parcelarios garantizado.

MÉTRICAS CUANTITATIVAS REGISTRADAS POR LA A100:
{json.dumps(metrics_payload, indent=2, ensure_ascii=False)}

MANDATOS DE INSPECCIÓN VISUAL MULTIMODAL:
1. Inspección de la Red Vial LPIS:
   - ¿Se aprecian claramente las vías carrozables y caminos de tierra en cian neón y blanco? ¿Garantizan que ninguna UPA invada el espacio vial?
2. Inspección de Viviendas y Asentamientos Campesinos:
   - ¿Están correctamente identificadas las casas y galpones rurales en magenta sin invadir las parcelas agrícolas? ¿El buffer de 6 m protege adecuadamente los patios?
3. Inspección de Árboles y Cercas Vivas (Lucas et al. 2026):
   - ¿Las cercas vivas lineales (verde bosque) actúan coherentemente como linderos divisorios entre parcelas agrícolas? ¿Se evita que los bosquetes y quebradas se conviertan en falsas parcelas de cultivo?
4. Delineación Pura de UPAs y Calidad Topológica LADM:
   - ¿Cómo se ajustan los polígonos amarillos al mosaico de cultivos? ¿Se confirma el 0.0000% de solapes inter-parcelarios?
5. Dictamen Colegiado y Recomendaciones para la Siguiente Iteración (v63):
   - Puntuación numérica (0-100) en cada dimensión.
   - 3 Fortalezas visuales indiscutibles.
   - 3 Oportunidades técnicas y operativas de mejora inmediata para nutrir el ciclo de optimización continua (v63).

Responde ÚNICA Y ESTRICTAMENTE con un objeto JSON con este formato exacto:
{{
  "modelo_auditor": "string",
  "scorecard_multimodal": {{
    "restitucion_red_vial_lpis": 0-100,
    "precision_viviendas_y_patios_ladm": 0-100,
    "manejo_arboles_y_cercas_vivas_lucas2026": 0-100,
    "calidad_delimitacion_upas_puras": 0-100,
    "regularidad_topologica_0_solapes": 0-100,
    "calidad_global_consolidada": 0-100
  }},
  "diagnostico_visual_por_cuadrante": {{
    "cuadrante_1_san_andres_centro": "string",
    "cuadrante_2_chocavi_laderas": "string",
    "cuadrante_3_pulucate_piedemonte": "string",
    "cuadrante_4_batzacon_interfase_vial": "string"
  }},
  "analisis_detallado_del_paradigma": {{
    "efectividad_red_vial_lpis": "string",
    "efectividad_viviendas_y_patios_6m": "string",
    "impacto_cercas_vivas_en_linderos": "string",
    "evaluacion_mosaico_planar_sin_solapes": "string"
  }},
  "fortalezas_visuales_identificadas": [
    "string", "string", "string"
  ],
  "recomendaciones_para_la_version_v63": [
    "string", "string", "string"
  ],
  "veredicto_tribunal": "Aprobado con Distinción / Sobresaliente / Requiere Ajustes"
}}
"""

async def run_single_multimodal_audit(client, model_name, images_paths, prompt):
    print(f"\n[MoE Multimodal] Iniciando auditoría con {model_name} (Thinking: HIGH)...")
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
    v62_dir = os.path.join(root_dir, "06_Codigo_y_Experimentos", "experimentos", "v62.0.0")
    if not os.path.exists(v62_dir):
        v62_dir = "./outputs"
        
    metrics_path = os.path.join(v62_dir, "metrics_summary_v62_a100.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics_payload = json.load(f)
    else:
        metrics_payload = {"version": "v62.0.0-A100-SOTA"}
        
    images = [
        os.path.join(v62_dir, f"comparativa_a100_cuadrante_{i}_san_andres.png")
        for i in range(1, 5)
    ]
    
    prompt = build_multimodal_audit_prompt(metrics_payload)
    client = genai.Client()
    
    models = ["gemini-4-argon", "gemini-3.1-pro-preview"]
    print("=" * 80)
    print("INICIANDO AUDITORÍA MoE MULTIMODAL PARALELA - VERSIÓN v62.0.0 SOTA")
    print(f"Modelos: {models} (ThinkingLevel.HIGH en Vertex AI Global)")
    print("=" * 80)
    
    tasks = [
        run_single_multimodal_audit(client, m, images, prompt)
        for m in models
    ]
    results = await asyncio.gather(*tasks)
    
    final_report = {
        "fecha": time.strftime("%Y-%m-%d %H:%M:%S"),
        "version_auditada": "v62.0.0-A100-SOTA",
        "cuadrantes_analizados": 4,
        "auditorias": {res.get("modelo_auditor", f"mod_{i}"): res for i, res in enumerate(results)},
        "score_consolidado_moe": {}
    }
    
    # Calcular promedios
    scores = [r.get("scorecard_multimodal", {}) for r in results if "scorecard_multimodal" in r]
    if scores:
        keys = list(scores[0].keys())
        consolidated = {}
        for k in keys:
            vals = [s.get(k, 0) for s in scores if k in s]
            consolidated[k] = round(float(np.mean(vals)), 1)
        final_report["score_consolidado_moe"] = consolidated
        
    rep_path = os.path.join(v62_dir, "auditoria_moe_multimodal_v62.json")
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Informe de Auditoría MoE Multimodal v62 guardado en: {rep_path}")

if __name__ == "__main__":
    asyncio.run(main())
