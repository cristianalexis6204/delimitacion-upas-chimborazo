"""
Auditor MoE Multimodal en Paralelo: Inspeccion Visual de Renders A100 (Gemini 4 Argon vs. Gemini 3.1 Pro)
TFM UNIR - Delimitacion Automatizada de UPAs en Chimborazo
Autor: Cristian Alexis Garcia Pumagualle
Director: Fernando Antonio Rufo Jimenez

Conecta con Vertex AI en region global bajo el proyecto 'tesis-bi-edison-paredes' (Edison).
Envia las imagenes PNG de alta definicion (lado a lado: Ortofoto VHR vs Delimitacion A100 SOTA)
a los dos modelos de razonamiento maximo (ThinkingLevel.HIGH) para su inspeccion visual directa.
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
Actúa como un Comité Colegiado de Mixture of Experts (MoE) en Visión por Computador, Sensores Remotos y Catastro LADM ISO 19152 de Máximo Nivel Técnico.

Se te presentan las imágenes de alta resolución correspondientes a los 4 cuadrantes agrícolas representativos de la Parroquia San Andrés (Chimborazo, Ecuador), procesados mediante el pipeline de inferencia masiva sobre una GPU NVIDIA A100-SXM4-80GB (Google Colab Pro).

Cada imagen muestra una comparativa lado a lado:
- PANEL IZQUIERDO: Ortofoto satelital VHR original (resolución 0.30 m/píxel, Zoom 18 de Google Satellite).
- PANEL DERECHO: Delimitación vectorial automática de UPAs (polígonos verdes translúcidos con bordes regulares 'shapely.coverage_simplify' y Meijering bimodal) y Edificaciones Campesinas (polígonos rojos con buffer de protección de 20 metros).

MÉTRICAS CUANTITATIVAS REGISTRADAS POR LA A100:
{json.dumps(metrics_payload, indent=2, ensure_ascii=False)}

MANDATOS DE INSPECCIÓN VISUAL MULTIMODAL:
1. Inspección Visual de Linderos y Delineación:
   - Observa visualmente el panel izquierdo vs. derecho. ¿Cómo se ajustan los polígonos verdes a los límites visuales reales (setos vivos de cabuya/lechero, zanjas de desagüe, caminos de tierra, cambios abruptos de textura de cultivo)?
   - ¿Detectas sobre-segmentación (división artificial de un mismo lote homogéneo) o sub-segmentación (fusión errónea de parcelas vecinas)?
2. Inspección Visual de Edificaciones y Viviendas Campesinas:
   - Examina los polígonos rojos. ¿Están correctamente identificadas las viviendas de teja de barro o zinc? ¿Invaden en algún punto el área de cultivo verde?
3. Evaluación de Regularidad Geométrica LADM:
   - Los polígonos ahora tienen una media de 11.97 vértices (frente a los 57 vértices hipercomplejos previos). ¿La simplificación planar mantiene la silueta agronómica o deforma excesivamente las esquinas de los terrenos?
4. Rendimiento Hardware A100 (50 segundos para 203 UPAs en 80 GB VRAM):
   - Evalúa el balance costo-beneficio de inferencia a escala provincial en la infraestructura Cloud de Google Colab.
5. Dictamen y Oportunidades de Mejora:
   - Puntuación en escala 0-100 para cada dimensión.
   - Identifica 3 fortalezas visuales indiscutibles y 3 oportunidades técnicas de mejora inmediata.

Responde ÚNICA Y ESTRICTAMENTE con un objeto JSON con este formato exacto:
{{
  "modelo_auditor": "string",
  "scorecard_multimodal": {{
    "fidelidad_visual_linderos": 0-100,
    "precision_exclusion_viviendas": 0-100,
    "regularidad_topologica_ladm": 0-100,
    "eficiencia_hpc_a100": 0-100,
    "calidad_global_consolidada": 0-100
  }},
  "diagnostico_visual_por_cuadrante": {{
    "cuadrante_1_valle_aluvial": "string",
    "cuadrante_2_laderas_andinas": "string",
    "cuadrante_3_piedemonte_chimborazo": "string",
    "cuadrante_4_interfase_vial": "string"
  }},
  "analisis_detallado": {{
    "fidelidad_linderos_vs_ortofoto": "string",
    "comportamiento_en_laderas_y_terrazas": "string",
    "efectividad_buffer_viviendas": "string",
    "evaluacion_regularizacion_ladm_11_vertices": "string"
  }},
  "fortalezas_visuales_identificadas": [
    "string", "string", "string"
  ],
  "oportunidades_de_mejora_visual_y_algoritmica": [
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
    # Adjuntar las imágenes como partes multimodales optimizadas
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
                    scale = max_dim / float(max(h, w))
                    im = cv2.resize(im, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                _, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 90])
                img_bytes = buf.tobytes()
                parts.append(types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"))
                print(f"    - Adjuntada imagen multimodal: {os.path.basename(img_p)} ({len(img_bytes)/1024:.1f} KB, res: {im.shape[1]}x{im.shape[0]})")
            else:
                parts.append(types.Part.from_bytes(data=raw_b, mime_type="image/png"))
                print(f"    - Adjuntada imagen original: {os.path.basename(img_p)} ({len(raw_b)/1024:.1f} KB)")
    
    parts.append(prompt)

    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.HIGH),
        temperature=0.2,
        response_mime_type="application/json"
    )

    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: client.models.generate_content(
            model=model_name,
            contents=parts,
            config=config
        )
    )

    elapsed = time.time() - t0
    txt = response.text.strip()
    if txt.startswith("```json"):
        txt = txt[7:]
    if txt.endswith("```"):
        txt = txt[:-3]
    txt = txt.strip()

    try:
        dictamen = json.loads(txt)
        if isinstance(dictamen, list):
            dictamen = dictamen[0]
    except Exception as e:
        dictamen = {"raw_response": txt, "parse_error": str(e)}

    thinking_tokens = getattr(response.usage_metadata, "thinking_token_count", 0) if response.usage_metadata else 0
    total_tokens = getattr(response.usage_metadata, "total_token_count", 0) if response.usage_metadata else 0

    print(f"[MoE Multimodal] {model_name} completado en {elapsed:.1f}s | Thinking Tokens: {thinking_tokens:,} | Total: {total_tokens:,}")
    return {
        "modelo": model_name,
        "tiempo_s": round(elapsed, 2),
        "thinking_tokens": thinking_tokens,
        "total_tokens": total_tokens,
        "dictamen": dictamen
    }

async def main():
    print("=" * 80)
    print("AUDITORÍA MoE MULTIMODAL DUAL EN PARALELO - RENDERS A100")
    print("Modelos: Gemini 4 Argon vs. Gemini 3.1 Pro (Vertex AI Global - Thinking HIGH)")
    print("=" * 80)

    repo_root = r"K:\Mi unidad\CURSOS Y CAPACITACIÓN PERSONAL\MASTER-INTELIGENCIA ARTIFICIAL\Aula Master IA\02_Segundo_Cuatrimestre\09_TFM\delimitacion-upas-chimborazo"
    outputs_dir = os.path.join(repo_root, "outputs")
    reportes_dir = os.path.join(repo_root, "reportes")
    os.makedirs(reportes_dir, exist_ok=True)

    metrics_json = os.path.join(outputs_dir, "metrics_summary_v61_a100.json")
    with open(metrics_json, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)

    image_paths = [
        os.path.join(outputs_dir, f"comparativa_a100_cuadrante_{cid}_san_andres.png")
        for cid in range(1, 5)
    ]

    prompt = build_multimodal_audit_prompt(metrics_data)
    client = genai.Client()

    # Ejecución paralela de ambos modelos insignia
    task_gemini4 = run_single_multimodal_audit(client, "gemini-4-argon", image_paths, prompt)
    task_gemini31 = run_single_multimodal_audit(client, "gemini-3.1-pro-preview", image_paths, prompt)

    res_gemini4, res_gemini31 = await asyncio.gather(task_gemini4, task_gemini31)

    audit_consolidated = {
        "fecha": str(datetime.date.today()) if "datetime" in globals() else "2026-10-27",
        "tipo_auditoria": "Multimodal Visual MoE (4 Renders VHR A100)",
        "acelerador": "NVIDIA A100-SXM4-80GB",
        "cuadrantes_evaluados": 4,
        "modelos": {
            "gemini_4_argon": res_gemini4,
            "gemini_3_1_pro": res_gemini31
        }
    }

    out_json = os.path.join(reportes_dir, "auditoria_moe_multimodal_a100.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(audit_consolidated, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Dictamen consolidado MoE Multimodal guardado en: {out_json}")

if __name__ == "__main__":
    import datetime
    asyncio.run(main())
