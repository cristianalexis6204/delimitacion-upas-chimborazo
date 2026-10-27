"""
Auditor MoE Multimodal en Paralelo: Inspeccion Visual de Renders A100 (Gemini 4 Argon vs. Gemini 3.1 Pro)
Version v65.0.0-A100-SOTA (Ciclo 4 Kaizen MoE)
TFM UNIR - Delimitacion Automatizada de UPAs en Chimborazo
Autor: Cristian Alexis Garcia Pumagualle
Director: Fernando Antonio Rufo Jimenez

Conecta con Vertex AI en region global bajo el proyecto 'tesis-bi-edison-paredes'.
Envia las imagenes PNG de alta definicion (lado a lado: Panel A Ortofoto VHR vs Panel B Delimitacion SOTA)
a los dos modelos de razonamiento maximo (ThinkingLevel.HIGH) para su inspeccion visual directa.
"""

import os
import sys
import json
import time
import asyncio
import datetime
import numpy as np
import cv2

os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
os.environ["GOOGLE_CLOUD_PROJECT"] = "tesis-bi-edison-paredes"
os.environ["GOOGLE_CLOUD_LOCATION"] = "global"

from google import genai
from google.genai import types

def build_multimodal_audit_prompt_v65(metrics_payload):
    return f"""
Actúa como un Comité Colegiado de Mixture of Experts (MoE) en Visión por Computador, Sensores Remotos y Catastro LADM ISO 19152 de Máximo Nivel Técnico.

Se te presentan las imágenes de alta resolución correspondientes a los 4 cuadrantes agrícolas representativos de la Parroquia San Andrés (Chimborazo, Ecuador), procesados mediante el pipeline de inferencia masiva sobre una GPU NVIDIA A100-SXM4-80GB (Google Colab Pro).

Cada imagen muestra una comparativa directa lado a lado:
- PANEL A (Izquierda): Ortofoto satelital VHR real sin procesar (resolución 0.30 m/píxel, Zoom 18 de Google Satellite).
- PANEL B (Derecha): Delimitación catastral multicapa automatizada v65.0.0-A100-SOTA:
  * UPAs Agrícolas Puras: Wireframe Amarillo Oro (#FACC15) con relleno fenológico translúcido.
  * Cercas Vivas / Linderos Arbóreos (Linear TOF): Línea continua Verde Esmeralda Neón (#22C55E).
  * Bosquetes y Masas Leñosas (Patch TOF): Polígonos Verde Bosque (#15803D).
  * Viviendas Campesinas Reales (LADM): Polígonos Magenta (#D946EF) con sombra 3D direccional SW.
  * Galpones y Tinglados Agrícolas: Polígonos Gris Pizarra Metálico (#64748B) con borde blanco.
  * Patios Campesinos y Eras: Polígonos Ocre Arena (#F59E0B) translúcidos.
  * Invernaderos Agrícolas (PGI): Polígonos Turquesa Aguamarina (#06B6D4) con buffer 2.0 m.
  * Red Vial LPIS Oficial: Trazado Cian Neón (#00E5FF) con línea discontinua blanca.

MÉTRICAS CUANTITATIVAS REGISTRADAS POR LA GPU A100 (v65):
{json.dumps(metrics_payload, indent=2, ensure_ascii=False)}

MANDATOS DE INSPECCIÓN VISUAL MULTIMODAL COMPARATIVA (PANEL A vs PANEL B):

1. COMPLETITUD CATASTRAL Y COBERTURA PLANAR AGRÍCOLA (RECALL DE UPAS):
   - Compara exhaustivamente el suelo cultivable del Panel A con los polígonos amarillos del Panel B.
   - En la versión previa (v63), se detectó una subsegmentación severa donde el 40-50% del suelo útil quedaba vacío.
   - ¿Se resolvió la completitud catastral? ¿Existe algún lote labrado, terraza andina o campo agrícola en el Panel A que haya quedado como un hueco vacío en el Panel B?
2. DETECCIÓN DE CERCOS VIVOS Y LINDEROS ARBÓREOS (Linear TOF):
   - Examina los trazos Verde Esmeralda Neón (#22C55E).
   - ¿Coinciden fielmente con las hileras de árboles perennes (retamas, lecheros, cipreses, pencos) visibles en el Panel A?
   - ¿Se diferencian claramente de los bosquetes verdes compactos (#15803D)?
3. SUPRESIÓN DE FALSOS POSITIVOS EN CASAS Y DETECCIÓN DE GALPONES:
   - Revisa las viviendas magenta y galpones grises.
   - ¿Se eliminaron los falsos positivos que antes ocurrían en parcelas de tierra volcánica seca? ¿Poseen sombra proyectada SW real?
   - ¿Cómo es la separación entre galpones de chapa de zinc (#64748B) e invernaderos plásticos (#06B6D4)?
4. DELINEACIÓN DE PATIOS CAMPESINOS Y ERAS:
   - Revisa las áreas en Ocre Arena (#F59E0B) contiguas a las casas.
   - ¿Aíslan con precisión el suelo compactado de trilla y maniobra sin invadir el área de cultivo?
5. SOBREFRAGMENTACIÓN Y REGULARIDAD TOPOLÓGICA LADM ISO 19152:
   - ¿Observas parcelas partidas artificialmente por diferencias de humedad interna del suelo?
   - ¿La simplificación planar mantiene bordes limpios sin solapes ni deformaciones excesivas?

Responde ÚNICA Y ESTRICTAMENTE con un objeto JSON con este formato exacto:
{{
  "modelo_auditor": "string",
  "scorecard_multimodal": {{
    "completitud_catastral_recall_upas": 0-100,
    "fidelidad_cercos_vivos_linear_tof": 0-100,
    "precision_exclusion_casas_y_galpones": 0-100,
    "delineacion_patios_campesinos": 0-100,
    "regularidad_topologica_ladm": 0-100,
    "calidad_global_consolidada": 0-100
  }},
  "diagnostico_comparativo_por_cuadrante": {{
    "cuadrante_1_valle_aluvial": "string",
    "cuadrante_2_laderas_andinas": "string",
    "cuadrante_3_piedemonte_chimborazo": "string",
    "cuadrante_4_interfase_vial_batzacon": "string"
  }},
  "analisis_detallado": {{
    "evaluacion_completitud_vs_v63": "string",
    "evaluacion_cercos_vivos_vs_ortofoto": "string",
    "supresion_falsos_positivos_viviendas": "string",
    "comportamiento_galpones_y_patios": "string",
    "verificacion_solapes_y_ladm": "string"
  }},
  "fortalezas_visuales_identificadas": [
    "string", "string", "string"
  ],
  "oportunidades_de_mejora_residual": [
    "string", "string", "string"
  ],
  "veredicto_tribunal": "Aprobado con Distinción / Sobresaliente / Requiere Ajustes"
}}
"""

async def run_single_multimodal_audit(client, model_name, images_paths, prompt):
    print(f"\n[MoE Multimodal] Iniciando auditoría con {model_name} (Thinking: HIGH)...")
    t0 = time.time()
    
    parts = []
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
                print(f"    - Adjuntada comparativa: {os.path.basename(img_p)} ({len(img_bytes)/1024:.1f} KB, res: {im.shape[1]}x{im.shape[0]})")
            else:
                parts.append(types.Part.from_bytes(data=raw_b, mime_type="image/png"))
                print(f"    - Adjuntada imagen original: {os.path.basename(img_p)} ({len(raw_b)/1024:.1f} KB)")
        else:
            print(f"    [Aviso]: No se encontro la imagen {img_p}")
            
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
    print("AUDITORÍA MoE MULTIMODAL DUAL EN PARALELO - RENDERS A100 v65")
    print("Modelos: Gemini 4 Argon vs. Gemini 3.1 Pro (Vertex AI Global - Thinking HIGH)")
    print("Enfoque: Comparativa Panel A (Ortofoto Real) vs Panel B (Delimitacion SOTA)")
    print("=" * 80)

    exp_dir = r"K:\Mi unidad\CURSOS Y CAPACITACIÓN PERSONAL\MASTER-INTELIGENCIA ARTIFICIAL\Aula Master IA\02_Segundo_Cuatrimestre\09_TFM\06_Codigo_y_Experimentos\experimentos\v65.0.0"
    metrics_json = os.path.join(exp_dir, "metrics_summary_v65_a100.json")
    
    if os.path.exists(metrics_json):
        with open(metrics_json, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)
    else:
        metrics_data = {"version": "v65.0.0-A100-SOTA", "aviso": "Metricas pendientes de ejecucion Colab A100"}

    image_paths = [
        os.path.join(exp_dir, f"comparativa_a100_cuadrante_{cid}_san_andres.png")
        for cid in range(1, 5)
    ]

    prompt = build_multimodal_audit_prompt_v65(metrics_data)
    client = genai.Client()

    task_gemini4 = run_single_multimodal_audit(client, "gemini-4-argon", image_paths, prompt)
    task_gemini31 = run_single_multimodal_audit(client, "gemini-3.1-pro-preview", image_paths, prompt)

    res_gemini4, res_gemini31 = await asyncio.gather(task_gemini4, task_gemini31)

    audit_consolidated = {
        "fecha": str(datetime.date.today()),
        "tipo_auditoria": "Multimodal Visual MoE Panel A vs Panel B (4 Renders VHR A100)",
        "acelerador": "NVIDIA A100-SXM4-80GB",
        "version_evaluada": "v65.0.0-A100-SOTA",
        "cuadrantes_evaluados": 4,
        "modelos": {
            "gemini_4_argon": res_gemini4,
            "gemini_3_1_pro": res_gemini31
        }
    }

    out_json = os.path.join(exp_dir, "auditoria_moe_multimodal_v65.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(audit_consolidated, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Dictamen consolidado MoE Multimodal v65 guardado en: {out_json}")

if __name__ == "__main__":
    asyncio.run(main())
