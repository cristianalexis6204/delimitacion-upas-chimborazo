"""
Motor MoE Dual en Paralelo: Auditoria de Deteccion y Algoritmos (Gemini 4 Argon vs. Gemini 3.1 Pro)
TFM UNIR - Delimitacion Automatizada de UPAs en Chimborazo
Autor: Cristian Alexis Garcia Pumagualle

Este script:
1. Conecta con Vertex AI en region global bajo el proyecto 'tesis-bi-edison-paredes' (Edison).
2. Lanza en paralelo dos motores de razonamiento maximo (ThinkingLevel.HIGH):
   - Motor A: gemini-4-argon
   - Motor B: gemini-3.1-pro-preview
3. Cada motor realiza la auditoria exhaustiva de los 4 scripts troncales (1,940 lineas) y las metricas de deteccion.
4. Genera reportes/auditoria_moe_dual_paralelo.json con telemetria de tokens, scorecards y recomendaciones.
"""

import os
import sys
import json
import time
import asyncio

os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
os.environ["GOOGLE_CLOUD_PROJECT"] = "tesis-bi-edison-paredes"
os.environ["GOOGLE_CLOUD_LOCATION"] = "global"

from google import genai
from google.genai import types

def build_audit_prompt(payload):
    return f"""
Actúa como un Comité Colegiado de Mixture of Experts (MoE) de Máximo Nivel Técnico para realizar una Auditoría Rigurosa y Crítica sobre la Calidad de Detección de UPAs y la Arquitectura de Algoritmos/Scripts de este Trabajo Fin de Máster (TFM en Inteligencia Artificial).

DATOS TÉCNICOS Y PERFILES DE CÓDIGO A AUDITAR:
{json.dumps(payload, indent=2, ensure_ascii=False)}

CONTEXTO Y ALCANCE DE LA EVALUACIÓN:
Se auditan 1,940 líneas de código Python en 4 componentes troncales:
1. 'run_san_andres_v60_master_pipeline.py' (991 líneas): Pipeline orquestador maestro que integra inferencia, generación de mosaicos 4 paneles, leyendas cartográficas, KML multicapa y reporte PDF ReportLab.
2. 'sam2_ridge_hybrid_segmenter.py' (368 líneas): Segmentador híbrido SAM-2.1 Hiera ViT-B + filtro de crestas multiescala Hessian-Meijering (sigmas [1.2, 2.0, 3.2]) + argmax planar competitivo + regularización Douglas-Peucker orientada (4 a 10 vértices).
3. 'rural_road_network_extractor.py' (275 líneas): Extractor de red vial rural (OSM caché local + Overpass API + crestas morfológicas) con buffer de servidumbre LPIS de 3.5 m.
4. 'rural_building_extractor.py' (306 líneas): Extractor de edificaciones rurales y viviendas campesinas con disambiguación CIELAB (a* > 134), sombra relativa 3D (Delta-L >= 14.0) y filtro de rectangularidad (>= 0.62).

MÉTRICAS REGISTRADAS:
- San Andrés (359 UPAs, 40.22 ha, 60 viviendas, 156 vías): mIoU 83.6%, Dice 91.1%, BF1 92.4%, solape 0.00%, conformidad LADM 100.0%.
- Ground Truth Calpi (100 UPAs): SAM-2.1 + Meijering (mIoU 80.5%) vs. SAM ViT-B (68.2%) vs. Mask R-CNN (61.4%) vs. UNet (58.7%).

MANDATOS DE CADA ROL EXPERTO:
1. Experto en Visión por Computador y Sensores Remotos: Evalúa la solidez de SAM-2.1 Hiera + Meijering. ¿La escala de sigmas es óptima para cercas vivas y linderos andinos? ¿Cómo se compara con el estado del arte de Lucas et al. (Science of Remote Sensing, 2026 / arXiv:2510.25239)? ¿Qué debilidades presenta en laderas empinadas y sombras orográficas?
2. Experto en Geodesia y Catastro LADM ISO 19152: Evalúa la estanqueidad planar (0.00% solape), la regularización Douglas-Peucker (4-10 vértices), el buffer vial de 3.5m y la prevención de falsos positivos en viviendas.
3. Experto en Ingeniería de Software, HPC y MLOps: Evalúa la calidad del código, deuda técnica, modularidad, uso de CUDA, manejo de memoria GPU (mixed precision FP16/BF16) y escalabilidad provincial del script monolítico de 991 líneas.
4. Experto Agronómico y Validación Censal (RENAGRO / MAG): Evalúa la fidelidad de las parcelas frente al minifundio andino de Chimborazo, terrazas, rotación de cultivos y encuestas censales.
5. Juez Sintetizador MoE: Asigna puntuaciones numéricas (0-100) en cada dimensión, identifica las 3 mayores fortalezas, los 3 mayores cuellos de botella y propone las 5 recomendaciones prioritarias de optimización.

Responde ÚNICA Y ESTRICTAMENTE con un objeto JSON con este formato exacto:
{{
  "modelo_auditor": "string",
  "scorecard_global": {{
    "calidad_deteccion_contornos": 0-100,
    "conformidad_catastral_ladm": 0-100,
    "arquitectura_software_y_modularidad": 0-100,
    "eficiencia_hpc_y_cuda": 0-100,
    "pertinencia_agronomica_censo": 0-100,
    "puntuacion_promedio_general": 0.0-100.0
  }},
  "evaluacion_expertos": {{
    "vision_por_computador": {{
      "puntaje": 0-100,
      "analisis_critico": "string con diagnostico profundo",
      "oportunidades_sota_lucas_2026": "string con recomendacion sobre Lucas et al."
    }},
    "catastro_ladm_iso19152": {{
      "puntaje": 0-100,
      "analisis_critico": "string con diagnostico topologico"
    }},
    "software_mlops_hpc": {{
      "puntaje": 0-100,
      "deuda_tecnica_identificada": "string con diagnostico de scripts",
      "recomendacion_cuda_memoria": "string sobre optimizacion de VRAM"
    }},
    "agronomia_renagro": {{
      "puntaje": 0-100,
      "analisis_minifundio_andino": "string con diagnostico censal"
    }}
  }},
  "sintesis_estrategica": {{
    "tres_mayores_fortalezas": ["string", "string", "string"],
    "tres_mayores_cuellos_de_botella": ["string", "string", "string"],
    "cinco_recomendaciones_prioritarias": [
      {{"orden": 1, "titulo": "string", "tipo": "Quick Win" | "Estructural", "impacto": "Alto" | "Medio", "esfuerzo": "Bajo" | "Medio" | "Alto", "descripcion": "string"}},
      {{"orden": 2, "titulo": "string", "tipo": "Quick Win" | "Estructural", "impacto": "Alto" | "Medio", "esfuerzo": "Bajo" | "Medio" | "Alto", "descripcion": "string"}},
      {{"orden": 3, "titulo": "string", "tipo": "Quick Win" | "Estructural", "impacto": "Alto" | "Medio", "esfuerzo": "Bajo" | "Medio" | "Alto", "descripcion": "string"}},
      {{"orden": 4, "titulo": "string", "tipo": "Quick Win" | "Estructural", "impacto": "Alto" | "Medio", "esfuerzo": "Bajo" | "Medio" | "Alto", "descripcion": "string"}},
      {{"orden": 5, "titulo": "string", "tipo": "Quick Win" | "Estructural", "impacto": "Alto" | "Medio", "esfuerzo": "Bajo" | "Medio" | "Alto", "descripcion": "string"}}
    ],
    "veredicto_final_memoria_tfm": "string con dictamen conclusivo para la defensa del TFM"
  }}
}}
"""

async def run_single_moe_audit(client, model_name, prompt):
    print(f"\n[INICIANDO AUDITORIA MoE]: Modelo {model_name} (Thinking HIGH en Vertex AI)...")
    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.HIGH, include_thoughts=True),
        response_mime_type="application/json"
    )

    t0 = time.time()
    # Ejecutar en hilo asíncrono para paralelismo real
    loop = asyncio.get_event_loop()
    res = await loop.run_in_executor(
        None,
        lambda: client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config
        )
    )
    dt = time.time() - t0

    dictamen = json.loads(res.text)
    if isinstance(dictamen, list):
        if len(dictamen) > 0 and isinstance(dictamen[0], dict):
            dictamen = dictamen[0]
        else:
            dictamen = {"raw_list": dictamen}
    dictamen["modelo_auditor"] = model_name

    thinking_tokens = getattr(res.usage_metadata, "thoughts_token_count", 0) if hasattr(res, "usage_metadata") else 0
    total_tokens = getattr(res.usage_metadata, "total_token_count", 0) if hasattr(res, "usage_metadata") else 0

    dictamen["telemetria"] = {
        "tiempo_segundos": round(dt, 2),
        "thinking_tokens": thinking_tokens,
        "total_tokens": total_tokens,
        "modelo": model_name
    }

    score = dictamen.get("scorecard_global", {}).get("puntuacion_promedio_general", 0.0)
    print(f"[FINALIZADO]: {model_name} completó en {dt:.1f}s ({thinking_tokens} thinking tokens) | Score Promedio: {score}/100")
    return dictamen

async def main_async():
    repo_root = r"K:\Mi unidad\CURSOS Y CAPACITACIÓN PERSONAL\MASTER-INTELIGENCIA ARTIFICIAL\Aula Master IA\02_Segundo_Cuatrimestre\09_TFM\delimitacion-upas-chimborazo"
    payload_file = os.path.join(repo_root, "data", "processed", "payload_auditoria_algoritmos_y_deteccion.json")
    reportes_dir = os.path.join(repo_root, "reportes")
    os.makedirs(reportes_dir, exist_ok=True)
    out_json = os.path.join(reportes_dir, "auditoria_moe_dual_paralelo.json")

    print("========================================================================")
    print("TORNEO MoE DUAL EN PARALELO: GEMINI 4 ARGON vs. GEMINI 3.1 PRO PREVIEW")
    print("Proyecto: tesis-bi-edison-paredes | Region: global | Thinking: HIGH")
    print("========================================================================")

    if not os.path.exists(payload_file):
        print(f"[ERROR] No existe el archivo de perfilado: {payload_file}")
        sys.exit(1)

    with open(payload_file, "r", encoding="utf-8") as f:
        payload = json.load(f)

    prompt = build_audit_prompt(payload)

    client = genai.Client(vertexai=True, project="tesis-bi-edison-paredes", location="global")

    # Lanzar ambos motores MoE en paralelo
    t_start = time.time()
    task_gemini4 = run_single_moe_audit(client, "gemini-4-argon", prompt)
    task_gemini31 = run_single_moe_audit(client, "gemini-3.1-pro-preview", prompt)

    resultado_gemini4, resultado_gemini31 = await asyncio.gather(task_gemini4, task_gemini31)
    t_total = time.time() - t_start

    print("\n========================================================================")
    print(f"[AMBOS MOTORES CONCLUYERON CON EXITO EN PARALELO EN {t_total:.1f}s]")
    print(f" - Gemini 4 Argon: Score {resultado_gemini4['scorecard_global']['puntuacion_promedio_general']}/100")
    print(f" - Gemini 3.1 Pro: Score {resultado_gemini31['scorecard_global']['puntuacion_promedio_general']}/100")
    print("========================================================================")

    dual_output = {
        "metadata": {
            "fecha_auditoria": time.strftime("%Y-%m-%d %H:%M:%S"),
            "proyecto": "tesis-bi-edison-paredes",
            "region": "global",
            "tiempo_total_paralelo_s": round(t_total, 2)
        },
        "motor_a_gemini_4_argon": resultado_gemini4,
        "motor_b_gemini_3_1_pro": resultado_gemini31
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(dual_output, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Resultados consolidados guardados en: {out_json}")

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
