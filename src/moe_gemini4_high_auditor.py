"""
Motor MoE (Mixture of Experts) Multi-Agente con Gemini 4 Argon en Thinking HIGH
TFM UNIR - Auditoria Censal-Agronomica de UPAs Delimitadas por IA
Autor: Cristian Alexis Garcia Pumagualle

Este script:
1. Conecta con Vertex AI en region global bajo el proyecto 'tesis-bi-edison-paredes' (Edison).
2. Orquesta un pool de 4 expertos impulsados por Gemini 4 Argon con Thinking HIGH:
   - Experto Agronomo Espectral (compatibilidad fenologica y firmas ExG/VARI/Meijering).
   - Auditor Catastral LADM ISO 19152 (regularidad, vertices 4-10, topologia planar, buffers).
   - Especialista Geodesico RENAGRO (desplazamiento GPS 10-25m y discrepancia superficial).
   - Juez Sintetizador MoE (calculo del ICPC ponderado y semaforo VERDE/AMARILLO/ROJO).
3. Audita las 17 encuestas agropecuarias de San Andres.
4. Genera el registro exhaustivo en reportes/auditoria_moe_gemini4_san_andres.json.
"""

import os
import sys
import json
import time

# Configuracion oficial de Vertex AI para Gemini 4 Argon en region global
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
os.environ["GOOGLE_CLOUD_PROJECT"] = "tesis-bi-edison-paredes"
os.environ["GOOGLE_CLOUD_LOCATION"] = "global"

from google import genai
from google.genai import types

def build_moe_prompt(payload):
    return f"""
Actúa como un Sistema Colegiado de Mixture of Experts (MoE) para auditar la consistencia entre una Encuesta Agropecuaria (RENAGRO/MAG) y una Parcela delimitada por IA (SAM 2.1 + Filtro Meijering).

DATOS DE ENTRADA DE LA UPA A AUDITAR:
{json.dumps(payload, indent=2, ensure_ascii=False)}

ROLES Y MANDATOS DE EVALUACIÓN:
1. Experto Agrónomo Espectral: Analiza rigurosamente si el cultivo declarado ("{payload['cultivo_declarado']}") es biológica y fenológicamente compatible con los índices de verdor ExG ({payload['firmas_espectrales']['exg_excess_green']}), VARI ({payload['firmas_espectrales']['vari_index']}), color RGB y la textura morfológica ({payload['firmas_espectrales']['descripcion_textura_meijering']}). Si el suelo está labrado o con biomasa baja, evalúa si corresponde a rotación o cosecha.
2. Auditor Catastral LADM ISO 19152: Evalúa la regularidad poligonal ({payload['n_vertices']} vértices, óptimo entre 4 y 10), estanqueidad topológica (0 solapes), buffer de servidumbre vial LPIS (3.5 m) y exclusión total de viviendas/patios en el área cultivada.
3. Especialista Geodésico RENAGRO: Evalúa si el desplazamiento GPS de {payload['desplazamiento_gps_m']} metros entre la vivienda encuestada y el centroide de la UPA se absorbe dentro de la tolerancia de campo (10 a 25 m), y evalúa el error de superficie del {payload['error_superficie_pct']}% entre lo declarado ({payload['superficie_declarada_m2']} m2) y lo delimitado ({payload['superficie_delimitada_m2']} m2). Si el error de área es > 40%, penaliza el puntaje dimensional.
4. Juez Sintetizador MoE: Pondera los dictámenes aplicando la fórmula oficial del Índice de Concordancia Predial-Censal (ICPC):
   ICPC = 0.40 * agronómico + 0.35 * dimensional + 0.25 * catastral
   Clasifica el resultado en:
   - "CONCORDANTE" (ICPC >= 75.0, semáforo "VERDE")
   - "OBSERVABLE" (55.0 <= ICPC < 75.0, semáforo "AMARILLO")
   - "DISCORDANTE" (ICPC < 55.0, semáforo "ROJO")

Debes responder ÚNICA Y ESTRICTAMENTE con un objeto JSON válido con este esquema exacto:
{{
  "id_auditoria": {payload["id_auditoria"]},
  "upa_encuesta_id": "{payload["upa_encuesta_id"]}",
  "segmentacion_upa_id": "{payload["segmentacion_upa_id"]}",
  "cultivo_declarado": "{payload["cultivo_declarado"]}",
  "cuadrante": "{payload["cuadrante"]}",
  "experto_agronomo": {{
    "puntaje": 0-100,
    "compatibilidad": "Alta" | "Media" | "Baja",
    "analisis_fenologico": "string conciso con rigor tecnico",
    "observaciones": "string"
  }},
  "experto_catastral_ladm": {{
    "puntaje": 0-100,
    "cumplimiento_iso19152": true | false,
    "analisis_topologico": "string conciso sobre geometria y linderos"
  }},
  "experto_geodesico_renagro": {{
    "puntaje": 0-100,
    "absorcion_desplazamiento_gps": true | false,
    "analisis_dimensional": "string conciso sobre desplazamiento y precision de superficie"
  }},
  "dictamen_sintetizador": {{
    "icpc_puntaje": 0.0-100.0,
    "clasificacion": "CONCORDANTE" | "OBSERVABLE" | "DISCORDANTE",
    "color_semaforo": "VERDE" | "AMARILLO" | "ROJO",
    "resumen_ejecutivo": "string con la justificacion colegiada formal",
    "recomendacion_censo": "string con recomendacion para el censo nacional agropecuario"
  }}
}}
"""

def main():
    repo_root = r"K:\Mi unidad\CURSOS Y CAPACITACIÓN PERSONAL\MASTER-INTELIGENCIA ARTIFICIAL\Aula Master IA\02_Segundo_Cuatrimestre\09_TFM\delimitacion-upas-chimborazo"
    payload_file = os.path.join(repo_root, "data", "processed", "payload_encuestas_auditables_san_andres.json")
    reportes_dir = os.path.join(repo_root, "reportes")
    os.makedirs(reportes_dir, exist_ok=True)
    out_json = os.path.join(reportes_dir, "auditoria_moe_gemini4_san_andres.json")

    print("========================================================================")
    print("SISTEMA MoE MULTI-AGENTE: AUDITORIA CON GEMINI 4 ARGON EN THINKING HIGH")
    print("Proyecto: tesis-bi-edison-paredes | Region: global | Entorno: Vertex AI")
    print("========================================================================")

    if not os.path.exists(payload_file):
        print(f"[ERROR] Archivo de payloads no encontrado: {payload_file}")
        sys.exit(1)

    with open(payload_file, "r", encoding="utf-8") as f:
        payloads = json.load(f)

    print(f"[OK] {len(payloads)} encuestas cargadas para auditoria MoE.")

    # Inicializar cliente Vertex AI
    client = genai.Client(vertexai=True, project="tesis-bi-edison-paredes", location="global")
    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.HIGH, include_thoughts=True),
        response_mime_type="application/json"
    )

    resultados_auditoria = []
    total = len(payloads)

    for i, payload in enumerate(payloads, 1):
        upa_id = payload["upa_encuesta_id"]
        seg_id = payload["segmentacion_upa_id"]
        cultivo = payload["cultivo_declarado"]
        print(f"\n[{i:02d}/{total:02d}] Auditando Encuesta {upa_id} -> UPA {seg_id} ({cultivo})...")

        prompt = build_moe_prompt(payload)

        retries = 3
        success = False
        while retries > 0 and not success:
            try:
                t0 = time.time()
                res = client.models.generate_content(
                    model="gemini-4-argon",
                    contents=prompt,
                    config=config
                )
                dt = time.time() - t0

                # Extraer JSON estructurado
                dictamen = json.loads(res.text)

                # Extraer metadatos de tokens y pensamientos si estan disponibles
                thinking_tokens = getattr(res.usage_metadata, "thoughts_token_count", 0) if hasattr(res, "usage_metadata") else 0
                total_tokens = getattr(res.usage_metadata, "total_token_count", 0) if hasattr(res, "usage_metadata") else 0

                dictamen["telemetria"] = {
                    "tiempo_segundos": round(dt, 2),
                    "thinking_tokens": thinking_tokens,
                    "total_tokens": total_tokens,
                    "modelo": "gemini-4-argon",
                    "modo_razonamiento": "ThinkingLevel.HIGH"
                }

                # Anexar coordenadas espaciales para el KML
                dictamen["coordenadas"] = payload["coordenadas"]
                dictamen["metricas_originales"] = {
                    "superficie_declarada_m2": payload["superficie_declarada_m2"],
                    "superficie_delimitada_m2": payload["superficie_delimitada_m2"],
                    "error_superficie_pct": payload["error_superficie_pct"],
                    "desplazamiento_gps_m": payload["desplazamiento_gps_m"],
                    "n_vertices": payload["n_vertices"],
                    "clase_ladm": payload["clase_ladm"]
                }

                resultados_auditoria.append(dictamen)

                icpc = dictamen["dictamen_sintetizador"]["icpc_puntaje"]
                clasif = dictamen["dictamen_sintetizador"]["clasificacion"]
                semaforo = dictamen["dictamen_sintetizador"]["color_semaforo"]
                print(f" -> [OK {dt:.1f}s | {thinking_tokens} thinking tokens] ICPC: {icpc}/100 | Dictamen: {clasif} ({semaforo})")
                success = True

            except Exception as e:
                retries -= 1
                print(f" -> [ERROR]: {e}. Reintentos restantes: {retries}")
                time.sleep(3)

        # Pequeña pausa para regular el ritmo de la API
        time.sleep(1.0)

    # Guardar resultados finales
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(resultados_auditoria, f, indent=2, ensure_ascii=False)

    print("\n========================================================================")
    print(f"[AUDITORIA MoE FINALIZADA] {len(resultados_auditoria)}/{total} encuestas auditadas.")
    print(f"Archivo generado: {out_json}")
    print("========================================================================")

if __name__ == "__main__":
    main()
