"""
================================================================================
ORQUESTADOR MoE-KAIZEN v3.0 - TRIBUNAL MULTI-AGENTE EN BUCLE CERRADO
Inspección Desacoplada por Cuadrante a Altitud Virtual de 250 m + BARS Rubric
10 Agentes: 7 Especialistas BARS + Crítico Adversarial + Juez LADM + Sintetizador
TFM UNIR - Delimitación Automatizada de UPAs en Chimborazo
================================================================================
"""

import os
import sys
import json
import time
import asyncio
import datetime
import re
from typing import Dict, Any, List, Optional

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

# Configuración canónica estricta de Vertex AI en región global para gemini-4-argon (Proyecto tesis-bi-edison-paredes)
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
os.environ["GOOGLE_CLOUD_PROJECT"] = "tesis-bi-edison-paredes"
os.environ["GOOGLE_CLOUD_LOCATION"] = "global"
if "GEMINI_API_KEY" in os.environ:
    del os.environ["GEMINI_API_KEY"]

from google import genai
from google.genai import types

# Importar módulos core
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from gis_telemetry_probe import run_gis_telemetry_probe
from visual_grounding_renderer import (
    prepare_individual_quadrant_dupla_parts,
    prepare_quadrant_zoom_parts_for_gemini,
    prepare_atlas_part_with_grid_for_gemini,
    compute_differential_delta_map,
    cv2_imread_unicode
)
from kaizen_memory_ledger import KaizenMemoryLedger, BARS_RUBRIC_SPEC

def call_gemini_with_backoff(client, model_name, contents, config, max_retries=7, initial_delay=15.0):
    for attempt in range(1, max_retries + 1):
        try:
            return client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config
            )
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "Resource exhausted" in err_msg:
                wait_sec = initial_delay * attempt
                print(f"[RateLimit 429] Intento {attempt}/{max_retries}. Pausando {wait_sec:.1f}s antes de reintentar...")
                time.sleep(wait_sec)
            else:
                if attempt == max_retries:
                    raise
                time.sleep(5.0)
    raise RuntimeError(f"Agotados los {max_retries} intentos de llamada a Gemini por cuota 429.")

def extract_function_code(script_path: str, func_name: str) -> str:
    if not os.path.exists(script_path):
        return f"# Error: {script_path} no encontrado"
    with open(script_path, "r", encoding="utf-8") as f:
        content = f.read()
    pattern = rf"(def {func_name}\([^)]*\):[\s\S]*?)(?=\ndef |\Z)"
    match = re.search(pattern, content)
    if match:
        return match.group(1).strip()
    return f"# Función {func_name} no encontrada por regex"

def build_specialist_prompt_v2(
    agent_id: str,
    agent_role: str,
    target_layer: str,
    func_name: str,
    func_code: str,
    telemetry_probe: Dict[str, Any],
    prior_tickets: List[Dict[str, Any]],
    version_tag: str
) -> str:
    return f"""
Actúa como el {agent_role} dentro del Tribunal Colegiado de Peritaje Catastral MoE-Kaizen v3.0 (LADM ISO 19152 / VHR Satellite Vision).

Tu misión es AUDITAR CON RIGOR CIENTÍFICO, DESGLOSE POR CUADRANTE Y ANCLAJE ESPACIAL A 250 m DE ALTITUD VIRTUAL la capa: '{target_layer}' en la versión {version_tag}.

TIENES A TU DISPOSICIÓN:
1. CÓDIGO PYTHON EXACTO DEL ALGORITMO EXPERTO ({func_name}):
```python
{func_code}
```

2. TELEMETRÍA DETERMINISTA GIS PRE-VLM v3.0 (MEDICIONES EXACTAS DE LA SONDA POR CUADRANTE):
{telemetry_probe.get('telemetry_text_prompt', 'Telemetría no disponible')}

3. TICKETS KAIZEN ABIERTOS DE LA VERSIÓN PREVIA:
{json.dumps(prior_tickets, indent=2, ensure_ascii=False) if prior_tickets else 'No hay tickets previos pendientes para esta capa.'}

4. 4 LÁMINAS ANALÍTICAS INDEPENDIENTES (UNA POR CADA CUADRANTE: Q1 Centro, Q2 Chocaví, Q3 Pulucate, Q4 Batzacón):
Cada lámina de cuadrante contiene:
- MITAD SUPERIOR: Vista completa del cuadrante R=350m con rejilla Set-of-Mark (A1..F6, cada celda ~58x58 m).
  Contiene dos recuadros indicadores de micro-zoom: [ZOOM A: 250m] en oro y [ZOOM B: 250m] en cian.
- MITAD INFERIOR (MICRO-ZOOMS A ALTITUD VIRTUAL DE 250 m | VENTANA: 110 m x 110 m, ampliación 3x):
  * ZOOM A (Panel Izquierdo Inferior): Foco en el clúster de edificaciones rurales y patios campesinos peri-domésticos.
  * ZOOM B (Panel Derecho Inferior): Foco en la zona de mayor densidad de minifundio, linderos agrícolas Meijering y cercos vivos.

SISTEMA DE CALIFICACIÓN: RÚBRICA BARS DE 5 DIMENSIONES:
Debes asignar un puntaje de 0 a 100 en cada una de las 5 dimensiones ponderadas:
- recall_sensibilidad (peso 30%): Cobertura de objetos reales visibles en las celdas A1-F6 y en los micro-zooms.
- precision_especificidad (peso 25%): Ausencia de falsas alarmas, caminos como parcelas, etc.
- adherencia_morfologica (peso 20%): Geometría regularizada, ortogonalidad, sin doble trazo.
- conformidad_topologica_circular (peso 15%): Confinamiento estricto al círculo R=350m sin derrames exteriores.
- rigor_codigo_y_documentacion (peso 10%): Coherencia exacta entre docstrings, parámetros y código.

Calcula el score_ponderado_bars = (recall*0.30 + precision*0.25 + morfologia*0.20 + circular*0.15 + codigo*0.10).

Responde ÚNICA Y ESTRICTAMENTE con un objeto JSON:
{{
  "agente_id": "{agent_id}",
  "rol_especialista": "{agent_role}",
  "capa_auditada": "{target_layer}",
  "calificacion_por_cuadrante": {{
    "Q1_San_Andres_Centro": 0-100,
    "Q2_Chocavi_Laderas": 0-100,
    "Q3_Pulucate_Quebradas": 0-100,
    "Q4_Batzacon_Planicie": 0-100
  }},
  "inspeccion_micro_zoom_250m": {{
    "zoom_a_peridomestico": "string con diagnóstico minucioso de lo observado en las ventanas Zoom A de 250m",
    "zoom_b_minifundio": "string con diagnóstico minucioso de lo observado en las ventanas Zoom B de 250m"
  }},
  "rubrica_bars": {{
    "recall_sensibilidad": 0-100,
    "precision_especificidad": 0-100,
    "adherencia_morfologica": 0-100,
    "conformidad_topologica_circular": 0-100,
    "rigor_codigo_y_documentacion": 0-100,
    "score_ponderado_bars": 0-100
  }},
  "evaluacion_tickets_previos": [
    {{
      "ticket_id": "string",
      "estado": "RESUELTO / PARCIAL / PENDIENTE",
      "evidencia": "string con celda ej. Q2-C4"
    }}
  ],
  "hallazgos_por_celda_set_of_mark": [
    {{
      "cuadrante": "Q1 / Q2 / Q3 / Q4",
      "celda": "A1..F6",
      "tipo_hallazgo": "FALSO_POSITIVO / FALSO_NEGATIVO / DERRAME_CIRCULAR / RECORTE_CORRECTO",
      "descripcion": "string preciso"
    }}
  ],
  "analisis_critico_del_codigo": "string detallado",
  "ajustes_exactos_propuestos": [
    {{
      "parametro_o_logica": "string",
      "valor_actual": "string",
      "valor_propuesto": "string",
      "justificacion": "string"
    }}
  ],
  "veredicto_capa": "Excelente / Aprobado / Requiere Ajustes"
}}
"""

def build_adversarial_critic_prompt(
    specialist_audits: Dict[str, Any],
    telemetry_probe: Dict[str, Any]
) -> str:
    return f"""
Actúa como el AGENTE 8: CRÍTICO ADVERSARIAL Y AUDITOR DE FRONTERAS (Red-Team Reviewer) en el Tribunal MoE-Kaizen v3.0.

Tu rol fundamental NO es ser complaciente, sino ACTUAR COMO ABOGADO DEL DIABLO para garantizar excelencia científica:
1. Detectar inflaciones artificiales de puntajes: Si un especialista califica con > 90/100 pero la telemetría GIS muestra colisiones, derrames o contradicciones, debes sancionarlo.
2. Auditar la Matriz de Colisiones Inter-Capa en metros cuadrados:
   - ¿Chocan viviendas con vías, patios o bosques?
   - ¿Sustraen los bosques o vías suelo agrícola legítimo?
3. Evaluar consistencia espacial entre los 4 cuadrantes heterogéneos: ¿Hay algún cuadrante descuidado?

EXPEDIENTE COMPLETO DE LOS 7 ESPECIALISTAS (CON SCORES POR CUADRANTE):
{json.dumps(specialist_audits, indent=2, ensure_ascii=False)}

MATRIZ DE COLISIÓN TOPOLÓGICA INTER-CAPA Y TELEMETRÍA GIS:
{json.dumps(telemetry_probe, indent=2, ensure_ascii=False)}

Responde ÚNICA Y ESTRICTAMENTE con un objeto JSON:
{{
  "agente_id": "agente_8_critico_adversarial",
  "rol": "Auditor Adversarial de Fronteras y Consistencia Inter-Capa",
  "calificaciones_ajustadas_por_auditoria": {{
    "agente_1_viviendas": 0-100,
    "agente_2_galpones_invernaderos": 0-100,
    "agente_3_patios_eras": 0-100,
    "agente_4_cercos_vivos": 0-100,
    "agente_5_bosques_quebradas": 0-100,
    "agente_6_red_vial": 0-100,
    "agente_7_upas_puras": 0-100
  }},
  "conflictos_inter_capa_detectados": [
    {{
      "capas_en_conflicto": "Capa A vs Capa B",
      "cuadrante_y_celda": "ej. Q2-D3",
      "descripcion": "string",
      "accion_correctiva_sugerida": "string"
    }}
  ],
  "calidad_argumentativa_tribunal": "ALTA / MEDIA / INFLADA",
  "dictamen_adversarial": "string con evaluación implacable"
}}
"""

def build_supreme_judge_prompt_v2(
    specialist_audits: Dict[str, Any],
    critic_audit: Dict[str, Any],
    telemetry_probe: Dict[str, Any],
    historical_summary: str,
    version_tag: str
) -> str:
    return f"""
Actúa como el AGENTE 9: JUEZ SUPREMO Y PRESIDENTE DEL TRIBUNAL MoE-KAIZEN LADM ISO 19152.

Debes emitir la SENTENCIA VINCULANTE sobre la versión {version_tag}.

CRITERIO DE HOMOLOGACIÓN: REGLA DE DOMINANCIA DE PARETO:
Para que {version_tag} sea HOMOLOGADA COMO NUEVA LÍNEA BASE MAESTRA debe cumplir simultáneamente:
1. Cero solape inter-parcelario medido por la sonda: solape == 0.0000% y conformidad LADM 100.0%.
2. Confinamiento circular perfecto: derrame total fuera de R=350m <= 0.01 m².
3. Mejora o preservación del score global ponderado frente al histórico.
4. Ausencia de regresiones críticas en capas estructurales (Viviendas, Vías, UPAs Puras) en cualquiera de los 4 cuadrantes.

EXPEDIENTE COMPLETO:
- HISTORIAL PREVIO (LIBRO MAYOR KAIZEN):
{historical_summary}

- TELEMETRÍA DETERMINISTA GIS v3.0:
{json.dumps(telemetry_probe, indent=2, ensure_ascii=False)}

- 7 ESPECIALISTAS (PUNTAJES BARS Y POR CUADRANTE):
{json.dumps(specialist_audits, indent=2, ensure_ascii=False)}

- DICTAMEN DEL CRÍTICO ADVERSARIAL:
{json.dumps(critic_audit, indent=2, ensure_ascii=False)}

Responde ÚNICA Y ESTRICTAMENTE con un objeto JSON:
{{
  "agente_id": "agente_9_juez_supremo_ladm",
  "rol": "Presidente del Tribunal Supremo MoE LADM ISO 19152",
  "scorecard_global": {{
    "calificacion_topologica_ladm": 0-100,
    "calificacion_cobertura_y_recall": 0-100,
    "calificacion_precision_geometrica": 0-100,
    "calificacion_global_v2": 0-100
  }},
  "calificacion_cuadrantes_global": {{
    "Q1_Centro": 0-100,
    "Q2_Chocavi": 0-100,
    "Q3_Pulucate": 0-100,
    "Q4_Batzacon": 0-100
  }},
  "cumple_puerta_pareto": true / false,
  "sentencia_tribunal": "{version_tag}_SUPERIOR / {version_tag}_EQUIVALENTE / {version_tag}_INFERIOR",
  "veredicto_homologacion": "HOMOLOGADA_NUEVA_LINEA_BASE / REQUIERE_ITERACION_KAIZEN",
  "fundamentacion_juridica_y_tecnica": "string exhaustivo",
  "conclusiones_para_memoria_tfm": [
    "string punto 1",
    "string punto 2",
    "string punto 3"
  ]
}}
"""

def build_patch_synthesizer_prompt(
    judge_verdict: Dict[str, Any],
    critic_audit: Dict[str, Any],
    specialist_audits: Dict[str, Any],
    script_path: str
) -> str:
    script_code = ""
    if os.path.exists(script_path):
        with open(script_path, "r", encoding="utf-8") as f:
            script_code = f.read()

    return f"""
Actúa como el AGENTE 10: ARQUITECTO SINTETIZADOR DE CÓDIGO Y PARCHES KAIZEN.

Tu misión es transformar todas las sugerencias aprobadas del Tribunal en PARCHES EXACTOS DE CÓDIGO PYTHON listos para aplicarse en la siguiente iteración.

SENTENCIA DEL JUEZ SUPREMO:
{json.dumps(judge_verdict, indent=2, ensure_ascii=False)}

DICTAMEN ADVERSARIAL:
{json.dumps(critic_audit, indent=2, ensure_ascii=False)}

AJUSTES PROPUESTOS POR LOS ESPECIALISTAS:
{json.dumps({k: v.get("ajustes_exactos_propuestos", []) for k, v in specialist_audits.items()}, indent=2, ensure_ascii=False)}

CÓDIGO PYTHON ACTUAL ({os.path.basename(script_path)}):
```python
{script_code[:8000]}
... [fragmento representativo] ...
```

Responde ÚNICA Y ESTRICTAMENTE con un objeto JSON:
{{
  "agente_id": "agente_10_patch_synthesizer",
  "rol": "Arquitecto Sintetizador de Código y Parches Kaizen",
  "parches_propuestos": [
    {{
      "modulo_o_funcion": "string",
      "descripcion": "string",
      "target_code": "string exacto a reemplazar",
      "replacement_code": "string con el reemplazo drop-in"
    }}
  ],
  "parametros_actualizados": {{
    "nombre_parametro": "nuevo_valor"
  }},
  "resumen_impacto_esperado": "string"
}}
"""

async def run_specialist_agent_v2(
    client: genai.Client,
    model_name: str,
    cfg: Dict[str, Any],
    script_path: str,
    telemetry_probe: Dict[str, Any],
    prior_tickets: List[Dict[str, Any]],
    version_tag: str,
    exp_dir: str
) -> Dict[str, Any]:
    aid = cfg["id"]
    func_name = cfg["func_name"]
    atlas_img = cfg["atlas_img"]
    
    print(f"[{aid}] Extrayendo código '{func_name}' y generando láminas de cuadrante con Micro-Zoom 250m...")
    func_code = extract_function_code(script_path, func_name)
    prompt = build_specialist_prompt_v2(
        aid, cfg["role"], cfg["layer"], func_name, func_code, telemetry_probe, prior_tickets, version_tag
    )
    
    parts = []
    # Usar las 4 duplas individuales desacopladas por cuadrante de la capa auditada
    layer_num = cfg.get("layer_num", 1)
    quad_parts = prepare_individual_quadrant_dupla_parts(exp_dir, layer_num, max_dim=2400)
    if quad_parts:
        parts.extend(quad_parts)
    else:
        # Fallback a generador de micro-zooms a altitud virtual de 250m por cuadrante
        zoom_foci = telemetry_probe.get("focos_micro_zoom_250m", {})
        export_zooms_dir = os.path.join(exp_dir, "quad_zooms_250m")
        quad_parts = prepare_quadrant_zoom_parts_for_gemini(atlas_img, zoom_foci, export_dir=export_zooms_dir)
        if quad_parts:
            parts.extend(quad_parts)
        else:
            img_part = prepare_atlas_part_with_grid_for_gemini(atlas_img, max_dim=2800)
            if img_part:
                parts.append(img_part)

    parts.append(prompt)
    
    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.HIGH),
        temperature=0.2,
        response_mime_type="application/json"
    )
    
    t0 = time.time()
    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: call_gemini_with_backoff(
                client=client,
                model_name=model_name,
                contents=parts,
                config=config
            )
        )
        elapsed = time.time() - t0
        raw_text = response.text.strip()
        if raw_text.startswith("```json"): raw_text = raw_text[7:]
        if raw_text.endswith("```"): raw_text = raw_text[:-3]
        parsed = json.loads(raw_text.strip())
        score = parsed.get("rubrica_bars", {}).get("score_ponderado_bars", "N/A")
        print(f"[{aid}] Completado en {elapsed:.1f}s | Score BARS Ponderado: {score}/100 | Veredicto: {parsed.get('veredicto_capa')}")
        return parsed
    except Exception as e:
        print(f"[{aid}] Error ({model_name}): {e}")
        return {
            "agente_id": aid,
            "error": str(e),
            "rubrica_bars": {"score_ponderado_bars": 75},
            "veredicto_capa": "Requiere Ajustes"
        }

async def run_orchestrator_cycle(exp_dir: str, version_tag: str, model_name: str = "gemini-4-argon") -> Dict[str, Any]:
    print("=" * 80)
    print(f"INICIANDO TRIBUNAL MoE-KAIZEN v3.0 (ALTITUD VIRTUAL 250m) - {version_tag}")
    print("=" * 80)
    
    # 1. Ejecutar sonda determinista GIS v3.0
    print("\n>>> Paso 1: Ejecutando Sonda Determinista GIS Pre-VLM v3.0...")
    telemetry_probe = run_gis_telemetry_probe(exp_dir)
    print(telemetry_probe["telemetry_text_prompt"])
    
    # 2. Cargar Libro Mayor Kaizen
    ledger = KaizenMemoryLedger()
    hist_summary = ledger.generate_evolution_summary()
    
    client = genai.Client()
    primary_model = model_name
    
    # Localizar script maestro
    script_candidates = [f for f in os.listdir(exp_dir) if f.startswith("colab_a100_") and f.endswith(".py")]
    script_path = os.path.join(exp_dir, script_candidates[0]) if script_candidates else os.path.join(exp_dir, "master.py")
    
    specialist_configs = [
        {
            "id": "agente_1_viviendas",
            "layer_num": 1,
            "role": "Especialista en Detección de Viviendas Rurales y Edificaciones LADM",
            "layer": "Viviendas LADM Circulares con Sombra 3D SO y Rama Concreto Gris",
            "func_name": "algo_1_viviendas_sombra3d_ladm",
            "atlas_img": os.path.join(exp_dir, "dupla_01_viviendas_ladm.png")
        },
        {
            "id": "agente_2_galpones_invernaderos",
            "layer_num": 2,
            "role": "Especialista en Infraestructura Agroindustrial (Invernaderos PGI y Galpones AMRI)",
            "layer": "Invernaderos PGI y Galpones de Zinc AMRI Desbloqueados en Círculo R=350m",
            "func_name": "algo_2_galpones_invernaderos_pgi_amri",
            "atlas_img": os.path.join(exp_dir, "dupla_02_galpones_e_invernaderos.png")
        },
        {
            "id": "agente_3_patios_eras",
            "layer_num": 3,
            "role": "Especialista en Morfología Peri-doméstica y Patios Campesinos",
            "layer": "Patios y Eras Campesinas Circulares con Tensor de Estructura C<0.25",
            "func_name": "algo_3_patios_eras_textura_grafo",
            "atlas_img": os.path.join(exp_dir, "dupla_03_patios_y_eras.png")
        },
        {
            "id": "agente_4_cercos_vivos",
            "layer_num": 4,
            "role": "Especialista en Agroforestería y Linderos Arbóreos (Trees Outside Forests - Linear TOF)",
            "layer": "Cercos Vivos Continuos con Poda de Nodos >=3 y Recorte Circular",
            "func_name": "algo_4_cercos_vivos_frangi_skeleton",
            "atlas_img": os.path.join(exp_dir, "dupla_04_cercos_vivos_tof.png")
        },
        {
            "id": "agente_5_bosques_quebradas",
            "layer_num": 5,
            "role": "Especialista en Geomorfología Riparia, Cañones y Masas Forestales",
            "layer": "Bosques y Quebradas Riparias con Tensor de Estructura C<0.32 y Recorte Circular",
            "func_name": "algo_5_bosques_quebradas_dosel_ripario",
            "atlas_img": os.path.join(exp_dir, "dupla_05_bosques_y_quebradas.png")
        },
        {
            "id": "agente_6_red_vial",
            "layer_num": 6,
            "role": "Especialista en Vialidad Rural y Servidumbres de Tránsito LPIS",
            "layer": "Red Vial Rural Oficial LPIS Acotada al Perímetro Circular R=350m",
            "func_name": "algo_6_red_vial_lpis",
            "atlas_img": os.path.join(exp_dir, "dupla_06_red_vial_lpis.png")
        },
        {
            "id": "agente_7_upas_puras",
            "layer_num": 7,
            "role": "Especialista en Delineación de Parcelas Agrícolas (SAM-HQ + Meijering)",
            "layer": "UPAs Agrícolas Puras Acotadas al Círculo LADM R=350m (Exclusión Total de Bosques)",
            "func_name": "algo_7_upas_samhq_meijering_v63base",
            "atlas_img": os.path.join(exp_dir, "dupla_07_upas_agricolas_puras.png")
        }
    ]
    
    # Checkpoint de respaldo
    checkpoint_path = os.path.join(exp_dir, f"checkpoint_moe_v3_{version_tag}.json")
    if os.path.exists(checkpoint_path):
        print(f"\n[Checkpoint Detectado]: Cargando deliberación previa desde {checkpoint_path}...")
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            cp_data = json.load(f)
        specialist_dict = cp_data["specialist_dict"]
        critic_audit = cp_data["critic_audit"]
        judge_verdict = cp_data["judge_verdict"]
    else:
        # Fase I: 7 Especialistas controlados por semáforo para respetar cuotas de API
        print("\n>>> Paso 2: Ejecutando Fase I: 7 Especialistas BARS con Duplas Individuales por Cuadrante...")
        sem = asyncio.Semaphore(2)
        async def run_with_sem(cfg_item):
            async with sem:
                res = await run_specialist_agent_v2(
                    client, primary_model, cfg_item, script_path, telemetry_probe,
                    ledger.get_open_tickets_for_agent(cfg_item["id"]), version_tag, exp_dir
                )
                await asyncio.sleep(3.0)
                return res

        tasks = [run_with_sem(cfg) for cfg in specialist_configs]
        specialist_results = await asyncio.gather(*tasks)
        specialist_dict = {res.get("agente_id"): res for res in specialist_results}
        
        # Fase II: Agente 8 - Crítico Adversarial
        print("\n>>> Paso 3: Ejecutando Fase II: Agente 8 - Crítico Adversarial y Auditor de Fronteras...")
        prompt_critic = build_adversarial_critic_prompt(specialist_dict, telemetry_probe)
        config_critic = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.HIGH),
            temperature=0.2,
            response_mime_type="application/json"
        )
        critic_parts = []
        ensamble_parts = prepare_individual_quadrant_dupla_parts(exp_dir, layer_num=8, max_dim=2200)
        if ensamble_parts:
            critic_parts.extend(ensamble_parts)
        critic_parts.append(prompt_critic)
        res_critic = call_gemini_with_backoff(client, primary_model, critic_parts, config_critic)
        raw_c = res_critic.text.strip()
        if raw_c.startswith("```json"): raw_c = raw_c[7:]
        if raw_c.endswith("```"): raw_c = raw_c[:-3]
        critic_audit = json.loads(raw_c.strip())
        print(f"[Crítico Adversarial] Calidad argumentativa: {critic_audit.get('calidad_argumentativa_tribunal')} | Conflictos detectados: {len(critic_audit.get('conflictos_inter_capa_detectados', []))}")
        
        # Fase III: Agente 9 - Juez Supremo LADM (Pareto Promotion Gate)
        print("\n>>> Paso 4: Ejecutando Fase III: Agente 9 - Juez Supremo y Puerta de Promoción de Pareto...")
        prompt_judge = build_supreme_judge_prompt_v2(specialist_dict, critic_audit, telemetry_probe, hist_summary, version_tag)
        res_judge = call_gemini_with_backoff(client, primary_model, [prompt_judge], config_critic)
        raw_j = res_judge.text.strip()
        if raw_j.startswith("```json"): raw_j = raw_j[7:]
        if raw_j.endswith("```"): raw_j = raw_j[:-3]
        judge_verdict = json.loads(raw_j.strip())
        score_global = judge_verdict.get("scorecard_global", {}).get("calificacion_global_v2", "N/A")
        cumple_pareto = judge_verdict.get("cumple_puerta_pareto", False)
        sentencia = judge_verdict.get("sentencia_tribunal", "N/A")
        print(f"[Juez Supremo LADM] Global: {score_global}/100 | Puerta de Pareto: {cumple_pareto} | Sentencia: {sentencia}")
        
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump({
                "specialist_dict": specialist_dict,
                "critic_audit": critic_audit,
                "judge_verdict": judge_verdict
            }, f, indent=2, ensure_ascii=False)
        print(f"[Checkpoint Guardado]: {checkpoint_path}")

    # Pausa de cortesía
    time.sleep(5)
    
    # 6. Fase IV: Agente 10 - Sintetizador de Parches
    print("\n>>> Paso 5: Ejecutando Fase IV: Agente 10 - Arquitecto Sintetizador de Parches...")
    config_patch = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.HIGH),
        temperature=0.2,
        response_mime_type="application/json"
    )
    prompt_patch = build_patch_synthesizer_prompt(judge_verdict, critic_audit, specialist_dict, script_path)
    res_patch = call_gemini_with_backoff(client, primary_model, [prompt_patch], config_patch)
    raw_p = res_patch.text.strip()
    if raw_p.startswith("```json"): raw_p = raw_p[7:]
    if raw_p.endswith("```"): raw_p = raw_p[:-3]
    patch_plan = json.loads(raw_p.strip())
    print(f"[Patch Synthesizer] Parches estructurados generados: {len(patch_plan.get('parches_propuestos', []))}")
    
    # 7. Consolidación y actualización del libro mayor
    full_audit = {
        "metadata": {
            "version": version_tag,
            "fecha": datetime.datetime.now().isoformat(),
            "arquitectura": "MoE-Kaizen v3.0 (Inspección Desacoplada a Altitud Virtual de 250m)",
            "telemetria_sonda": telemetry_probe["resumen_global"],
            "cuadrantes": telemetry_probe.get("cuadrantes", {})
        },
        "scorecard_7_especialistas": specialist_dict,
        "auditoria_adversarial_agente_8": critic_audit,
        "sentencia_tribunal_supremo": {
            primary_model: judge_verdict
        },
        "plan_parches_agente_10": patch_plan,
        "metrics_summary": telemetry_probe.get("metricas_reportadas", {})
    }
    
    out_json = os.path.join(exp_dir, f"auditoria_moe_kaizen_v3_{version_tag}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(full_audit, f, indent=2, ensure_ascii=False)
    print(f"\n[Auditoría v3 Guardada]: {out_json}")
    
    patch_json = os.path.join(exp_dir, f"kaizen_action_plan_v3_{version_tag}.json")
    with open(patch_json, "w", encoding="utf-8") as f:
        json.dump(patch_plan, f, indent=2, ensure_ascii=False)
    print(f"[Plan de Parches v3 Guardado]: {patch_json}")
    
    ledger.record_version_audit(version_tag, full_audit)
    return full_audit

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ejecutar Tribunal MoE-Kaizen v3.0")
    parser.add_argument("--exp_dir", type=str, default=r"K:\Mi unidad\CURSOS Y CAPACITACIÓN PERSONAL\MASTER-INTELIGENCIA ARTIFICIAL\Aula Master IA\02_Segundo_Cuatrimestre\09_TFM\06_Codigo_y_Experimentos\experimentos\v100.0.0")
    parser.add_argument("--version", type=str, default="v100.0.0")
    parser.add_argument("--model", type=str, default="gemini-4-argon")
    args = parser.parse_args()
    
    print(f"Iniciando ciclo MoE-Kaizen v3.0 sobre: {args.exp_dir} (Versión: {args.version})")
    asyncio.run(run_orchestrator_cycle(args.exp_dir, args.version, model_name=args.model))
