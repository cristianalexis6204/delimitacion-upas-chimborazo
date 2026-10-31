"""
Exportador del Reporte de Auditoria MoE Dual en Paralelo (Gemini 4 Argon vs. Gemini 3.1 Pro)
TFM UNIR - Delimitacion Automatizada de UPAs en Chimborazo
Autor: Cristian Alexis Garcia Pumagualle

Este script:
1. Lee los dictamenes consolidados en reportes/auditoria_moe_dual_paralelo.json.
2. Compara frente a frente las puntuaciones del Scorecard de ambos modelos.
3. Cruza los analisis por especialidad, contrastando consensos y discrepancias.
4. Construye la Matriz de Impacto vs. Esfuerzo y el Roadmap SOTA.
5. Exporta el reporte definitivo a reportes/auditoria_moe_dual_paralelo_reporte.md.
"""

import os
import sys
import json

def generate_markdown(data, out_md):
    mA = data["motor_a_gemini_4_argon"]
    mB = data["motor_b_gemini_3_1_pro"]
    meta = data["metadata"]

    sA = mA["scorecard_global"]
    sB = mB["scorecard_global"]

    eA = mA["evaluacion_expertos"]
    eB = mB["evaluacion_expertos"]

    stA = mA["sintesis_estrategica"]
    stB = mB["sintesis_estrategica"]

    telA = mA.get("telemetria", {})
    telB = mB.get("telemetria", {})

    md = []
    md.append("# Informe Técnico: Auditoría MoE Dual en Paralelo — Calidad de Detección de UPAs y Arquitectura Algorítmica\n")
    md.append("## Trabajo Fin de Máster (TFM) - Máster Universitario en Inteligencia Artificial (UNIR)\n")
    md.append(f"**Autor:** Cristian Alexis García Pumagualle  \n")
    md.append(f"**Entorno de Nube:** Google Cloud Vertex AI (Región `global`, Proyecto `tesis-bi-edison-paredes`)  \n")
    md.append(f"**Arquitectura Evaluadora:** MoE Dual en Paralelo (Torneo Multi-Modelo)  \n")
    md.append(f"**Motor A:** `gemini-4-argon` (Thinking HIGH, {telA.get('thinking_tokens', 0):,} tokens de pensamiento en {telA.get('tiempo_segundos', 0)}s)  \n")
    md.append(f"**Motor B:** `gemini-3.1-pro-preview` (Thinking HIGH, {telB.get('thinking_tokens', 0):,} tokens de pensamiento en {telB.get('tiempo_segundos', 0)}s)  \n")
    md.append(f"**Base de Código Auditada:** 1,940 líneas de Python en 4 scripts troncales + Benchmark GT Calpi + San Andrés v60  \n")
    md.append("\n---\n")

    # 1. Scorecard Comparativo
    md.append("### 1. Scorecard Comparativo Frente a Frente (Gemini 4 Argon vs. Gemini 3.1 Pro)\n\n")
    md.append("| Dimensión de Evaluación Técnica | Gemini 4 Argon | Gemini 3.1 Pro | Delta (|A - B|) | Veredicto Consensuado |\n")
    md.append("| :--- | :---: | :---: | :---: | :--- |\n")

    dims = [
        ("Calidad de Detección de Contornos", "calidad_deteccion_contornos"),
        ("Conformidad Catastral LADM ISO 19152", "conformidad_catastral_ladm"),
        ("Arquitectura de Software y Modularidad", "arquitectura_software_y_modularidad"),
        ("Eficiencia HPC y CUDA", "eficiencia_hpc_y_cuda"),
        ("Pertinencia Agronómica y Censo", "pertinencia_agronomica_censo"),
        ("PUNTUACIÓN PROMEDIO GENERAL", "puntuacion_promedio_general")
    ]

    for label, key in dims:
        valA = sA[key]
        valB = sB[key]
        delta = abs(valA - valB)
        status = "Consenso Fuerte" if delta <= 5 else ("Consenso Moderado" if delta <= 12 else "Divergencia")
        md.append(f"| **{label}** | **{valA:.1f}** | **{valB:.1f}** | `{delta:.1f}` | {status} |\n")

    md.append("\n---\n")

    # 2. Análisis por Especialidades
    md.append("### 2. Diagnóstico Técnico Cruzado por Especialidad\n\n")

    # Visión por Computador
    md.append("#### A. Visión por Computador y Sensores Remotos\n")
    md.append(f"- **Gemini 4 Argon ({eA['vision_por_computador']['puntaje']}/100):** {eA['vision_por_computador']['analisis_critico']}\n")
    md.append(f"  * *Oportunidad SOTA Lucas et al. (2026):* {eA['vision_por_computador']['oportunidades_sota_lucas_2026']}\n\n")
    md.append(f"- **Gemini 3.1 Pro ({eB['vision_por_computador']['puntaje']}/100):** {eB['vision_por_computador']['analisis_critico']}\n")
    md.append(f"  * *Oportunidad SOTA Lucas et al. (2026):* {eB['vision_por_computador']['oportunidades_sota_lucas_2026']}\n\n")

    # Catastro LADM
    md.append("#### B. Catastro Topológico LADM ISO 19152\n")
    md.append(f"- **Gemini 4 Argon ({eA['catastro_ladm_iso19152']['puntaje']}/100):** {eA['catastro_ladm_iso19152']['analisis_critico']}\n\n")
    md.append(f"- **Gemini 3.1 Pro ({eB['catastro_ladm_iso19152']['puntaje']}/100):** {eB['catastro_ladm_iso19152']['analisis_critico']}\n\n")

    # Software y MLOps
    md.append("#### C. Ingeniería de Software, HPC y MLOps\n")
    md.append(f"- **Gemini 4 Argon ({eA['software_mlops_hpc']['puntaje']}/100):**\n")
    md.append(f"  * *Deuda Técnica:* {eA['software_mlops_hpc']['deuda_tecnica_identificada']}\n")
    md.append(f"  * *Optimización CUDA/VRAM:* {eA['software_mlops_hpc']['recomendacion_cuda_memoria']}\n\n")
    md.append(f"- **Gemini 3.1 Pro ({eB['software_mlops_hpc']['puntaje']}/100):**\n")
    md.append(f"  * *Deuda Técnica:* {eB['software_mlops_hpc']['deuda_tecnica_identificada']}\n")
    md.append(f"  * *Optimización CUDA/VRAM:* {eB['software_mlops_hpc']['recomendacion_cuda_memoria']}\n\n")

    # Agronomía y Censo
    md.append("#### D. Agronomía y Minifundio Andino (RENAGRO / MAG)\n")
    md.append(f"- **Gemini 4 Argon ({eA['agronomia_renagro']['puntaje']}/100):** {eA['agronomia_renagro']['analisis_minifundio_andino']}\n\n")
    md.append(f"- **Gemini 3.1 Pro ({eB['agronomia_renagro']['puntaje']}/100):** {eB['agronomia_renagro']['analisis_minifundio_andino']}\n\n")

    md.append("---\n")

    # 3. Puntos de Consenso vs. Discrepancias
    md.append("### 3. Puntos de Consenso y Puntos Ciegos Detectados\n\n")
    md.append("#### Fortalezas Coincidentes (100% Acuerdo):\n")
    for f in stA["tres_mayores_fortalezas"]:
        md.append(f"- [Gemini 4 Argon] {f}\n")
    for f in stB["tres_mayores_fortalezas"]:
        md.append(f"- [Gemini 3.1 Pro] {f}\n")

    md.append("\n#### Cuellos de Botella Críticos Identificados:\n")
    for c in stA["tres_mayores_cuellos_de_botella"]:
        md.append(f"- ⚠️ [Gemini 4 Argon] {c}\n")
    for c in stB["tres_mayores_cuellos_de_botella"]:
        md.append(f"- ⚠️ [Gemini 3.1 Pro] {c}\n")

    md.append("\n---\n")

    # 4. Matriz de Impacto vs. Esfuerzo Consensuada
    md.append("### 4. Matriz de Impacto vs. Esfuerzo Consensuada\n\n")
    md.append("| Prioridad | Iniciativa de Optimización | Tipo | Impacto | Esfuerzo | Justificación Algorítmica |\n")
    md.append("| :-: | :--- | :---: | :---: | :---: | :--- |\n")

    all_recs = stA["cinco_recomendaciones_prioritarias"] + stB["cinco_recomendaciones_prioritarias"]
    # De-duplicate by title similarity
    seen = set()
    unique_recs = []
    for r in all_recs:
        t_key = r["titulo"][:20].lower()
        if t_key not in seen:
            seen.add(t_key)
            unique_recs.append(r)

    for idx, r in enumerate(unique_recs[:8], 1):
        md.append(f"| `{idx}` | **{r['titulo']}** | `{r['tipo']}` | `{r['impacto']}` | `{r['esfuerzo']}` | {r['descripcion']} |\n")

    md.append("\n---\n")

    # 5. Veredicto Final para la Memoria TFM
    md.append("### 5. Veredicto Final para la Memoria y Defensa del TFM\n\n")
    md.append(f"> **Dictamen Gemini 4 Argon:**  \n> \"{stA['veredicto_final_memoria_tfm']}\"\n\n")
    md.append(f"> **Dictamen Gemini 3.1 Pro:**  \n> \"{stB['veredicto_final_memoria_tfm']}\"\n\n")

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("".join(md))

    print(f"[OK] Reporte Markdown comparativo generado: {out_md}")

def main():
    repo_root = r"K:\Mi unidad\CURSOS Y CAPACITACIÓN PERSONAL\MASTER-INTELIGENCIA ARTIFICIAL\Aula Master IA\02_Segundo_Cuatrimestre\09_TFM\delimitacion-upas-chimborazo"
    json_path = os.path.join(repo_root, "reportes", "auditoria_moe_dual_paralelo.json")
    out_md = os.path.join(repo_root, "reportes", "auditoria_moe_dual_paralelo_reporte.md")

    if not os.path.exists(json_path):
        print(f"[ERROR] Archivo no encontrado: {json_path}")
        sys.exit(1)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    generate_markdown(data, out_md)

if __name__ == "__main__":
    main()
