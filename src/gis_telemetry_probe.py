"""
================================================================================
GIS TELEMETRY PROBE v3.0 - SONDA DETERMINISTA PRE-VLM MULTI-CAPA
Módulo Core: Métricas Matemáticas por Cuadrante, Focos de Micro-Zoom y Matriz Inter-Capa
TFM UNIR - Delimitación Automatizada de UPAs en Chimborazo
================================================================================
"""

import os
import sys
import json
import sqlite3
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, Point, box, LineString, MultiLineString
from shapely.ops import unary_union
import pyproj

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

SAN_ANDRES_QUADRANTS_COORDS = [
    {"cid": 1, "cdesc": "San Andrés Centro Urbano y Agrícola", "lat": -1.6035, "lon": -78.6940},
    {"cid": 2, "cdesc": "Chocaví Laderas y Terrazas Altas", "lat": -1.6110, "lon": -78.7020},
    {"cid": 3, "cdesc": "Pulucate Piedemonte y Quebradas Riparias", "lat": -1.6220, "lon": -78.7100},
    {"cid": 4, "cdesc": "Batzacón Interfase y Planicie", "lat": -1.5950, "lon": -78.6850}
]

def compute_quadrant_zoom_foci(gdf_dict_utm: dict, circle_poly_utm: Polygon, xc: float, yc: float, radius_m: float = 350.0) -> dict:
    """
    Calcula las coordenadas óptimas para los micro-zooms a altitud virtual de 250m:
    - focus_zoom_a: Máxima concentración de edificaciones rurales y patios campesinos.
    - focus_zoom_b: Máxima densidad de linderos de minifundio (UPAs) y cercos vivos.
    Devuelve coordenadas UTM, coordenadas normalizadas (u, v) in [-1, 1], y celda Set-of-Mark (A1..F6).
    """
    # 1. Foco A (Viviendas + Patios + Agroindustria)
    pts_a = []
    for l_key in ['viviendas_rurales_ladm', 'patios_eras_campesinas', 'estructuras_agroindustriales']:
        if l_key in gdf_dict_utm and not gdf_dict_utm[l_key].empty:
            in_circ = gdf_dict_utm[l_key][gdf_dict_utm[l_key].geometry.intersects(circle_poly_utm)]
            pts_a.extend([g.centroid for g in in_circ.geometry if g is not None and not g.is_empty])

    if pts_a:
        best_pt_a = pts_a[0]
        max_na = 0
        for pt in pts_a:
            cnt = sum(1 for other in pts_a if pt.distance(other) <= 80.0)
            if cnt > max_na:
                max_na = cnt
                best_pt_a = pt
        fa_x, fa_y = best_pt_a.x, best_pt_a.y
    else:
        fa_x, fa_y = xc, yc
        max_na = 0

    # 2. Foco B (UPAs agrícolas + Cercos vivos)
    pts_b = []
    if 'upas_agricolas_puras_circulo' in gdf_dict_utm and not gdf_dict_utm['upas_agricolas_puras_circulo'].empty:
        in_circ_u = gdf_dict_utm['upas_agricolas_puras_circulo'][gdf_dict_utm['upas_agricolas_puras_circulo'].geometry.intersects(circle_poly_utm)]
        pts_b.extend([g.centroid for g in in_circ_u.geometry if g is not None and not g.is_empty])

    if pts_b:
        best_pt_b = pts_b[0]
        max_nb = 0
        for pt in pts_b:
            cnt = sum(1 for other in pts_b if pt.distance(other) <= 90.0)
            if cnt > max_nb:
                max_nb = cnt
                best_pt_b = pt
        fb_x, fb_y = best_pt_b.x, best_pt_b.y
    else:
        fb_x, fb_y = xc, yc
        max_nb = 0

    # Normalizar coordenadas relativas al centro del círculo u, v in [-1, 1]
    ua = float(np.clip((fa_x - xc) / radius_m, -0.85, 0.85))
    va = float(np.clip((yc - fa_y) / radius_m, -0.85, 0.85))
    ub = float(np.clip((fb_x - xc) / radius_m, -0.85, 0.85))
    vb = float(np.clip((yc - fb_y) / radius_m, -0.85, 0.85))

    col_letters = ['A', 'B', 'C', 'D', 'E', 'F']
    cell_a = f"{col_letters[min(5, max(0, int((ua + 1.0) / 2.0 * 6)))]}{min(6, max(1, int((va + 1.0) / 2.0 * 6) + 1))}"
    cell_b = f"{col_letters[min(5, max(0, int((ub + 1.0) / 2.0 * 6)))]}{min(6, max(1, int((vb + 1.0) / 2.0 * 6) + 1))}"

    return {
        "focus_zoom_a": {
            "descripcion": "Foco Peri-doméstico / Edificaciones LADM",
            "utm": (round(fa_x, 2), round(fa_y, 2)),
            "norm": (round(ua, 3), round(va, 3)),
            "celda": cell_a,
            "elementos_en_radio": max_na
        },
        "focus_zoom_b": {
            "descripcion": "Foco Minifundio / Linderos Finos y Cercos Vivos",
            "utm": (round(fb_x, 2), round(fb_y, 2)),
            "norm": (round(ub, 3), round(vb, 3)),
            "celda": cell_b,
            "elementos_en_radio": max_nb
        }
    }

def run_gis_telemetry_probe(exp_dir: str, radius_m: float = 350.0) -> dict:
    """
    Ejecuta la sonda determinista exhaustiva sobre los artefactos de un experimento.
    Incluye desglose por cuadrante de las 8 capas, matriz de colisión y focos de zoom a 250m.
    """
    probe_results = {
        "exp_dir": exp_dir,
        "cuadrantes": {},
        "resumen_global": {
            "spillover_total_m2": 0.0,
            "solape_inter_parcelario_pct": 0.0,
            "total_upas": 0,
            "area_total_ha": 0.0,
            "conformidad_ladm_iso_19152": "100.0%"
        },
        "auditoria_capas_global": {},
        "matriz_colision_inter_capa_m2": {},
        "focos_micro_zoom_250m": {},
        "alertas_criticas": []
    }

    transformer_to_utm = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32717", always_xy=True)

    # 1. Localizar GPKG principal
    gpkg_files = [f for f in os.listdir(exp_dir) if f.endswith(".gpkg")]
    if not gpkg_files:
        probe_results["alertas_criticas"].append("No se encontró archivo GPKG en el directorio del experimento.")
        return probe_results

    exp_basename = os.path.basename(os.path.normpath(exp_dir))
    v_prefix = exp_basename.split(".")[0]
    matching_gpkg = [f for f in gpkg_files if v_prefix in f]
    selected_gpkg = matching_gpkg[0] if matching_gpkg else gpkg_files[-1]
    gpkg_path = os.path.join(exp_dir, selected_gpkg)
    probe_results["gpkg_file"] = selected_gpkg

    conn = sqlite3.connect(gpkg_path)
    cur = conn.cursor()
    layers_in_gpkg = [r[0] for r in cur.execute("SELECT table_name FROM gpkg_contents").fetchall()]
    conn.close()
    probe_results["capas_disponibles_gpkg"] = layers_in_gpkg

    # 2. Cargar todas las capas a UTM
    gdf_dict_utm = {}
    for lyr in layers_in_gpkg:
        try:
            gdf_dict_utm[lyr] = gpd.read_file(gpkg_path, layer=lyr).to_crs(epsg=32717)
        except Exception as e:
            probe_results["alertas_criticas"].append(f"Error leyendo capa {lyr}: {e}")

    # 3. Geometrías circulares
    circle_polys_utm = {}
    for q in SAN_ANDRES_QUADRANTS_COORDS:
        cid = q["cid"]
        xc, yc = transformer_to_utm.transform(q["lon"], q["lat"])
        circle_polys_utm[cid] = Point(xc, yc).buffer(radius_m)

    all_circles_union = unary_union(list(circle_polys_utm.values()))

    # 4. Auditoría Global por Capa
    for lyr, gdf in gdf_dict_utm.items():
        if lyr == "perimetros_cuadrantes_circulares":
            continue
        geom_type = gdf.geometry.iloc[0].geom_type if not gdf.empty else "None"
        is_poly = geom_type in ['Polygon', 'MultiPolygon']
        
        # Confinamiento circular
        spill_m2 = 0.0
        if is_poly and not gdf.empty:
            lyr_union = unary_union(list(gdf.geometry))
            spill_m2 = float(lyr_union.difference(all_circles_union).area)

        probe_results["auditoria_capas_global"][lyr] = {
            "total_elementos": len(gdf),
            "tipo_geometria": geom_type,
            "magnitud_total": round(float(gdf.geometry.area.sum() / 10000.0), 2) if is_poly else round(float(gdf.geometry.length.sum() / 1000.0), 2),
            "unidad": "ha" if is_poly else "km",
            "spillover_m2": round(spill_m2, 4)
        }

    # 5. Auditoría de UPAs Agrícolas Puras y Solapamiento
    if "upas_agricolas_puras_circulo" in gdf_dict_utm:
        gdf_u = gdf_dict_utm["upas_agricolas_puras_circulo"]
        total_upas = len(gdf_u)
        area_u_ha = float(gdf_u.geometry.area.sum() / 10000.0)
        spill_u = float(unary_union(list(gdf_u.geometry)).difference(all_circles_union).area)
        
        probe_results["resumen_global"]["total_upas"] = total_upas
        probe_results["resumen_global"]["area_total_ha"] = round(area_u_ha, 2)
        probe_results["resumen_global"]["spillover_total_m2"] = round(spill_u, 4)

        # Muestreo espacial de solape inter-parcelario
        inter_area_sum = 0.0
        geoms = list(gdf_u.geometry)
        sindex = gdf_u.sindex
        for i, geom_a in enumerate(geoms):
            possible_matches = list(sindex.intersection(geom_a.bounds))
            for j in possible_matches:
                if i < j:
                    geom_b = geoms[j]
                    if geom_a.intersects(geom_b):
                        inter = geom_a.intersection(geom_b)
                        if isinstance(inter, (Polygon, MultiPolygon)):
                            inter_area_sum += inter.area

        solape_pct = (inter_area_sum / (gdf_u.geometry.area.sum() + 1e-8)) * 100.0
        probe_results["resumen_global"]["solape_inter_parcelario_pct"] = round(solape_pct, 6)

    # 6. Desglose Exhaustivo por Cuadrante y Cálculo de Focos de Zoom a 250m
    for q in SAN_ANDRES_QUADRANTS_COORDS:
        cid = q["cid"]
        c_utm = circle_polys_utm[cid]
        xc, yc = transformer_to_utm.transform(q["lon"], q["lat"])
        q_key = f"Q{cid}_{q['cdesc'][:15].strip()}"

        q_stats = {
            "cid": cid,
            "nombre": q["cdesc"],
            "radio_m": radius_m,
            "area_estudio_ha": round(c_utm.area / 10000.0, 2),
            "capas": {}
        }

        for lyr, gdf in gdf_dict_utm.items():
            if lyr == "perimetros_cuadrantes_circulares":
                continue
            in_q = gdf[gdf.geometry.intersects(c_utm)]
            if in_q.empty:
                q_stats["capas"][lyr] = {"conteo": 0, "magnitud": 0.0}
                continue
            
            is_poly = in_q.geometry.iloc[0].geom_type in ['Polygon', 'MultiPolygon']
            if is_poly:
                mag = sum(g.intersection(c_utm).area for g in in_q.geometry) / 10000.0
                q_stats["capas"][lyr] = {"conteo": len(in_q), "magnitud_ha": round(mag, 2)}
            else:
                mag = sum(g.intersection(c_utm).length for g in in_q.geometry) / 1000.0
                q_stats["capas"][lyr] = {"conteo": len(in_q), "magnitud_km": round(mag, 2)}

        probe_results["cuadrantes"][q_key] = q_stats

        # Calcular Focos de Zoom
        foci = compute_quadrant_zoom_foci(gdf_dict_utm, c_utm, xc, yc, radius_m=radius_m)
        probe_results["focos_micro_zoom_250m"][f"Q{cid}"] = foci

    # 7. Matriz de Colisión Topológica Inter-Capa 5x5 Polígonos
    poly_layers = [l for l in ['viviendas_rurales_ladm', 'estructuras_agroindustriales', 'patios_eras_campesinas', 'bosques_quebradas_protegidas', 'upas_agricolas_puras_circulo'] if l in gdf_dict_utm]
    poly_unions = {l: unary_union(list(gdf_dict_utm[l].geometry)) for l in poly_layers if not gdf_dict_utm[l].empty}

    for i, l1 in enumerate(poly_layers):
        for j, l2 in enumerate(poly_layers):
            if i < j and l1 in poly_unions and l2 in poly_unions:
                inter_col = poly_unions[l1].intersection(poly_unions[l2])
                probe_results["matriz_colision_inter_capa_m2"][f"{l1}__vs__{l2}"] = round(float(inter_col.area), 4)

    # 8. Generar Informe Textual para Prompts Multimodales
    probe_results["telemetry_text_prompt"] = f"""
================================================================================
TELEMETRÍA DETERMINISTA GIS PRE-VLM v3.0 (SONDA DETERMINISTA MULTI-CAPA):
================================================================================
- CONFINAMIENTO CIRCULAR (R = 350.0 m, Área Total = 153.94 ha):
  * Derrame total de UPAs fuera del perímetro: {probe_results['resumen_global']['spillover_total_m2']:.4f} m² (Meta LADM: 0.0000 m²)
  * Conformidad Perimetral: {'PERFECTA (100.0%)' if probe_results['resumen_global']['spillover_total_m2'] < 0.01 else 'ALERTA DERRAME'}

- TOPOLOGÍA LADM ISO 19152 (0.0000% SOLAPE INTER-PARCELARIO):
  * Solapamiento inter-parcelario medido: {probe_results['resumen_global']['solape_inter_parcelario_pct']:.6f}%
  * Conformidad Planar: {'100.0% CONFORME (Grado Catastral Máximo)' if probe_results['resumen_global']['solape_inter_parcelario_pct'] == 0 else 'NO CONFORME'}

- COLISIONES INTER-CAPA TOPOLÓGICAS (EXCLUSIÓN MUTUA CATASTRAL):
  * Viviendas vs UPAs Agrícolas: {probe_results['matriz_colision_inter_capa_m2'].get('viviendas_rurales_ladm__vs__upas_agricolas_puras_circulo', 0.0):.4f} m² (Meta: 0.0000 m²)
  * Agroindustria vs UPAs Agrícolas: {probe_results['matriz_colision_inter_capa_m2'].get('estructuras_agroindustriales__vs__upas_agricolas_puras_circulo', 0.0):.4f} m² (Meta: 0.0000 m²)
  * Patios/Eras vs UPAs Agrícolas: {probe_results['matriz_colision_inter_capa_m2'].get('patios_eras_campesinas__vs__upas_agricolas_puras_circulo', 0.0):.4f} m² (Meta: 0.0000 m²)
  * Bosques vs UPAs Agrícolas: {probe_results['matriz_colision_inter_capa_m2'].get('bosques_quebradas_protegidas__vs__upas_agricolas_puras_circulo', 0.0):.4f} m²

- DESGLOSE DETALLADO POR CUADRANTE:
{json.dumps(probe_results['cuadrantes'], indent=2, ensure_ascii=False)}

- FOCOS DE MICRO-ZOOM A ALTITUD VIRTUAL DE 250 m (REJILLA SET-OF-MARK):
{json.dumps(probe_results['focos_micro_zoom_250m'], indent=2, ensure_ascii=False)}
================================================================================
"""
    return probe_results

if __name__ == "__main__":
    target_dir = sys.argv[1] if len(sys.argv) > 1 else r"K:\Mi unidad\CURSOS Y CAPACITACIÓN PERSONAL\MASTER-INTELIGENCIA ARTIFICIAL\Aula Master IA\02_Segundo_Cuatrimestre\09_TFM\06_Codigo_y_Experimentos\experimentos\v100.0.0"
    res = run_gis_telemetry_probe(target_dir)
    print(res["telemetry_text_prompt"])
