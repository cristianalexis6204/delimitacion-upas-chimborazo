"""
PIPELINE MASTER v58.0.0 - DELIMITACIÓN SOTA POR META SAM-2.1 / SAM-HQ,
CRESTAS DE MEIJERING MULTIESCALA Y CONFORMIDAD CATASTRAL LADM ISO 19152 (TFM UNIR)
Autor: Cristian Alexis García Pumagualle
Director: Fernando Antonio Rufo Jiménez

Innovaciones Críticas de la Versión v58.0.0:
1. Adopción de Arquitecturas SOTA en Hugging Face:
   - Meta SAM-2.1 (Hierarchical Vision Transformer - Hiera) + SysCV SAM-HQ (High-Quality Token).
   - Inferencia acelerada en GPU Nvidia A100 (Google Colab Pro).
2. Erradicación Total de Falsos Positivos de Viviendas en Parcelas Agrícolas:
   - Disambiguación espectral CIELAB (canal a* > 134) entre suelo andino pardo y teja de barro cocido.
   - Verificación de Sombra Solar Relativa 3D (Delta-L >= 14.0 DN en cuadrante suroeste).
   - Homogeneidad textural interna (std <= 38.0) y rectangularidad real >= 0.62.
3. Segmentación Híbrida Meta SAM-2.1/SAM-HQ + Crestas Multiescala de Meijering:
   - Inferencia semántica y Mosaico Planar Competitivo Argmax (0.00% solapes artificiales).
4. Conformidad Catastral LADM ISO 19152:
   - Regularización geométrica de 4 a 8 vértices por parcela, 0 invasión vial y 0 solapes habitacionales.
5. Cobertura Provincial en 4 Escenarios de Chimborazo:
   - Calpi, Licán, Colta y Guano con entregables completos GeoPackage OGC, KML Wireframe y Visor Leaflet.
"""

import os
import sys
import math
import json
import datetime
import shutil
import numpy as np
import cv2
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, Point, MultiPolygon
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from download_sam2_and_samhq_weights import ensure_all_models
from sam2_ridge_hybrid_segmenter import Sam2RidgeHybridSegmenter
from topological_boundary_reconciler import TopologicalBoundaryReconciler
from cadastral_metrics_evaluator import CadastralMetricsEvaluator
from rural_road_network_extractor import RuralRoadNetworkExtractor
from rural_building_extractor import RuralBuildingExtractor
from interactive_map_builder import InteractiveMapBuilder
from kml_multilayer_exporter import KMLMultilayerExporter
from run_comparative_benchmark_7_models import run_benchmark_7_models

def latlon_to_tile(lat, lon, zoom):
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + (1.0 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return (xtile, ytile)

def tile_to_latlon(xtile, ytile, zoom):
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)

def fetch_tile_imagery(xtile, ytile, zoom):
    import urllib.request
    url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{ytile}/{xtile}"
    headers = {'User-Agent': 'Mozilla/5.0 (TFM-UNIR-v58-MasterPipeline)'}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=15) as response:
        arr = np.asarray(bytearray(response.read()), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def get_scene_patch(lat, lon, zoom=18, radius=2, crop_size=400):
    xtile, ytile = latlon_to_tile(lat, lon, zoom)
    tiles = []
    for dy in range(-radius, radius + 1):
        row = []
        for dx in range(-radius, radius + 1):
            tile_img = fetch_tile_imagery(xtile + dx, ytile + dy, zoom)
            row.append(tile_img)
        tiles.append(np.hstack(row))
    full_mosaic = np.vstack(tiles)

    top_lat, left_lon = tile_to_latlon(xtile - radius, ytile - radius, zoom)
    bot_lat, right_lon = tile_to_latlon(xtile + radius + 1, ytile + radius + 1, zoom)

    h, w, _ = full_mosaic.shape
    px = int((lon - left_lon) / (right_lon - left_lon) * w)
    py = int((top_lat - lat) / (top_lat - bot_lat) * h)

    min_x, max_x = max(0, px - crop_size), min(w, px + crop_size)
    min_y, max_y = max(0, py - crop_size), min(h, py + crop_size)

    crop_rgb = full_mosaic[min_y:max_y, min_x:max_x]

    c_left_lon = left_lon + (min_x / w) * (right_lon - left_lon)
    c_right_lon = left_lon + (max_x / w) * (right_lon - left_lon)
    c_top_lat = top_lat - (min_y / h) * (top_lat - bot_lat)
    c_bot_lat = top_lat - (max_y / h) * (top_lat - bot_lat)

    return crop_rgb, (c_left_lon, c_right_lon, c_top_lat, c_bot_lat)

def run_v58_pipeline():
    version = "v58.0.0"
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outputs_dir = os.path.join(base_dir, "outputs")
    reportes_fig_dir = os.path.join(base_dir, "..", "06_Codigo_y_Experimentos", "reportes", "figuras")
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(reportes_fig_dir, exist_ok=True)

    print("=" * 80)
    print(f"   EJECUTANDO PIPELINE MASTER {version} - PROVINCIA DE CHIMBORAZO")
    print(f"   SOTA META SAM-2.1 / SAM-HQ + HESSIAN-MEIJERING + CONFORMIDAD LADM")
    print("=" * 80)

    # 1. Asegurar pesos de los modelos SOTA
    print("\n[PASO 1]: Verificación y Descarga de Checkpoints SOTA...")
    ensure_all_models()

    # 2. Inicializar Módulos y Extractores
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[PASO 2]: Inicializando Módulos en {device.upper()}...")
    road_extractor = RuralRoadNetworkExtractor()
    building_extractor = RuralBuildingExtractor()
    hybrid_segmenter = Sam2RidgeHybridSegmenter(model_variant="sam2.1", device=device)
    topological_reconciler = TopologicalBoundaryReconciler()
    metrics_evaluator = CadastralMetricsEvaluator(pixel_size_m=0.30)

    # 3. Escenarios de Estudio en Chimborazo
    escenarios = [
        {'id': 1, 'name': '1_Cultivos_Ladera_Calpi', 'lat': -1.6350, 'lon': -78.7850, 'zoom': 18, 'desc': 'Minifundio andino de alta pendiente'},
        {'id': 2, 'name': '2_Caserio_Rural_Lican', 'lat': -1.6550, 'lon': -78.7150, 'zoom': 18, 'desc': 'Interfase periurbana-rural densa'},
        {'id': 3, 'name': '3_Valle_Agricola_Colta', 'lat': -1.7450, 'lon': -78.7550, 'zoom': 18, 'desc': 'Intensa parcelación hortícola lacustre'},
        {'id': 4, 'name': '4_Valle_Hortifruticola_Guano', 'lat': -1.6150, 'lon': -78.6350, 'zoom': 18, 'desc': 'Valle aluvial con cercas vivas y frutales'}
    ]

    all_polygons_gdf = []
    all_roads_gdf = []
    all_buildings_gdf = []
    all_vertices_count = []
    total_buildings_detected_all = 0

    for esc in escenarios:
        esc_id = esc['id']
        esc_name = esc['name']
        print(f"\n>>> Procesando Escenario [{esc_id}/4]: {esc_name} ({esc['desc']})...")

        # Ingesta satelital
        crop_rgb, bbox = get_scene_patch(esc['lat'], esc['lon'], zoom=esc['zoom'], radius=2, crop_size=400)
        h_img, w_img, _ = crop_rgb.shape

        # Preprocesamiento espectral
        lab = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2LAB)
        l_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
        r_f = crop_rgb[:, :, 0].astype(float)
        g_f = crop_rgb[:, :, 1].astype(float)
        b_f = crop_rgb[:, :, 2].astype(float)
        exg = (2.0 * g_f - r_f - b_f) / 255.0

        # Restricciones viales y edificaciones LADM
        road_res = road_extractor.extract_roads(crop_rgb, l_clahe, exg, None, bbox, esc_name=esc_name)
        bldg_res = building_extractor.extract_buildings(crop_rgb, l_clahe, exg, bbox)
        road_mask = road_res['road_mask_px']
        bldg_mask = bldg_res['building_mask_px']

        bldgs = bldg_res['buildings']
        total_buildings_detected_all += len(bldgs)
        all_buildings_gdf.extend(bldgs)
        if road_res['roads_gdf'] is not None and len(road_res['roads_gdf']) > 0:
            all_roads_gdf.extend(road_res['roads_gdf'].to_dict('records'))

        # Delineación Híbrida SAM-2.1/SAM-HQ + Meijering + LADM
        parcels_e, energy = hybrid_segmenter.delineate_parcels(
            crop_rgb=crop_rgb,
            l_clahe=l_clahe,
            exg=exg,
            bbox=bbox,
            road_mask=road_mask,
            bldg_mask=bldg_mask,
            esc_id=esc_id,
            esc_name=esc_name
        )

        for p in parcels_e:
            p['escenario'] = esc_name
            all_vertices_count.append(p['n_vertices'])
            all_polygons_gdf.append(p)

        print(f"    [OK] {len(parcels_e)} UPAs regularizadas LADM en {esc_name}.")

    # 4. Consolidación de GeoDataFrame Provincial
    print(f"\n[PASO 4]: Consolidación Vectorial de {len(all_polygons_gdf)} UPAs en Chimborazo...")
    gdf_upas = gpd.GeoDataFrame(all_polygons_gdf, crs="EPSG:4326")
    gdf_upas['upa_id'] = [f"UPA_{i+1:04d}" for i in range(len(gdf_upas))]

    # Guardar GeoPackage y KML
    gpkg_path = os.path.join(outputs_dir, f"upas_chimborazo_{version.replace('.', '_')}.gpkg")
    gdf_upas.to_file(gpkg_path, driver="GPKG", layer="upas_cadastrales_ladm")
    print(f"[GeoPackage OGC Oficial]: {gpkg_path}")

    kml_path = os.path.join(outputs_dir, f"upas_chimborazo_{version.replace('.', '_')}.kml")
    kml_exporter = KMLMultilayerExporter()
    kml_exporter.export(gdf_upas, all_roads_gdf, all_buildings_gdf, kml_path, title=f"UPAs Chimborazo {version}")
    print(f"[KML Wireframe Oficial]: {kml_path}")

    # Visor Interactivo Leaflet
    map_path = os.path.join(outputs_dir, f"mapa_interactivo_{version.replace('.', '_')}.html")
    map_builder = InteractiveMapBuilder()
    map_builder.build_map(gdf_upas, all_roads_gdf, all_buildings_gdf, map_path, title=f"Catastro Rural Chimborazo {version}")
    print(f"[Visor Interactivo Leaflet]: {map_path}")

    # 5. Ejecución del Benchmark Comparativo de 7 Modelos
    print("\n[PASO 5]: Ejecutando Benchmark Riguroso de 7 Modelos contra Ground Truth...")
    df_benchmark = run_benchmark_7_models()

    # Copiar gráficos a la carpeta de reportes de la memoria LaTeX
    for f in [f"benchmark_comparativo_7_modelos_{version.replace('.', '_')}.png",
             f"benchmark_comparativo_7_modelos_{version.replace('.', '_')}.csv"]:
        src_f = os.path.join(outputs_dir, f)
        dst_f = os.path.join(reportes_fig_dir, f)
        if os.path.exists(src_f):
            shutil.copy(src_f, dst_f)

    # 6. Resumen de Métricas JSON
    exact_total_ha = float(gdf_upas['area_ha'].sum())
    escenarios_info = {}
    for esc in escenarios:
        polys_e = [p for p in all_polygons_gdf if p['escenario'] == esc['name']]
        escenarios_info[esc['name']] = {
            'upas': len(polys_e),
            'ha': round(sum(p['area_ha'] for p in polys_e), 2),
            'vert': round(float(np.mean([p['n_vertices'] for p in polys_e])), 1) if polys_e else 7.0
        }

    # Extraer métricas de M7 (Propuesto v58) del benchmark
    m7_row = df_benchmark[df_benchmark['model_name'].str.contains('v58')].iloc[0]
    
    # Monte Carlo de Sensibilidad GPS
    mc_results = metrics_evaluator.run_monte_carlo_gps_sensitivity(baseline_iou=float(m7_row['miou_pct'])/100.0, sample_upas=20)
    gps_plot_path = os.path.join(outputs_dir, f"curva_sensibilidad_gps_{version.replace('.', '_')}.png")
    metrics_evaluator.generate_gps_sensitivity_plot(gps_plot_path, mc_results, version=version)
    shutil.copy(gps_plot_path, os.path.join(reportes_fig_dir, f"curva_sensibilidad_gps_{version.replace('.', '_')}.png"))

    metrics_summary = {
        'version': version,
        'model_architecture': "Meta SAM-2.1 Hiera-Large + SysCV SAM-HQ + Hessian-Meijering Multiscale",
        'total_polygons': len(gdf_upas),
        'total_buildings': total_buildings_detected_all,
        'total_area_ha': round(exact_total_ha, 2),
        'mean_vertices': round(float(np.mean(all_vertices_count)), 2),
        'duplicated_polygons': int(gdf_upas.geometry.duplicated().sum()),
        'miou_real_ground_truth': float(m7_row['miou_pct']) / 100.0,
        'dice_real_ground_truth': float(m7_row['dice_pct']) / 100.0,
        'boundary_f1_score_real': float(m7_row['bf1_pct']) / 100.0,
        'precision_real': float(m7_row['precision_pct']) / 100.0,
        'recall_real': float(m7_row['recall_pct']) / 100.0,
        'ladm_compliance_pct': float(m7_row['ladm_pct']),
        'overlap_pct': float(m7_row['solape_pct']),
        'escenarios_info': escenarios_info,
        'monte_carlo_gps_sensitivity': mc_results,
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    metrics_path = os.path.join(outputs_dir, f"metrics_summary_{version.replace('.', '_')}.json")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_summary, f, indent=2, ensure_ascii=False)
    print(f"\n[Métricas Oficiales JSON]: {metrics_path}")

    print("\n" + "=" * 80)
    print(f" PIPELINE MASTER {version} FINALIZADO SATISFACTORIAMENTE")
    print(f" Total UPAs: {len(gdf_upas)} | Superficie: {exact_total_ha:.2f} ha | mIoU SOTA: {m7_row['miou_pct']}% | LADM: {m7_row['ladm_pct']}%")
    print("=" * 80)
    return metrics_summary

if __name__ == "__main__":
    run_v58_pipeline()
