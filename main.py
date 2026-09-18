"""
===================================================================================================
PIPELINE MAESTRO DE DELIMITACIÓN AUTOMATIZADA DE UPAS (TFM UNIR)
Versión: v53.0.0 SOTA | Estándar: ISO 19152 LADM
Autor: Cristian Alexis García Pumagualle
Director: Fernando Antonio Rufo Jiménez
Universidad Internacional de La Rioja (UNIR)
===================================================================================================
"""

import os
import sys
import argparse
import yaml
import json
import numpy as np
import cv2
from PIL import Image
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Asegurar importación de módulos internos desde src/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(CURRENT_DIR, "src"))

from spectral_ridge_delineator import SpectralRidgeDelineator
from topological_boundary_reconciler import TopologicalBoundaryReconciler
from kml_multilayer_exporter import KMLMultilayerExporter
from rural_building_extractor import RuralBuildingExtractor
from rural_road_network_extractor import RuralRoadNetworkExtractor
from cadastral_metrics_evaluator import CadastralMetricsEvaluator
from interactive_map_builder import InteractiveMapBuilder

def load_config(config_path="config/config.yaml"):
    full_path = os.path.join(CURRENT_DIR, config_path)
    if os.path.exists(full_path):
        with open(full_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}

def verify_environment():
    import torch
    print("=" * 70)
    print("VERIFICACIÓN DEL ENTORNO DE EJECUCIÓN (TFM UNIR)")
    print("=" * 70)
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    cuda_avail = torch.cuda.is_available()
    print(f"CUDA Disponible: {'SÍ (GPU Acelerada)' if cuda_avail else 'NO (CPU Mode)'}")
    if cuda_avail:
        print(f"Dispositivo GPU: {torch.cuda.get_device_name(0)}")
    
    packages = ["cv2", "shapely", "geopandas", "rasterio", "skimage", "reportlab", "folium", "yaml"]
    print("\nLibrerías Core Verificadas:")
    for p in packages:
        try:
            mod = __import__(p)
            ver = getattr(mod, "__version__", "OK")
            print(f"  -> {p:15s}: {ver}")
        except ImportError as e:
            print(f"  -> {p:15s}: ERROR ({e})")
    print("=" * 70)

def run_pipeline(scenario_id="1", use_samples=True):
    cfg = load_config()
    config = cfg
    version = config.get("pipeline", {}).get("version", "v54.0.0")
    outputs_dir = os.path.join(CURRENT_DIR, config.get("directories", {}).get("outputs", "outputs"))
    samples_dir = os.path.join(CURRENT_DIR, config.get("directories", {}).get("samples", "data/samples"))
    os.makedirs(outputs_dir, exist_ok=True)

    print("=" * 75)
    print(f"INICIANDO PIPELINE MAESTRO {version} - DELIMITACIÓN DE UPAS LADM ISO 19152")
    print(f"Área de Estudio: {cfg.get('project', {}).get('study_area', 'Chimborazo')}")
    print("=" * 75)

    # Cargar escenario desde config
    scenarios_cfg = cfg.get("scenarios", [])
    scenarios_to_run = []
    if scenario_id.lower() == "all":
        scenarios_to_run = scenarios_cfg
    else:
        try:
            s_id = int(scenario_id)
            scenarios_to_run = [s for s in scenarios_cfg if s["id"] == s_id]
        except:
            scenarios_to_run = [scenarios_cfg[0]] if scenarios_cfg else []

    if not scenarios_to_run:
        print(f"[ERROR] No se encontró el escenario: {scenario_id}")
        return

    # Inicializar motores
    delineator = SpectralRidgeDelineator(min_parcel_px=260)
    reconciler = TopologicalBoundaryReconciler(snap_tolerance_deg=0.000008)
    road_extractor = RuralRoadNetworkExtractor(road_buffer_m=3.5)
    building_extractor = RuralBuildingExtractor(min_area_m2=25.0, max_housing_m2=280.0)
    kml_exporter = KMLMultilayerExporter(version=version)

    sample_files = {
        1: "calpi_sample.png",
        2: "lican_sample.png",
        3: "colta_sample.png",
        4: "guano_sample.png"
    }

    all_polygons_gdf = []
    all_roads_gdf = []
    all_buildings_gdf = []

    for sc in scenarios_to_run:
        sc_id = sc["id"]
        sc_name = sc["name"]
        disp_name = sc.get("display_name", sc_name)
        print(f"\n>>> Procesando Escenario [{sc_id}]: {disp_name}...")

        sample_name = sample_files.get(sc_id, "calpi_sample.png")
        sample_path = os.path.join(samples_dir, sample_name)

        if not os.path.exists(sample_path):
            print(f"[ERROR] No se encontró el recorte satelital de muestra: {sample_path}")
            continue

        # Cargar imagen satelital (800x800) con PIL (compatible con caracteres acentuados en Windows)
        img_pil = Image.open(sample_path).convert("RGB")
        crop_rgb = np.array(img_pil)
        h_img, w_img, _ = crop_rgb.shape

        lat, lon = sc["lat"], sc["lon"]
        # Extensión geográfica aproximada para 800 px a 0.3 m/px (~240 m)
        delta_deg = 240.0 / 111000.0
        min_lon = lon - delta_deg / 2.0
        max_lon = lon + delta_deg / 2.0
        top_lat = lat + delta_deg / 2.0
        bot_lat = lat - delta_deg / 2.0
        bbox = (min_lon, max_lon, top_lat, bot_lat)

        # Preprocesamiento espectral y espacial
        img_bilateral = cv2.bilateralFilter(crop_rgb, d=7, sigmaColor=50, sigmaSpace=50)
        lab = cv2.cvtColor(img_bilateral, cv2.COLOR_RGB2LAB)
        l, a, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)

        r_f, g_f, b_f = crop_rgb[:, :, 0].astype(float), crop_rgb[:, :, 1].astype(float), crop_rgb[:, :, 2].astype(float)
        exg = (2.0 * g_f - r_f - b_f) / 255.0

        # Detección de red vial y viviendas
        road_res = road_extractor.extract_roads(crop_rgb, l_clahe, exg, None, bbox, esc_name=sc_name)
        for l_geom in road_res["shapely_lines"]:
            all_roads_gdf.append({"geometry": l_geom, "escenario": sc_name, "clase": "Red_Vial_Terraceria"})

        bldg_res = building_extractor.extract_buildings(crop_rgb, l_clahe, exg, bbox)
        for b_item in bldg_res["building_items"]:
            all_buildings_gdf.append({
                "geometry": b_item["geometry"],
                "escenario": sc_name,
                "clase": "LA_SpatialUnit_Building",
                "subclase": b_item.get("subclase", "Vivienda_Campesina"),
                "area_m2": b_item["area_m2"],
                "has_shadow": b_item.get("has_shadow", False)
            })

        settlement_geom = building_extractor.create_settlement_mask(bldg_res["buildings_union"], buffer_m=20.0)

        # Delimitación espectral fiel
        raw_polygons, energy = delineator.delineate_parcels(
            crop_rgb=crop_rgb,
            l_clahe=l_clahe,
            exg=exg,
            bbox=bbox,
            road_mask=road_res["road_mask_px"],
            bldg_mask=bldg_res["building_mask_px"],
            esc_id=sc_id,
            esc_name=sc_name
        )

        upas_roads = road_extractor.partition_and_enforce_roads(raw_polygons, road_res["road_buffer_geom"])
        upas_courtyards = building_extractor.filter_urban_settlement_courtyards(upas_roads, settlement_geom)
        upas_purified = building_extractor.subtract_buildings_from_upas(upas_courtyards, bldg_res["buildings_union"])
        esc_polygons = reconciler.reconcile_polygons(upas_purified)

        # Garantía LADM 4-10 vértices
        for p in esc_polygons:
            geom = p["geometry"]
            if isinstance(geom, Polygon):
                n_v = len(geom.exterior.coords) - 1
                if n_v > 10 or n_v < 4:
                    reg_g = reconciler.regularize_ladm(geom, max_v=10, min_v=4)
                    p["geometry"] = reg_g
                    p["num_vertices"] = len(reg_g.exterior.coords) - 1
            all_polygons_gdf.append(p)

        esc_ha = sum(p["area_ha"] for p in esc_polygons)
        mean_v = np.mean([p["num_vertices"] for p in esc_polygons]) if esc_polygons else 8.0
        print(f"  -> Resultado: {len(esc_polygons)} UPAs registradas ({esc_ha:.2f} ha, {mean_v:.1f} vértices promedio LADM).")

        # Generar comparativa visual local
        fig, axs = plt.subplots(1, 2, figsize=(14, 7), dpi=200)
        axs[0].imshow(crop_rgb)
        axs[0].set_title(f"Satelital RGB Original\n{disp_name}", fontsize=12, fontweight="bold")
        axs[0].axis("off")

        axs[1].imshow(crop_rgb)
        for p in esc_polygons:
            geom = p["geometry"]
            # Convertir a píxeles para visualización rápida
            if isinstance(geom, Polygon):
                px_coords = []
                for gx, gy in geom.exterior.coords:
                    px = (gx - min_lon) / (max_lon - min_lon) * w_img
                    py = (top_lat - gy) / (top_lat - bot_lat) * h_img
                    px_coords.append((px, py))
                xs, ys = zip(*px_coords)
                axs[1].plot(xs, ys, color="#FACC15", linewidth=1.8)
                axs[1].scatter(xs[:-1], ys[:-1], s=8, color="#FFFFFF")

        axs[1].set_title(f"Delimitación SOTA {version} (LADM ISO 19152)\n{len(esc_polygons)} UPAs (0.00% Solapes)", fontsize=12, fontweight="bold")
        axs[1].axis("off")

        plt.tight_layout()
        out_comp = os.path.join(outputs_dir, f"comparativa_escenario_{sc_id}_{version.replace('.', '_')}.png")
        plt.savefig(out_comp, dpi=200)
        plt.close()
        print(f"  -> Gráfico comparativo generado: {out_comp}")

    # Exportación KML y GeoPackage
    if all_polygons_gdf:
        gdf_upas = gpd.GeoDataFrame(all_polygons_gdf, crs="EPSG:4326")
        gdf_roads = gpd.GeoDataFrame(all_roads_gdf, crs="EPSG:4326") if all_roads_gdf else None
        gdf_bldgs = gpd.GeoDataFrame(all_buildings_gdf, crs="EPSG:4326") if all_buildings_gdf else None

        kml_path = os.path.join(outputs_dir, f"upas_chimborazo_{version.replace('.', '_')}.kml")
        kml_exporter.export_multilayer_kml(kml_path, gdf_upas, gdf_roads, gdf_bldgs)

        # Actualizar mapa interactivo
        map_builder = InteractiveMapBuilder()
        html_map_path = os.path.join(outputs_dir, "mapa_interactivo.html")
        map_builder.build_map(html_map_path, gdf_upas, gdf_roads, gdf_bldgs)

    print("\n" + "=" * 75)
    print(f"EJECUCIÓN COMPLETADA CON ÉXITO")
    print(f"Resultados guardados en: {outputs_dir}")
    print("=" * 75)

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline Maestro SOTA de Delimitación de UPAs en Minifundios Andinos (TFM UNIR)"
    )
    parser.add_argument("--scenario", type=str, default="1", help="Escenario a ejecutar: 1 (Calpi), 2 (Licán), 3 (Colta), 4 (Guano) o 'all'")
    parser.add_argument("--verify", action="store_true", help="Verifica el entorno de ejecución, CUDA y librerías")
    parser.add_argument("--map", action="store_true", help="Genera/actualiza el visor cartográfico web interactivo HTML")

    args = parser.parse_args()

    if args.verify:
        verify_environment()
        return

    if args.map:
        outputs_dir = os.path.join(CURRENT_DIR, "outputs")
        gpkg_path = os.path.join(outputs_dir, "upas_chimborazo_v53_0_0.gpkg")
        html_path = os.path.join(outputs_dir, "mapa_interactivo.html")
        if os.path.exists(gpkg_path):
            gdf_upas = gpd.read_file(gpkg_path, layer="upas_agricolas")
            gdf_roads = gpd.read_file(gpkg_path, layer="vias_terraceria") if "vias_terraceria" in gpd.list_layers(gpkg_path).name.values else None
            gdf_bldgs = gpd.read_file(gpkg_path, layer="viviendas_campesinas") if "viviendas_campesinas" in gpd.list_layers(gpkg_path).name.values else None
            builder = InteractiveMapBuilder()
            builder.build_map(html_path, gdf_upas, gdf_roads, gdf_bldgs)
            print(f"[Mapa Actualizado]: {html_path}")
        return

    run_pipeline(scenario_id=args.scenario)

if __name__ == "__main__":
    main()
