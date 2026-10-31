"""
Generador de Renders Visuales de Alta Definición para Auditoría Multimodal MoE
Basado en los resultados de la A100: upas_san_andres_v61_a100.gpkg
"""

import os
import sys
import math
import numpy as np
import cv2
import pyproj
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def deg2num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)

def num2deg(xtile, ytile, zoom):
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)

def fetch_tile(x, y, z, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    tile_path = os.path.join(cache_dir, f"tile_{z}_{x}_{y}.png")
    if os.path.exists(tile_path):
        img = cv2.imread(tile_path)
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    url = f"https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
    import urllib.request
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
        with urllib.request.urlopen(req, timeout=5) as resp:
            arr = np.asarray(bytearray(resp.read()), dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                cv2.imwrite(tile_path, img)
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception:
        pass
    return np.zeros((256, 256, 3), dtype=np.uint8)

def get_scene_patch(lat, lon, zoom=18, radius=3, crop_size=600, cache_dir="./data/tile_cache"):
    cx, cy = deg2num(lat, lon, zoom)
    tiles = []
    for dy in range(-radius, radius + 1):
        row = []
        for dx in range(-radius, radius + 1):
            t = fetch_tile(cx + dx, cy + dy, zoom, cache_dir)
            row.append(t)
        tiles.append(np.hstack(row))
    full_mosaic = np.vstack(tiles)
    
    H, W, _ = full_mosaic.shape
    center_y, center_x = H // 2, W // 2
    half_crop = crop_size // 2
    min_y = max(0, center_y - half_crop)
    max_y = min(H, center_y + half_crop)
    min_x = max(0, center_x - half_crop)
    max_x = min(W, center_x + half_crop)
    crop_rgb = full_mosaic[min_y:max_y, min_x:max_x]
    
    top_lat, left_lon = num2deg(cx - radius, cy - radius, zoom)
    bot_lat, right_lon = num2deg(cx + radius + 1, cy + radius + 1, zoom)
    
    c_left_lon = left_lon + (min_x / W) * (right_lon - left_lon)
    c_right_lon = left_lon + (max_x / W) * (right_lon - left_lon)
    c_top_lat = top_lat - (min_y / H) * (top_lat - bot_lat)
    c_bot_lat = top_lat - (max_y / H) * (top_lat - bot_lat)
    
    return crop_rgb, (c_left_lon, c_right_lon, c_top_lat, c_bot_lat)

def main():
    repo_root = r"K:\Mi unidad\CURSOS Y CAPACITACIÓN PERSONAL\MASTER-INTELIGENCIA ARTIFICIAL\Aula Master IA\02_Segundo_Cuatrimestre\09_TFM\delimitacion-upas-chimborazo"
    outputs_dir = os.path.join(repo_root, "outputs")
    cache_dir = os.path.join(repo_root, "data", "tile_cache")
    gpkg_path = os.path.join(outputs_dir, "upas_san_andres_v61_a100.gpkg")

    gdf_upas = gpd.read_file(gpkg_path, layer="upas_delimitadas_a100")
    gdf_bldgs = gpd.read_file(gpkg_path, layer="edificaciones_ladm")

    transformer_to_utm = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32717", always_xy=True)

    cuadrantes = [
        {"id": 1, "name": "1_San_Andres_Centro_Valle_Aluvial", "lat": -1.5957, "lon": -78.7008, "desc": "San Andrés Centro (Valle Aluvial)"},
        {"id": 2, "name": "2_San_Andres_Chocavi_Laderas_Andinas", "lat": -1.5830, "lon": -78.6920, "desc": "Chocaví (Laderas Andinas)"},
        {"id": 3, "name": "3_San_Andres_Pulucate_Piedemonte", "lat": -1.5620, "lon": -78.7250, "desc": "Pulucate (Piedemonte Chimborazo)"},
        {"id": 4, "name": "4_San_Andres_Batzacon_Interfase_Vial", "lat": -1.6150, "lon": -78.6780, "desc": "Batzacón (Interfase Vial Rural)"}
    ]

    images_saved = []

    for c in cuadrantes:
        cid = c["id"]
        cname = c["name"]
        cdesc = c["desc"]
        print(f"Renderizando Cuadrante A100 {cid}: {cdesc}...")

        crop_rgb, bbox = get_scene_patch(c["lat"], c["lon"], zoom=18, radius=3, crop_size=600, cache_dir=cache_dir)
        h_img, w_img, _ = crop_rgb.shape
        min_lon, max_lon, top_lat, bot_lat = bbox
        x_min, y_min = transformer_to_utm.transform(min_lon, bot_lat)
        x_max, y_max = transformer_to_utm.transform(max_lon, top_lat)

        # Filtrar geometrías que intersecan la caja del cuadrante
        quad_upas = gdf_upas[gdf_upas['escenario_id'] == cid] if 'escenario_id' in gdf_upas.columns else gdf_upas[gdf_upas.geometry.intersects(gdf_upas.unary_union)]

        fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=220)

        # Panel 1: Ortofoto original
        axes[0].imshow(crop_rgb)
        axes[0].set_title(f"Ortofoto VHR Satelital (0.30 m/px) - {cdesc}", fontsize=12, fontweight='bold', pad=10)
        axes[0].axis('off')

        # Panel 2: Superposición de polígonos delimitados por A100
        axes[1].imshow(crop_rgb)
        axes[1].set_title(f"Delimitación A100 SOTA (Coverage Simplify LADM) - {cdesc}", fontsize=12, fontweight='bold', pad=10)

        # Plot UPAs
        for _, row in quad_upas.iterrows():
            geom = row.geometry
            if geom.geom_type == "Polygon":
                coords = list(geom.exterior.coords)
                px_coords = []
                for ux, uy in coords:
                    px = (ux - x_min) / (x_max - x_min) * w_img
                    py = (y_max - uy) / (y_max - y_min) * h_img
                    px_coords.append((px, py))
                
                poly_patch = mpatches.Polygon(
                    px_coords, 
                    closed=True, 
                    edgecolor='#16A34A', 
                    facecolor=(0.13, 0.77, 0.37, 0.32), 
                    linewidth=1.8
                )
                axes[1].add_patch(poly_patch)

        # Plot Viviendas
        for _, b in gdf_bldgs.iterrows():
            geom_b = b.geometry
            if geom_b.intersects(quad_upas.unary_union if not quad_upas.empty else geom_b):
                coords_b = list(geom_b.exterior.coords)
                b_coords = []
                for bx, by in coords_b:
                    bpx = (bx - x_min) / (x_max - x_min) * w_img
                    bpy = (y_max - by) / (y_max - y_min) * h_img
                    b_coords.append((bpx, bpy))
                bldg_patch = mpatches.Polygon(
                    b_coords, 
                    closed=True, 
                    edgecolor='#DC2626', 
                    facecolor=(0.95, 0.20, 0.20, 0.70), 
                    linewidth=1.5
                )
                axes[1].add_patch(bldg_patch)

        # Leyenda LADM
        patch_upa = mpatches.Patch(color='#16A34A', label=f'UPAs Delimitadas (A100 SAM-HQ: {len(quad_upas)})')
        patch_bldg = mpatches.Patch(color='#DC2626', label='Edificaciones LADM (Buffer 20m)')
        axes[1].legend(handles=[patch_upa, patch_bldg], loc='lower right', framealpha=0.9, fontsize=9)

        axes[1].axis('off')
        plt.tight_layout()
        out_png = os.path.join(outputs_dir, f"comparativa_a100_cuadrante_{cid}_san_andres.png")
        plt.savefig(out_png, bbox_inches='tight')
        plt.close()
        images_saved.append(out_png)
        print(f"  [OK] Guardado: {out_png}")

    print("\nTodos los renders A100 generados con éxito para la inspección multimodal MoE.")
    return images_saved

if __name__ == "__main__":
    main()
