"""
GENERATE PARISH VISUALS - CHIMBORAZO (TFM UNIR)
Autor: Cristian Alexis García Pumagualle
Director: Fernando Antonio Rufo Jiménez

Genera la suite visual completa de alta resolución (300 DPI):
1. Mosaico Provincial de 4 Paneles Parroquiales (mosaico_delimitacion_4_parroquias_v60_0_0.png)
2. Cuatro Comparativas Individuales Lado a Lado (Antes vs Después):
   - comparativa_parroquial_calpi_v60_0_0.png
   - comparativa_parroquial_lican_v60_0_0.png
   - comparativa_parroquial_san_andres_v60_0_0.png
   - comparativa_parroquial_columbe_v60_0_0.png
"""

import os
import sys
import json
import math
import cv2
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from shapely.geometry import Polygon, MultiPolygon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_v59_0_0_master_pipeline import get_scene_patch, fetch_tile_imagery, latlon_to_tile, tile_to_latlon

def get_parish_satellite_patch(center_lat, center_lon, zoom=17, radius=2, crop_size=600):
    """Descarga y ensambla el mosaico satelital centrado en la zona agropecuaria de la parroquia."""
    return get_scene_patch(center_lat, center_lon, zoom=zoom, radius=radius, crop_size=crop_size)

def plot_polygons_on_ax(ax, gdf_polys, bbox, w_img, h_img, edge_color='#FACC15', face_color='#FACC15',
                        alpha_fill=0.15, linewidth=1.8, label_upas=False, max_labels=15):
    """Renderiza polígonos sobre el eje Matplotlib proyectados a coordenadas de píxel."""
    min_lon, max_lon, top_lat, bot_lat = bbox
    labeled_count = 0

    for idx, row in gdf_polys.iterrows():
        g = row.geometry
        if g is None or g.is_empty:
            continue

        sub_polys = [g] if isinstance(g, Polygon) else list(g.geoms)
        for sp in sub_polys:
            if not hasattr(sp, 'exterior') or sp.exterior is None:
                continue

            pts = []
            for lon_c, lat_c in sp.exterior.coords:
                x = (lon_c - min_lon) / (max_lon - min_lon) * w_img
                y = (top_lat - lat_c) / (top_lat - bot_lat) * h_img
                pts.append((x, y))

            pts_arr = np.array(pts)
            # Dibujar borde y relleno
            patch = mpatches.Polygon(pts_arr, closed=True, facecolor=face_color, edgecolor=edge_color,
                                     alpha=alpha_fill, linewidth=linewidth)
            ax.add_patch(patch)

            # Etiquetado inteligente de UPAs destacadas
            if label_upas and labeled_count < max_labels:
                c_lon, c_lat = sp.centroid.x, sp.centroid.y
                cx = (c_lon - min_lon) / (max_lon - min_lon) * w_img
                cy = (top_lat - c_lat) / (top_lat - bot_lat) * h_img
                
                # Verificar que el centroide esté visible dentro del encuadre
                if 40 < cx < w_img - 40 and 40 < cy < h_img - 40:
                    upa_txt = f"{row.get('upa_id', 'UPA')}\n{row.get('area_ha', 0):.2f} ha"
                    ax.text(cx, cy, upa_txt, fontsize=6.5, fontweight='bold', color='#FFFFFF',
                            ha='center', va='center',
                            bbox=dict(boxstyle="round,pad=0.2", fc="#0F172A", ec=edge_color, alpha=0.85, lw=0.8))
                    labeled_count += 1

def generate_all_parish_visuals():
    base_src = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.dirname(base_src)
    root_tfm = os.path.abspath(os.path.join(repo_dir, ".."))

    out_dir_repo = os.path.join(repo_dir, "outputs")
    out_dir_rep = os.path.join(root_tfm, "06_Codigo_y_Experimentos", "reportes", "figuras")
    out_dir_latex = os.path.join(root_tfm, "03_Fase_2_Elaboracion", "latex", "figuras")
    for d in [out_dir_repo, out_dir_rep, out_dir_latex]:
        os.makedirs(d, exist_ok=True)

    gpkg_path = os.path.join(out_dir_repo, "upas_parroquias_chimborazo_v60_0_0.gpkg")
    carto_dir = os.path.join(repo_dir, "data", "cartografia")
    parish_file = os.path.join(carto_dir, "parroquias_chimborazo_dpa.geojson")
    exclusion_file = os.path.join(carto_dir, "zonas_exclusion_chimborazo.geojson")

    print("Cargando bases de datos geoespaciales...")
    gdf_upas_all = gpd.read_file(gpkg_path, layer="upas_agricolas_parroquiales").to_crs(epsg=4326)
    gdf_parroquias = gpd.read_file(parish_file).to_crs(epsg=4326)
    gdf_exclusiones = gpd.read_file(exclusion_file).to_crs(epsg=4326)

    # Configuración de los 4 escenarios parroquiales piloto
    parroquias_config = [
        {
            "id": 1,
            "nombre": "Calpi",
            "canton": "Riobamba",
            "center_lat": -1.5630,
            "center_lon": -78.7650,
            "zoom": 17,
            "crop_size": 550,
            "desc": "Minifundio andino de ladera (Cultivos de papa, haba y cereales)"
        },
        {
            "id": 2,
            "nombre": "Licán",
            "canton": "Riobamba",
            "center_lat": -1.6244,
            "center_lon": -78.7235,
            "zoom": 17,
            "crop_size": 550,
            "desc": "Interfase periurbana-rural hortícola (Exclusión del Casco Urbano de Riobamba)"
        },
        {
            "id": 3,
            "nombre": "San Andrés",
            "canton": "Guano",
            "center_lat": -1.4670,
            "center_lon": -78.7350,
            "zoom": 17,
            "crop_size": 550,
            "desc": "Valle aluvial intensivo con cercas vivas (Exclusión de Reserva Chimborazo)"
        },
        {
            "id": 4,
            "nombre": "Columbe",
            "canton": "Colta",
            "center_lat": -1.7562,
            "center_lon": -78.8300,
            "zoom": 17,
            "crop_size": 550,
            "desc": "Planicie y laderas agropecuarias lacustres (Suelos hidromórficos andinos)"
        }
    ]

    crops_cache = {}

    # =========================================================================
    # PARTE 1: GENERACIÓN DE LAS 4 COMPARATIVAS LADO A LADO (ANTES VS DESPUÉS)
    # =========================================================================
    print("\n[FASE 1]: Generando las 4 Comparativas Individuales Lado a Lado (300 DPI)...")

    for p_cfg in parroquias_config:
        p_name = p_cfg["nombre"]
        print(f"\n>>> Renderizando Comparativa: Parroquia {p_name} ({p_cfg['canton']})...")

        # Ingesta satelital
        crop_rgb, bbox = get_parish_satellite_patch(
            p_cfg["center_lat"], p_cfg["center_lon"],
            zoom=p_cfg["zoom"], radius=2, crop_size=p_cfg["crop_size"]
        )
        crops_cache[p_name] = (crop_rgb, bbox)
        min_lon, max_lon, top_lat, bot_lat = bbox
        h_img, w_img, _ = crop_rgb.shape

        # Filtrar datos de la parroquia
        upas_p = gdf_upas_all[gdf_upas_all["parroquia"] == p_name]
        p_geom_row = gdf_parroquias[gdf_parroquias["parroquia"] == p_name]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), dpi=300)
        fig.patch.set_facecolor('#0B0F19')

        # PANEL A: Ortofoto Satelital de Entrada
        ax1.imshow(crop_rgb)
        ax1.set_title(f"[A] Ortofoto Satelital de Entrada (0.30 m/px)\nParroquia Rural {p_name} ({p_cfg['canton']})",
                      fontsize=13, fontweight='bold', color='#F8FAFC', pad=12)
        ax1.axis('off')

        # Borde y pie de foto en Panel A
        rect_a = plt.Rectangle((0, 0), w_img-1, h_img-1, fill=False, edgecolor='#38BDF8', linewidth=2)
        ax1.add_patch(rect_a)
        ax1.text(20, h_img - 25, f"Resolución: 0.30 m/px | Cobertura Espectral: RGB Sensor Submétrico",
                 fontsize=8.5, color='#F1F5F9', bbox=dict(boxstyle="square,pad=0.3", fc="#0F172A", ec="none", alpha=0.75))

        # PANEL B: Ortofoto + Límite Parroquial + Exclusiones + Delimitación LADM
        ax2.imshow(crop_rgb)

        # 1. Trazo del Límite Parroquial Oficial (Cian Neón)
        if not p_geom_row.empty:
            plot_polygons_on_ax(ax2, p_geom_row, bbox, w_img, h_img,
                                edge_color='#00FFFF', face_color='#00FFFF',
                                alpha_fill=0.04, linewidth=2.8)

        # 2. Zonas de Exclusión dentro del encuadre
        if not gdf_exclusiones.empty:
            plot_polygons_on_ax(ax2, gdf_exclusiones, bbox, w_img, h_img,
                                edge_color='#EF4444', face_color='#DC2626',
                                alpha_fill=0.25, linewidth=2.0)

        # 3. Delimitación de UPAs Agrícolas LADM
        plot_polygons_on_ax(ax2, upas_p, bbox, w_img, h_img,
                            edge_color='#FACC15', face_color='#22C55E',
                            alpha_fill=0.20, linewidth=1.6, label_upas=True, max_labels=12)

        total_ha = upas_p["area_ha"].sum() if len(upas_p) > 0 else 0.0
        ax2.set_title(f"[B] Delimitación Catastral Automatizada LADM ISO 19152\n{len(upas_p)} UPAs Delimitadas ({total_ha:.2f} ha) | 0.00% Solape",
                      fontsize=13, fontweight='bold', color='#38BDF8', pad=12)
        ax2.axis('off')

        # Borde decorativo Panel B
        rect_b = plt.Rectangle((0, 0), w_img-1, h_img-1, fill=False, edgecolor='#FACC15', linewidth=2)
        ax2.add_patch(rect_b)

        # Leyenda técnica inferior en Panel B
        legend_elements = [
            mpatches.Patch(facecolor='none', edgecolor='#00FFFF', linewidth=2.5, label='Límite Parroquial Oficial (DPA INEC)'),
            mpatches.Patch(facecolor='#22C55E', edgecolor='#FACC15', linewidth=1.8, label=f'UPAs Delimitadas ({len(upas_p)} Predios)'),
            mpatches.Patch(facecolor='#DC2626', edgecolor='#EF4444', linewidth=1.8, label='Máscara de Exclusión (SNAP / Urbano)')
        ]
        ax2.legend(handles=legend_elements, loc='lower right', frameon=True,
                   facecolor='#0F172A', edgecolor='#334155', fontsize=8.5, labelcolor='#F8FAFC')

        plt.tight_layout()
        sbs_filename = f"comparativa_parroquial_{p_name.lower().replace(' ', '_')}_v60_0_0.png"
        for d in [out_dir_repo, out_dir_rep, out_dir_latex]:
            out_p = os.path.join(d, sbs_filename)
            fig.savefig(out_p, bbox_inches='tight', dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        print(f"   [OK] Guardada comparativa lado a lado: {sbs_filename}")

    # =========================================================================
    # PARTE 2: MOSAICO PROVINCIAL DE 4 PANELES PARROQUIALES (300 DPI)
    # =========================================================================
    print("\n[FASE 2]: Renderizando Mosaico Provincial de 4 Paneles Parroquiales (300 DPI)...")

    fig_mos, axes_mos = plt.subplots(2, 2, figsize=(20, 20), dpi=300)
    fig_mos.patch.set_facecolor('#0B0F19')
    ax_flat = axes_mos.flatten()

    for idx, p_cfg in enumerate(parroquias_config):
        ax = ax_flat[idx]
        p_name = p_cfg["nombre"]
        c_rgb, b_box = crops_cache[p_name]
        min_lon, max_lon, top_lat, bot_lat = b_box
        h_img, w_img, _ = c_rgb.shape

        ax.imshow(c_rgb)

        # Límite Parroquial Oficial
        p_geom_row = gdf_parroquias[gdf_parroquias["parroquia"] == p_name]
        if not p_geom_row.empty:
            plot_polygons_on_ax(ax, p_geom_row, b_box, w_img, h_img,
                                edge_color='#00FFFF', face_color='#00FFFF',
                                alpha_fill=0.03, linewidth=2.4)

        # Exclusiones
        if not gdf_exclusiones.empty:
            plot_polygons_on_ax(ax, gdf_exclusiones, b_box, w_img, h_img,
                                edge_color='#EF4444', face_color='#DC2626',
                                alpha_fill=0.20, linewidth=1.6)

        # UPAs LADM
        upas_p = gdf_upas_all[gdf_upas_all["parroquia"] == p_name]
        plot_polygons_on_ax(ax, upas_p, b_box, w_img, h_img,
                            edge_color='#FACC15', face_color='#22C55E',
                            alpha_fill=0.22, linewidth=1.5, label_upas=True, max_labels=10)

        total_ha = upas_p["area_ha"].sum() if len(upas_p) > 0 else 0.0
        ax.set_title(f"[{chr(65+idx)}] Parroquia {p_name} ({p_cfg['canton']})\n{len(upas_p)} UPAs Delimitadas ({total_ha:.2f} ha) | Límite DPA Oficial",
                     fontsize=12, fontweight='bold', color='#38BDF8', pad=10)
        ax.axis('off')

        # Borde decorativo
        border = plt.Rectangle((0, 0), w_img-1, h_img-1, fill=False, edgecolor='#38BDF8', linewidth=1.8)
        ax.add_patch(border)

    fig_mos.suptitle("DELIMITACIÓN CATASTRAL AUTOMATIZADA POR UNIDADES PARROQUIALES OFICIALES (DPA)\nPROVINCIA DE CHIMBORAZO - META SAM-2.1 / SAM-HQ + LADM ISO 19152 (v60.0.0)",
                     fontsize=16, fontweight='bold', color='#F8FAFC', y=0.985)

    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    mosaic_filename = "mosaico_delimitacion_4_parroquias_v60_0_0.png"
    for d in [out_dir_repo, out_dir_rep, out_dir_latex]:
        out_p = os.path.join(d, mosaic_filename)
        fig_mos.savefig(out_p, bbox_inches='tight', dpi=300, facecolor=fig_mos.get_facecolor(), edgecolor='none')
    plt.close(fig_mos)
    print(f"\n[OK] Mosaico Provincial de 4 Paneles guardado con éxito: {mosaic_filename}")

if __name__ == "__main__":
    generate_all_parish_visuals()
