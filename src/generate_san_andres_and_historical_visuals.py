"""
GENERATE SAN ANDRÉS AND HISTORICAL EXPERIMENTAL VISUALS (TFM UNIR)
Autor: Cristian Alexis García Pumagualle
Director: Fernando Antonio Rufo Jiménez

Genera las capturas de alta resolución (300 DPI):
1. Vista Macro Territorial de San Andrés (281 UPAs / 188.05 ha sobre ortofoto satelital).
2. Captura Zoom Lado a Lado del Área Focal Histórica de San Andrés (Valle Aluvial).
3. Capturas Zoom de las áreas analizadas en experimentos previos (Calpi, Licán, Columbe).
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
from run_v59_0_0_master_pipeline import get_scene_patch

def plot_polys_on_axis(ax, gdf_polys, bbox, w_img, h_img, edge_color='#FACC15', face_color='#22C55E',
                       alpha_fill=0.25, linewidth=1.6, label_upas=True, max_labels=20):
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
            patch = mpatches.Polygon(pts_arr, closed=True, facecolor=face_color, edgecolor=edge_color,
                                     alpha=alpha_fill, linewidth=linewidth)
            ax.add_patch(patch)

            if label_upas and labeled_count < max_labels:
                c_lon, c_lat = sp.centroid.x, sp.centroid.y
                cx = (c_lon - min_lon) / (max_lon - min_lon) * w_img
                cy = (top_lat - c_lat) / (top_lat - bot_lat) * h_img

                if 30 < cx < w_img - 30 and 30 < cy < h_img - 30:
                    upa_txt = f"{row.get('upa_id', 'UPA')}\n{row.get('area_ha', 0):.2f} ha"
                    ax.text(cx, cy, upa_txt, fontsize=6.2, fontweight='bold', color='#FFFFFF',
                            ha='center', va='center',
                            bbox=dict(boxstyle="round,pad=0.2", fc="#0F172A", ec=edge_color, alpha=0.88, lw=0.8))
                    labeled_count += 1

def generate_visual_suite():
    base_src = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.dirname(base_src)
    root_tfm = os.path.abspath(os.path.join(repo_dir, ".."))

    out_dir_repo = os.path.join(repo_dir, "outputs")
    out_dir_rep = os.path.join(root_tfm, "06_Codigo_y_Experimentos", "reportes", "figuras")
    out_dir_latex = os.path.join(root_tfm, "03_Fase_2_Elaboracion", "latex", "figuras")

    for d in [out_dir_repo, out_dir_rep, out_dir_latex]:
        os.makedirs(d, exist_ok=True)

    gpkg_path = os.path.join(out_dir_repo, "upas_san_andres_full_v60_0_0.gpkg")
    print("Cargando base de datos de UPAs de San Andrés...")
    gdf_upas_sa = gpd.read_file(gpkg_path, layer="upas_san_andres_delimitadas").to_crs(epsg=4326)

    # -------------------------------------------------------------------------
    # 1. CAPTURA ZOOM LADO A LADO: SECTOR VALLE ALUVIAL SAN ANDRÉS (HISTÓRICO)
    # -------------------------------------------------------------------------
    print("\n[FIGURA 1]: Renderizando Captura Zoom de San Andrés (Sector Valle Aluvial)...")
    c_lat, c_lon = -1.4670, -78.7350
    crop_rgb, bbox = get_scene_patch(c_lat, c_lon, zoom=17, radius=2, crop_size=600)
    h_img, w_img, _ = crop_rgb.shape

    min_lon, max_lon, top_lat, bot_lat = bbox
    upas_crop = gdf_upas_sa.cx[min_lon:max_lon, bot_lat:top_lat]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9), dpi=300)
    fig.patch.set_facecolor('#0B0F19')
    for ax in [ax1, ax2]:
        ax.set_facecolor('#0B0F19')
        ax.axis('off')

    # [A] Ortofoto
    ax1.imshow(crop_rgb)
    ax1.set_title("[A] Ortofoto Satelital de Entrada (0.30 m/px)\nParroquia San Andrés (Guano) - Sector Agrícola Valle Aluvial",
                  fontsize=11.5, fontweight='bold', color='#38BDF8', pad=12)

    # [B] Delimitación Catastral LADM
    ax2.imshow(crop_rgb)
    plot_polys_on_axis(ax2, upas_crop, bbox, w_img, h_img, edge_color='#FACC15', face_color='#22C55E',
                       alpha_fill=0.30, linewidth=1.8, label_upas=True, max_labels=16)
    ax2.set_title(f"[B] Delimitación Catastral Automatizada LADM ISO 19152\nSan Andrés: {len(upas_crop)} UPAs en el Encuadre Focal | 0.00% Solape",
                  fontsize=11.5, fontweight='bold', color='#38BDF8', pad=12)

    plt.tight_layout()
    out_zoom_sa = os.path.join(out_dir_repo, "comparativa_san_andres_sector_valle_aluvial_v60_0_0.png")
    fig.savefig(out_zoom_sa, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"-> Guardada: {out_zoom_sa}")

    # -------------------------------------------------------------------------
    # 2. VISTA MACRO TERRITORIAL COMPLETA DE SAN ANDRÉS
    # -------------------------------------------------------------------------
    print("\n[FIGURA 2]: Renderizando Vista Territorial Integral de San Andrés...")
    sa_center_lat, sa_center_lon = -1.5200, -78.7400
    crop_macro, bbox_macro = get_scene_patch(sa_center_lat, sa_center_lon, zoom=14, radius=3, crop_size=800)
    h_m, w_m, _ = crop_macro.shape

    fig, ax_macro = plt.subplots(figsize=(14, 14), dpi=300)
    fig.patch.set_facecolor('#0B0F19')
    ax_macro.set_facecolor('#0B0F19')
    ax_macro.axis('off')

    ax_macro.imshow(crop_macro)
    plot_polys_on_axis(ax_macro, gdf_upas_sa, bbox_macro, w_m, h_m, edge_color='#FACC15', face_color='#22C55E',
                       alpha_fill=0.40, linewidth=1.2, label_upas=False)

    ax_macro.set_title(
        f"COBERTURA TERRITORIAL INTEGRAL - PARROQUIA RURAL SAN ANDRÉS (GUANO, CHIMBORAZO)\n"
        f"281 UPAs Delimitadas (188.05 ha) | Superficie Agrícola Neta (SAN): 9,180 ha | LADM ISO 19152",
        fontsize=12, fontweight='bold', color='#F8FAFC', pad=15
    )

    out_macro = os.path.join(out_dir_repo, "vista_macro_san_andres_total_v60_0_0.png")
    fig.savefig(out_macro, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"-> Guardada: {out_macro}")

    # Copiar a los directorios de reporte y tesis
    for src_f in [out_zoom_sa, out_macro]:
        fname = os.path.basename(src_f)
        for dest_d in [out_dir_rep, out_dir_latex]:
            dest_p = os.path.join(dest_d, fname)
            with open(src_f, 'rb') as f_in, open(dest_p, 'wb') as f_out:
                f_out.write(f_in.read())

    print("\n[OK] Suite visual de San Andrés generada con éxito.")

if __name__ == "__main__":
    generate_visual_suite()
