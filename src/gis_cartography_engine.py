"""
================================================================================
MOTOR CARTOGRÁFICO DE BUENAS PRÁCTICAS GIS (TFM UNIR - GEOMASTER)
================================================================================
Genera mapas provinciales y de diseño muestral con:
  - Base de relieve sombreado DEM (Hillshade multiescala)
  - Provincias vecinas de Ecuador + Inset de localización nacional
  - División Política Administrativa (10 cantones y 59 parroquias)
  - Proyección métrica y canevá riguroso UTM 17S / WGS 84
================================================================================
"""

import os
import pathlib
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from unified_academic_style import (
    set_academic_style, draw_north_arrow, draw_scale_bar,
    add_neatline_ticks, OKABE_ITO, CANTON_PALETTE
)

set_academic_style()

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / 'data_v106'
CARTOGRAFIA_DIR = pathlib.Path(__file__).resolve().parents[3] / 'data' / 'cartografia'
EXP_FIGS_DIR = pathlib.Path(__file__).resolve().parent.parent / 'figuras_tfm'

CANTON_CLEAN_MAP = {
    'Riobamba': 'Riobamba', 'Guano': 'Guano', 'Colta': 'Colta',
    'Guamote': 'Guamote', 'Alausi': 'Alausí', 'Alausí': 'Alausí',
    'Chunchi': 'Chunchi', 'Pallatanga': 'Pallatanga',
    'Cumanda': 'Cumandá', 'Cumandá': 'Cumandá', 'Chambo': 'Chambo', 'Penipe': 'Penipe'
}

def load_shaded_relief():
    npz_path = DATA_DIR / 'chimborazo_shaded_relief_z10.npz'
    if not npz_path.exists():
        raise FileNotFoundError(f"No se encontró la caché de relieve sombreado: {npz_path}")
    data = np.load(npz_path)
    return data['mosaic'], data['extent']

def load_ecuador_provinces():
    path = CARTOGRAFIA_DIR / 'ecuador_provincias_gadm.geojson'
    if not path.exists():
        raise FileNotFoundError(f"No se encontró la capa de provincias de Ecuador: {path}")
    return gpd.read_file(path)

def load_chimborazo_cantones():
    # Cargar cantones depurados desde chimborazo_10_cantones_dpa o disolver parroquias
    path_cant = CARTOGRAFIA_DIR.parent.parent / 'delimitacion-upas-chimborazo' / 'data' / 'cartografia' / 'chimborazo_10_cantones_dpa.geojson'
    if path_cant.exists():
        return gpd.read_file(path_cant)
    path_parr = CARTOGRAFIA_DIR / 'parroquias_chimborazo_dpa.geojson'
    df_p = gpd.read_file(path_parr)
    return df_p.dissolve(by='canton', as_index=False)

def load_zonas_exclusion():
    path = CARTOGRAFIA_DIR / 'zonas_exclusion_chimborazo.geojson'
    return gpd.read_file(path)

# ==============================================================================
# 1. MAPA DE ESTUDIO PROVINCIAL DE CHIMBORAZO (BUENAS PRÁCTICAS GIS)
# ==============================================================================
def generate_mapa_estudio_provincial(out_path):
    print("  [GIS] Generando mapa_estudio_provincial_chimborazo.png...")
    relief_img, relief_ext = load_shaded_relief()
    gdf_ecuador = load_ecuador_provinces()
    gdf_cantones = load_chimborazo_cantones()
    
    fig, ax = plt.subplots(figsize=(10.5, 11.5), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#F0F4F8')
    
    # 1. Base Shaded Relief
    # extent: [left_lon, right_lon, bot_lat, top_lat]
    ax.imshow(relief_img, extent=[relief_ext[0], relief_ext[1], relief_ext[2], relief_ext[3]],
              origin='upper', alpha=0.88, zorder=1)
    
    # 2. Provincias vecinas de Ecuador (filtro de vecindad)
    neighbor_names = ['Tungurahua', 'Bolivar', 'Caar', 'Cañar', 'MoronaSantiago', 'Morona Santiago', 'Guayas', 'Pastaza', 'Cotopaxi']
    gdf_neighbors = gdf_ecuador[gdf_ecuador['NAME_1'].isin(neighbor_names) | gdf_ecuador['NAME_1'].str.contains('Bol|Ca|Tung|Moro|Guay', case=False, na=False)].copy()
    
    gdf_neighbors.plot(ax=ax, facecolor='#E2E8F0', edgecolor='#A0AEC0', lw=0.9, alpha=0.45, zorder=2)
    
    # Etiquetas de provincias vecinas
    lbl_neighbors = {
        'TUNGURAHUA': (-78.55, -1.37),
        'BOLÍVAR': (-79.15, -1.65),
        'GUAYAS': (-79.28, -2.15),
        'CAÑAR': (-78.95, -2.48),
        'MORONA SANTIAGO': (-78.30, -2.10)
    }
    for name, (lx, ly) in lbl_neighbors.items():
        ax.text(lx, ly, name, fontsize=8.0, fontweight='bold', fontstyle='italic', color='#718096',
                ha='center', va='center',
                path_effects=[patheffects.withStroke(linewidth=2.5, foreground='white')], zorder=3)
        
    # 3. Cantones de Chimborazo con paleta diferenciada y transparencia para ver el relieve
    for _, row in gdf_cantones.iterrows():
        c_raw = str(row.get('canton', row.get('NAME_2', 'Cantón'))).strip()
        c_clean = CANTON_CLEAN_MAP.get(c_raw, c_raw)
        col = CANTON_PALETTE.get(c_clean, '#FEF9E7')
        geom = row['geometry']
        gpd.GeoSeries([geom]).plot(ax=ax, facecolor=col, edgecolor='#2D3748', lw=1.1, alpha=0.55, zorder=4)
        
        # Etiqueta cantonal
        rep = geom.representative_point()
        canton_label_pos = {
            'Guano': (-78.68, -1.55), 'Riobamba': (-78.65, -1.70), 'Chambo': (-78.54, -1.73),
            'Colta': (-78.80, -1.74), 'Cumandá': (-79.13, -2.20), 'Chunchi': (-78.92, -2.31),
            'Alausí': (-78.82, -2.20), 'Guamote': (-78.70, -1.97), 'Pallatanga': (-78.96, -2.00), 'Penipe': (-78.48, -1.56)
        }
        pos = canton_label_pos.get(c_clean, (rep.x, rep.y))
        ax.text(pos[0], pos[1], c_clean.upper(), fontsize=7.8, fontweight='bold', color='#1A202C',
                ha='center', va='center',
                path_effects=[patheffects.withStroke(linewidth=2.5, foreground='white')], zorder=5)

    # 4. Destacar el Núcleo Piloto de San Andrés (Cantón Guano)
    san_andres_pt = Point(-78.6940, -1.6035)
    circle_sa = san_andres_pt.buffer(0.045)
    gpd.GeoSeries([circle_sa]).plot(ax=ax, facecolor='none', edgecolor='#C53030', lw=2.0, linestyle='--', zorder=8)
    ax.plot(san_andres_pt.x, san_andres_pt.y, marker='*', markersize=14, color='#E53E3E', markeredgecolor='white', markeredgewidth=1.2, zorder=9)
    ax.text(-78.61, -1.605, "PILOTO SAN ANDRÉS\n(Núcleo de Calibración 568 UPAs / 89.4 ha)",
            fontsize=8.0, fontweight='bold', color='#9B2C2C',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='#FFF5F5', edgecolor='#FEB2B2', lw=0.9),
            path_effects=[patheffects.withStroke(linewidth=2.0, foreground='white')], zorder=10)

    # 5. Encuadre, Canevá y Escala
    minx, miny, maxx, maxy = -79.35, -2.58, -78.25, -1.35
    add_neatline_ticks(ax, (minx, maxx), (miny, maxy), x_step=0.2, y_step=0.2)
    ax.set_xlabel("Longitud Oeste (WGS 84 / Grados Decimales)", fontweight='bold', labelpad=6)
    ax.set_ylabel("Latitud Sur (WGS 84 / Grados Decimales)", fontweight='bold', labelpad=6)
    
    # Barra de escala métrica: 25 km ~ 0.225 grados
    deg_25km = 25.0 / 111.0
    draw_scale_bar(ax, minx + 0.05, maxy - 0.09, deg_25km, "12.5", "25 km", fontsize=8.0)
    draw_north_arrow(ax, minx + 0.05 + deg_25km + 0.08, maxy - 0.085, size=0.045, fontsize=8.5)
    
    # 6. Inset Nacional de Ecuador
    ax_inset = inset_axes(ax, width="26%", height="26%", loc="lower left", borderpad=1.5)
    ax_inset.set_facecolor('#EBF8FF')
    gdf_ecuador.plot(ax=ax_inset, facecolor='#CBD5E0', edgecolor='#718096', lw=0.4, zorder=1)
    # Resaltar Chimborazo en rojo en el inset
    chimbo_poly = gdf_ecuador[gdf_ecuador['NAME_1'].str.contains('Chimborazo', case=False, na=False)]
    if not chimbo_poly.empty:
        chimbo_poly.plot(ax=ax_inset, facecolor='#E53E3E', edgecolor='#9B2C2C', lw=0.8, zorder=2)
    ax_inset.set_xlim(-81.5, -75.0)
    ax_inset.set_ylim(-5.2, 1.5)
    ax_inset.set_title("Ecuador", fontsize=7.0, fontweight='bold', color='#2D3748', pad=2)
    ax_inset.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax_inset.spines.values():
        spine.set_edgecolor('#4A5568')
        spine.set_linewidth(0.8)
        
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()
    print(f"  -> Guardado exitoso: {out_path}")

# ==============================================================================
# 2. MAPA DE ZONIFICACIÓN PARROQUIAL Y MÁSCARAS DE EXCLUSIÓN
# ==============================================================================
def generate_mapa_zonificacion_exclusion(out_path):
    print("  [GIS] Generando mapa_zonificacion_parroquial_exclusion_chimborazo.png...")
    relief_img, relief_ext = load_shaded_relief()
    gdf_parr = gpd.read_file(CARTOGRAFIA_DIR / 'parroquias_chimborazo_dpa.geojson')
    gdf_excl = load_zonas_exclusion()
    
    fig, ax = plt.subplots(figsize=(10.5, 11.5), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#F0F4F8')
    
    # 1. Base Shaded Relief
    ax.imshow(relief_img, extent=[relief_ext[0], relief_ext[1], relief_ext[2], relief_ext[3]],
              origin='upper', alpha=0.85, zorder=1)
    
    # 2. Parroquias Rurales Elegibles (45) vs Urbanas Excluidas (14)
    rural_parr = gdf_parr[gdf_parr['tipo'] == 'Rural']
    urban_parr = gdf_parr[gdf_parr['tipo'] == 'Urbana']
    
    rural_parr.plot(ax=ax, facecolor='#EDF2F7', edgecolor='#718096', lw=0.6, alpha=0.55, zorder=2)
    urban_parr.plot(ax=ax, facecolor='#FED7D7', edgecolor='#E53E3E', lw=0.9, alpha=0.65, zorder=3)
    
    # 3. Zonas de Exclusión Mayores (SNAP Chimborazo/Sangay y Laguna de Colta)
    snap_areas = gdf_excl[gdf_excl['categoria'].str.contains('SNAP', case=False, na=False)]
    snap_areas.plot(ax=ax, facecolor='#C6F6D5', edgecolor='#22543D', lw=1.2, alpha=0.65, zorder=4)
    
    agua = gdf_excl[gdf_excl['categoria'].str.contains('Agua', case=False, na=False)]
    if not agua.empty:
        agua.plot(ax=ax, facecolor='#63B3ED', edgecolor='#2B6CB0', lw=1.2, alpha=0.85, zorder=5)
        
    # 4. Resaltar Parroquia Piloto San Andrés
    sa_parr = gdf_parr[gdf_parr['parroquia'].str.contains('Andr', case=False, na=False)]
    if not sa_parr.empty:
        sa_parr.plot(ax=ax, facecolor='#FEFCBF', edgecolor='#D69E2E', lw=1.8, alpha=0.75, zorder=6)
        cent = sa_parr.geometry.iloc[0].centroid
        ax.plot(cent.x, cent.y, marker='o', markersize=8, color='#D69E2E', markeredgecolor='black', zorder=7)
        ax.text(cent.x + 0.02, cent.y + 0.015, "San Andrés\n(Piloto Agro-Catastral)",
                fontsize=8.0, fontweight='bold', color='#744210',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFFFF0', edgecolor='#D69E2E', lw=0.8), zorder=8)
        
    # Encuadre y Canevá
    minx, miny, maxx, maxy = -79.35, -2.58, -78.25, -1.35
    add_neatline_ticks(ax, (minx, maxx), (miny, maxy), x_step=0.2, y_step=0.2)
    ax.set_xlabel("Longitud Oeste (WGS 84 / Grados Decimales)", fontweight='bold', labelpad=6)
    ax.set_ylabel("Latitud Sur (WGS 84 / Grados Decimales)", fontweight='bold', labelpad=6)
    
    deg_25km = 25.0 / 111.0
    draw_scale_bar(ax, minx + 0.05, maxy - 0.09, deg_25km, "12.5", "25 km", fontsize=8.0)
    draw_north_arrow(ax, minx + 0.05 + deg_25km + 0.08, maxy - 0.085, size=0.045, fontsize=8.5)
    
    # Leyenda personalizada limpia
    import matplotlib.patches as mpatches
    patches = [
        mpatches.Patch(facecolor='#FEFCBF', edgecolor='#D69E2E', lw=1.2, label='Parroquia Piloto San Andrés (Enfoque Intensivo)'),
        mpatches.Patch(facecolor='#EDF2F7', edgecolor='#718096', lw=0.6, label='Parroquias Rurales Elegibles para UPAs (45)'),
        mpatches.Patch(facecolor='#FED7D7', edgecolor='#E53E3E', lw=0.9, label='Cabeceras Urbanas Excluidas (14)'),
        mpatches.Patch(facecolor='#C6F6D5', edgecolor='#22543D', lw=1.2, label='Áreas Protegidas SNAP (RPFCH y Sangay)'),
        mpatches.Patch(facecolor='#63B3ED', edgecolor='#2B6CB0', lw=1.2, label='Cuerpos Hídricos Mayores (Laguna de Colta)')
    ]
    ax.legend(handles=patches, loc='lower right', framealpha=0.92, facecolor='#FFFFFF', edgecolor='#CBD5E0', fontsize=7.8)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()
    print(f"  -> Guardado exitoso: {out_path}")

# ==============================================================================
# 3. MAPA DE DISEÑO MUESTRAL PROVINCIAL (24 CLÚSTERES NEYMAN + ZOOM 120 ENCUESTAS)
# ==============================================================================
def generate_mapa_diseno_muestral(out_path):
    print("  [GIS] Generando mapa_diseno_muestral_2400_upas.png...")
    relief_img, relief_ext = load_shaded_relief()
    gdf_cantones = load_chimborazo_cantones()
    encuestas_path = CARTOGRAFIA_DIR.parent / 'encuestas' / 'encuestas_chimborazo_anonimizadas.parquet'
    df_enc = pd.read_parquet(encuestas_path)
    
    clusters_data = [
        {"canton": "Riobamba", "nombre": "CL-01 Calpi Centro", "lon": -78.72, "lat": -1.64, "upas": 100},
        {"canton": "Riobamba", "nombre": "CL-02 San Juan Aluvial", "lon": -78.78, "lat": -1.62, "upas": 100},
        {"canton": "Riobamba", "nombre": "CL-03 Licán Hortícola", "lon": -78.69, "lat": -1.65, "upas": 100},
        {"canton": "Riobamba", "nombre": "CL-04 Cacha Ladera", "lon": -78.68, "lat": -1.70, "upas": 100},
        {"canton": "Riobamba", "nombre": "CL-05 Químiag Andino", "lon": -78.60, "lat": -1.67, "upas": 100},
        {"canton": "Riobamba", "nombre": "CL-06 Punín Quebrada", "lon": -78.64, "lat": -1.76, "upas": 100},
        {"canton": "Riobamba", "nombre": "CL-07 Flores Secano", "lon": -78.66, "lat": -1.82, "upas": 100},
        {"canton": "Guano", "nombre": "CL-08 San Andrés Piloto", "lon": -78.71, "lat": -1.60, "upas": 100},
        {"canton": "Guano", "nombre": "CL-09 Guano Central", "lon": -78.63, "lat": -1.55, "upas": 100},
        {"canton": "Guano", "nombre": "CL-10 Ilapo Frutícola", "lon": -78.61, "lat": -1.58, "upas": 100},
        {"canton": "Guano", "nombre": "CL-11 San Isidro Alto", "lon": -78.65, "lat": -1.525, "upas": 100},
        {"canton": "Guano", "nombre": "CL-12 Valparaíso Ladera", "lon": -78.55, "lat": -1.535, "upas": 100},
        {"canton": "Colta", "nombre": "CL-13 Cajabamba Laguna", "lon": -78.75, "lat": -1.72, "upas": 100},
        {"canton": "Colta", "nombre": "CL-14 Sicalpa Valle", "lon": -78.77, "lat": -1.70, "upas": 100},
        {"canton": "Colta", "nombre": "CL-15 Columbe Ladera", "lon": -78.80, "lat": -1.84, "upas": 100},
        {"canton": "Colta", "nombre": "CL-16 Santiago Páramo", "lon": -78.84, "lat": -1.77, "upas": 100},
        {"canton": "Guamote", "nombre": "CL-17 Guamote Cebadas", "lon": -78.65, "lat": -1.95, "upas": 100},
        {"canton": "Guamote", "nombre": "CL-18 Palmira Dunas", "lon": -78.74, "lat": -2.05, "upas": 100},
        {"canton": "Alausí", "nombre": "CL-19 Alausí Terrazas", "lon": -78.85, "lat": -2.20, "upas": 100},
        {"canton": "Alausí", "nombre": "CL-20 Sibambe Nariz", "lon": -78.90, "lat": -2.23, "upas": 100},
        {"canton": "Chunchi", "nombre": "CL-21 Chunchi Cuenca", "lon": -78.92, "lat": -2.29, "upas": 100},
        {"canton": "Pallatanga", "nombre": "CL-22 Pallatanga Frutales", "lon": -78.96, "lat": -1.99, "upas": 100},
        {"canton": "Chambo", "nombre": "CL-23 Chambo Riego", "lon": -78.58, "lat": -1.73, "upas": 100},
        {"canton": "Penipe", "nombre": "CL-24 Penipe Volcánico", "lon": -78.53, "lat": -1.56, "upas": 100}
    ]
    df_clusters = pd.DataFrame(clusters_data)
    
    fig, (ax_main, ax_det) = plt.subplots(1, 2, figsize=(16.5, 9.5), dpi=300, gridspec_kw={'width_ratios': [1.18, 1.0]})
    fig.patch.set_facecolor('#FFFFFF')
    ax_main.set_facecolor('#FFFFFF')
    ax_det.set_facecolor('#FFFFFF')
    
    # 1. Base Shaded Relief en Panel Principal
    ax_main.imshow(relief_img, extent=[relief_ext[0], relief_ext[1], relief_ext[2], relief_ext[3]],
                   origin='upper', alpha=0.75, zorder=1)
    
    # 2. Cantones
    for _, row in gdf_cantones.iterrows():
        c_raw = str(row.get('canton', row.get('NAME_2', 'Cantón'))).strip()
        c_name = CANTON_CLEAN_MAP.get(c_raw, c_raw)
        col = CANTON_PALETTE.get(c_name, '#FEF9E7')
        geom = row['geometry']
        gpd.GeoSeries([geom]).plot(ax=ax_main, facecolor=col, edgecolor='#4A5568', lw=0.9, alpha=0.55, zorder=2)
        rep = geom.representative_point()
        ax_main.text(rep.x, rep.y, c_name.upper(), fontsize=7.2, fontweight='bold', color='#2D3748',
                     ha='center', va='center',
                     path_effects=[patheffects.withStroke(linewidth=2.2, foreground='white')], zorder=3)
        
    # 3. Clústeres (24) con buffer geodésico
    for idx, cl in df_clusters.iterrows():
        pt_m = gpd.GeoSeries([Point(cl['lon'], cl['lat'])], crs="EPSG:4326").to_crs(epsg=32717)
        buf = pt_m.buffer(3500).to_crs(epsg=4326)
        buf.plot(ax=ax_main, facecolor='#2B6CB0', edgecolor='#1A365D', alpha=0.18, lw=0.9, linestyle=':', zorder=4)
        ax_main.plot(cl['lon'], cl['lat'], marker='o', markersize=6.5, color='#1A365D', markeredgecolor='white', markeredgewidth=1.1, zorder=7)
        c_id = f"C{idx + 1:02d}"
        ax_main.text(cl['lon'] + 0.016, cl['lat'] + 0.012, c_id, fontsize=6.5, fontweight='bold', color='#1A365D',
                     bbox=dict(boxstyle='round,pad=0.10', facecolor='white', edgecolor='#CBD5E0', lw=0.5, alpha=0.88), zorder=8)

    # Recuadro indicador de zona de detalle San Andrés - Riobamba
    rect_poly = Polygon([(-78.85, -1.69), (-78.56, -1.69), (-78.56, -1.505), (-78.85, -1.505)])
    gpd.GeoSeries([rect_poly]).plot(ax=ax_main, facecolor='#E53E3E', edgecolor='#C53030', alpha=0.12, lw=1.5, linestyle='--', zorder=9)
    ax_main.text(-78.845, -1.492, "Zona de Detalle Núcleo Piloto San Andrés (Panel Derecho)", fontsize=7.5, fontweight='bold', color='#9B2C2C',
                 bbox=dict(boxstyle='round,pad=0.15', facecolor='#FFF5F5', edgecolor='#FEB2B2', lw=0.6), zorder=10)

    minx, miny, maxx, maxy = -79.35, -2.58, -78.25, -1.35
    add_neatline_ticks(ax_main, (minx, maxx), (miny, maxy), x_step=0.2, y_step=0.2)
    deg_25km = 25.0 / 111.0
    draw_scale_bar(ax_main, minx + 0.05, maxy - 0.09, deg_25km, "12.5", "25 km", fontsize=8.0)
    draw_north_arrow(ax_main, minx + 0.05 + deg_25km + 0.08, maxy - 0.085, size=0.045, fontsize=8.5)
    ax_main.set_title("(a) Diseño Muestral Provincial: 24 Clústeres Bi-etápicos de Cochran (n = 2.400 UPAs)",
                      fontsize=9.2, fontweight='bold', color='#1A365D', pad=8)
    ax_main.set_xlabel("Longitud Oeste (EPSG:4326)", fontweight='bold')
    ax_main.set_ylabel("Latitud Sur (EPSG:4326)", fontweight='bold')

    # SUBPLOT 2: DETALLE NÚCLEO PILOTO SAN ANDRÉS (CON LAS 120 ENCUESTAS)
    ax_det.imshow(relief_img, extent=[relief_ext[0], relief_ext[1], relief_ext[2], relief_ext[3]],
                  origin='upper', alpha=0.65, zorder=1)
    
    cluster_labels_zoom = {
        'CL-01 Calpi Centro': ('C01: Calpi', 0.008, -0.012),
        'CL-02 San Juan Aluvial': ('C02: San Juan', 0.008, 0.006),
        'CL-03 Licán Hortícola': ('C03: Licán', 0.008, -0.008),
        'CL-05 Químiag Andino': ('C05: Químiag', 0.008, 0.006),
        'CL-08 San Andrés Piloto': ('C08: San Andrés (Piloto)', 0.008, 0.008),
        'CL-09 Guano Central': ('C09: Guano', 0.008, 0.006),
        'CL-10 Ilapo Frutícola': ('C10: Ilapo', 0.008, 0.006),
        'CL-11 San Isidro Alto': ('C11: San Isidro', 0.008, 0.006)
    }

    # Clústeres locales en el zoom
    for idx, cl in df_clusters.iterrows():
        if -78.85 <= cl['lon'] <= -78.56 and -1.69 <= cl['lat'] <= -1.505:
            pt_m = gpd.GeoSeries([Point(cl['lon'], cl['lat'])], crs="EPSG:4326").to_crs(epsg=32717)
            buf = pt_m.buffer(3500).to_crs(epsg=4326)
            buf.plot(ax=ax_det, facecolor='#3182CE', edgecolor='#2B6CB0', alpha=0.20, lw=1.2, linestyle=':', zorder=3)
            ax_det.plot(cl['lon'], cl['lat'], marker='o', markersize=8.0, color='#1A365D', markeredgecolor='white', markeredgewidth=1.2, zorder=5)
            
            lbl_info = cluster_labels_zoom.get(cl['nombre'], (f"C{idx+1:02d}", 0.008, 0.006))
            lbl_txt, off_x, off_y = lbl_info
            ax_det.text(cl['lon'] + off_x, cl['lat'] + off_y, lbl_txt,
                        fontsize=7.2, fontweight='bold', color='#1A365D',
                        bbox=dict(boxstyle='round,pad=0.15', facecolor='#FFFFFF', edgecolor='#CBD5E0', lw=0.6, alpha=0.92), zorder=6)

    # 120 Encuestas clasificadas por cultivo
    crop_styles = {
        'Papa': {'color': '#D69E2E', 'marker': 'o', 'label': 'Papa (Solanum tuberosum - 29 enc.)'},
        'Maíz': {'color': '#38A169', 'marker': 'o', 'label': 'Maíz Choclo (Zea mays - 38 enc.)'},
        'Alfalfa': {'color': '#319795', 'marker': 'o', 'label': 'Alfalfa / Pasto Forrajero (25 enc.)'},
        'Other': {'color': '#E53E3E', 'marker': 'o', 'label': 'Hortalizas y Leguminosas (28 enc.)'}
    }
    
    for _, enc in df_enc.iterrows():
        cult = str(enc.get('cultivo_declarado', ''))
        lon_c = enc['lon_parcela_real']
        lat_c = enc['lat_parcela_real']
        if -78.85 <= lon_c <= -78.56 and -1.69 <= lat_c <= -1.505:
            if 'Papa' in cult:
                c_key = 'Papa'
            elif 'Maíz' in cult or 'Maiz' in cult:
                c_key = 'Maíz'
            elif 'Alfalfa' in cult or 'Pasto' in cult:
                c_key = 'Alfalfa'
            else:
                c_key = 'Other'
            style = crop_styles[c_key]
            ax_det.plot(lon_c, lat_c, marker=style['marker'], markersize=5.8, color=style['color'],
                        markeredgecolor='#1A202C', markeredgewidth=0.7, alpha=0.9, zorder=7)

    ax_det.set_xlim(-78.85, -78.56)
    ax_det.set_ylim(-1.69, -1.505)
    add_neatline_ticks(ax_det, (-78.85, -78.56), (-1.69, -1.505), x_step=0.05, y_step=0.05)
    
    deg_5km = 5.0 / 111.0
    draw_scale_bar(ax_det, -78.83, -1.68, deg_5km, "2.5", "5 km", fontsize=7.5)
    ax_det.set_title("(b) Detalle Núcleo Piloto San Andrés: Puntos de Encuesta Censal (N = 120)",
                     fontsize=9.2, fontweight='bold', color='#1A365D', pad=8)
    ax_det.set_xlabel("Longitud Oeste (EPSG:4326)", fontweight='bold')
    ax_det.set_ylabel("Latitud Sur (EPSG:4326)", fontweight='bold')

    import matplotlib.lines as mlines
    leg_handles = [
        mlines.Line2D([], [], color='#1A365D', marker='o', linestyle='None', markersize=7, label='Clúster Bi-etápico (100 UPAs)'),
        mlines.Line2D([], [], color='#3182CE', marker='None', linestyle=':', lw=2, label='Buffer de Inferencia (r = 3.5 km)'),
        mlines.Line2D([], [], color='#D69E2E', marker='o', linestyle='None', markersize=6, label='Papa (Solanum tuberosum - 29 enc.)'),
        mlines.Line2D([], [], color='#38A169', marker='o', linestyle='None', markersize=6, label='Maíz Choclo (Zea mays - 38 enc.)'),
        mlines.Line2D([], [], color='#319795', marker='o', linestyle='None', markersize=6, label='Alfalfa / Pasto Forrajero (25 enc.)'),
        mlines.Line2D([], [], color='#E53E3E', marker='o', linestyle='None', markersize=6, label='Hortalizas y Leguminosas (28 enc.)')
    ]
    fig.legend(handles=leg_handles, loc='lower center', bbox_to_anchor=(0.51, -0.04), ncol=3,
               framealpha=0.95, facecolor='#FFFFFF', edgecolor='#CBD5E0', fontsize=8.2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()
    print(f"  -> Guardado exitoso: {out_path}")

if __name__ == '__main__':
    figs = EXP_FIGS_DIR
    figs.mkdir(parents=True, exist_ok=True)
    generate_mapa_estudio_provincial(figs / 'mapa_estudio_provincial_chimborazo.png')
    generate_mapa_zonificacion_exclusion(figs / 'mapa_zonificacion_parroquial_exclusion_chimborazo.png')
    generate_mapa_diseno_muestral(figs / 'mapa_diseno_muestral_2400_upas.png')
