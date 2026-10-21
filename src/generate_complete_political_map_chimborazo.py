"""
GENERATE COMPLETE POLITICAL MAP OF CHIMBORAZO (TFM UNIR)
Autor: Cristian Alexis García Pumagualle
Director: Fernando Antonio Rufo Jiménez

Genera el Mapa Político-Administrativo y Ambiental Oficial de la Provincia de Chimborazo (300 DPI):
- Jerarquía Provincial: Límite provincial exterior (479,592 ha).
- Jerarquía Cantonal: Los 10 cantones oficiales con fronteras y cartelas toponímicas.
- Jerarquía Parroquial: Las 59 parroquias oficiales DPA INEC (45 rurales y 14 urbanas).
- Áreas Protegidas SNAP: Reserva de Producción de Fauna Chimborazo y Parque Nacional Sangay.
- Cascos Urbanos Consolidados y Cuerpos de Agua (Laguna de Colta).
- Foco Estratégico: Resaltado de la demarcación territorial de San Andrés (Guano).
"""

import os
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from shapely.geometry import Point

def generate_political_map():
    base_src = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.dirname(base_src)
    root_tfm = os.path.abspath(os.path.join(repo_dir, ".."))

    carto_dir = os.path.join(repo_dir, "data", "cartografia")
    parish_file = os.path.join(carto_dir, "parroquias_chimborazo_dpa.geojson")
    exclusion_file = os.path.join(carto_dir, "zonas_exclusion_chimborazo.geojson")

    out_repo = os.path.join(repo_dir, "outputs", "mapa_politico_provincial_cantonal_parroquial_chimborazo.png")
    out_rep = os.path.join(root_tfm, "06_Codigo_y_Experimentos", "reportes", "figuras", "mapa_politico_provincial_cantonal_parroquial_chimborazo.png")
    out_latex = os.path.join(root_tfm, "03_Fase_2_Elaboracion", "latex", "figuras", "mapa_politico_provincial_cantonal_parroquial_chimborazo.png")

    for d in [os.path.dirname(out_repo), os.path.dirname(out_rep), os.path.dirname(out_latex)]:
        os.makedirs(d, exist_ok=True)

    print("Cargando capas cartográficas oficiales...")
    gdf_parroquias = gpd.read_file(parish_file).to_crs(epsg=32717)
    gdf_exclusiones = gpd.read_file(exclusion_file).to_crs(epsg=32717)

    # Disolver para obtener cantones y límite provincial
    gdf_cantones = gdf_parroquias.dissolve(by="canton", as_index=False)
    gdf_provincia = gdf_parroquias.dissolve()

    # Guardar capas vectoriales GeoJSON para interoperabilidad SIG
    gdf_cantones.to_file(os.path.join(carto_dir, "chimborazo_10_cantones_dpa.geojson"), driver="GeoJSON")
    gdf_provincia.to_file(os.path.join(carto_dir, "chimborazo_limite_provincial_dpa.geojson"), driver="GeoJSON")
    print("-> Capas vectoriales cantonales y provinciales guardadas en data/cartografia/")

    fig, ax = plt.subplots(figsize=(14, 16), dpi=300)
    fig.patch.set_facecolor('#0B0F19')
    ax.set_facecolor('#0F172A')

    # 1. Parroquias Rurales de Fondo
    rurales = gdf_parroquias[gdf_parroquias["tipo"] == "Rural"]
    rurales.plot(ax=ax, color='#1E293B', edgecolor='#334155', linewidth=0.6, alpha=0.9, zorder=2)

    # 2. Parroquias Urbanas
    urbanas = gdf_parroquias[gdf_parroquias["tipo"] == "Urbana"]
    urbanas.plot(ax=ax, color='#450A0A', edgecolor='#B91C1C', linewidth=0.9, alpha=0.8, hatch='//', zorder=3)

    # 3. Áreas Protegidas del SNAP
    snap_rpf = gdf_exclusiones[gdf_exclusiones["id_exclusion"] == "EXC_SNAP_RPFCH"]
    if not snap_rpf.empty:
        snap_rpf.plot(ax=ax, color='#064E3B', edgecolor='#10B981', linewidth=1.5, alpha=0.85, hatch='\\\\', zorder=4)

    snap_pns = gdf_exclusiones[gdf_exclusiones["id_exclusion"] == "EXC_SNAP_PNS"]
    if not snap_pns.empty:
        snap_pns.plot(ax=ax, color='#065F46', edgecolor='#34D399', linewidth=1.4, alpha=0.85, hatch='//', zorder=4)

    # 4. Cuerpos Hídricos Mayores (Laguna de Colta)
    colta_water = gdf_exclusiones[gdf_exclusiones["id_exclusion"] == "EXC_AGUA_COLTA"]
    if not colta_water.empty:
        colta_water.plot(ax=ax, color='#1E3A8A', edgecolor='#38BDF8', linewidth=1.4, alpha=0.95, zorder=5)

    # 5. Fronteras Cantonales (Línea intermedia destacada)
    gdf_cantones.plot(ax=ax, facecolor='none', edgecolor='#94A3B8', linewidth=1.4, linestyle='-', zorder=6)

    # 6. Límite Provincial Exterior (Línea gruesa institucional)
    gdf_provincia.plot(ax=ax, facecolor='none', edgecolor='#F8FAFC', linewidth=2.6, zorder=7)

    # 7. Resaltado Especial: Parroquia San Andrés (Guano)
    san_andres = gdf_parroquias[gdf_parroquias["parroquia"].str.contains("Andr", case=False, na=False)]
    if not san_andres.empty:
        san_andres.plot(ax=ax, color='#0284C7', edgecolor='#00FFFF', linewidth=2.4, alpha=0.35, zorder=8)
        sa_row = san_andres.iloc[0]
        sa_c = sa_row.geometry.centroid
        ax.annotate(
            "SAN ANDRÉS (GUANO)\n16,037 ha Brutas | 9,180 ha SAN\n★ ÁREA DE PROCESAMIENTO TOTAL ★",
            xy=(sa_c.x, sa_c.y),
            xytext=(35, 25), textcoords="offset points",
            ha='left', va='center',
            fontsize=9.5, fontweight='bold', color='#FFFFFF',
            arrowprops=dict(arrowstyle="->", color='#00FFFF', lw=1.5),
            bbox=dict(boxstyle="round,pad=0.4", fc="#082F49", ec="#00FFFF", alpha=0.95, lw=1.5),
            zorder=12
        )

    # 8. Rotulado Toponímico de los 10 Cantones (Posicionamiento Inteligente de Cartelas)
    canton_offsets = {
        "Riobamba": (10, -5),
        "Guano": (25, 10),
        "Colta": (-30, 0),
        "Alausí": (-20, 15),
        "Chambo": (25, -5),
        "Chunchi": (-15, -15),
        "Guamote": (10, 0),
        "Pallatanga": (-25, 0),
        "Penipe": (20, 10),
        "Cumandá": (-10, -10)
    }

    for _, row in gdf_cantones.iterrows():
        c_name = row["canton"]
        centroid = row.geometry.centroid
        ox, oy = canton_offsets.get(c_name, (0, 0))

        ax.annotate(
            f"CANTÓN {c_name.upper()}",
            xy=(centroid.x, centroid.y),
            xytext=(ox, oy), textcoords="offset points",
            ha='center', va='center',
            fontsize=8.5, fontweight='black', color='#F1F5F9',
            bbox=dict(boxstyle="square,pad=0.25", fc="#0F172A", ec="#64748B", alpha=0.85, lw=1.0),
            zorder=10
        )

    # Título Institucional y Rotulado Técnico
    ax.set_title(
        "ORGANIZACIÓN POLÍTICO-ADMINISTRATIVA Y ÁREAS PROTEGIDAS (SNAP)\nPROVINCIA DE CHIMBORAZO - DPA INEC / SNAP / LADM ISO 19152 (TFM UNIR)",
        fontsize=13.5, fontweight='bold', color='#F8FAFC', pad=20
    )

    ax.set_xlabel("Coordenada Este UTM Zona 17 Sur (m)", fontsize=10, color='#94A3B8', labelpad=10)
    ax.set_ylabel("Coordenada Norte UTM Zona 17 Sur (m)", fontsize=10, color='#94A3B8', labelpad=10)
    ax.tick_params(colors='#94A3B8', labelsize=8.5)
    ax.grid(True, linestyle='--', color='#1E293B', alpha=0.7)

    # Rosa de los Vientos (Norte Cartográfico) ubicada limpiamente en el margen superior derecho
    x_min, y_min, x_max, y_max = gdf_provincia.total_bounds
    arrow_x = x_min + (x_max - x_min) * 0.94
    arrow_y = y_min + (y_max - y_min) * 0.80
    ax.annotate('N', xy=(arrow_x, arrow_y + 8000), xytext=(arrow_x, arrow_y),
                arrowprops=dict(facecolor='#00FFFF', edgecolor='#FFFFFF', width=3.5, headwidth=10),
                ha='center', va='center', fontsize=12, fontweight='bold', color='#FFFFFF', zorder=15)

    # Leyenda Cartográfica de 6 Niveles
    legend_elements = [
        Line2D([0], [0], color='#F8FAFC', lw=2.6, label='Límite Provincial Oficial (Chimborazo: 479,592 ha)'),
        Line2D([0], [0], color='#94A3B8', lw=1.4, label='Límites Cantonales Oficiales (10 Cantones DPA)'),
        mpatches.Patch(facecolor='#1E293B', edgecolor='#334155', lw=0.8, label='Parroquias Rurales Elegibles (45 Parroquias)'),
        mpatches.Patch(facecolor='#450A0A', edgecolor='#B91C1C', lw=1.0, hatch='//', label='Parroquias Urbanas Excluidas (14 Cabeceras)'),
        mpatches.Patch(facecolor='#064E3B', edgecolor='#10B981', lw=1.2, hatch='\\\\', label='SNAP: Reserva Faunística Chimborazo y PN Sangay'),
        mpatches.Patch(facecolor='#1E3A8A', edgecolor='#38BDF8', lw=1.2, label='Cuerpos Hídricos Mayores (Laguna de Colta)'),
        mpatches.Patch(facecolor='#0284C7', edgecolor='#00FFFF', lw=2.0, alpha=0.5, label='Demarcación Objetivo: Parroquia San Andrés (9,180 ha SAN)')
    ]
    ax.legend(handles=legend_elements, loc='lower left', facecolor='#0B0F19', edgecolor='#334155',
              fontsize=8.5, labelcolor='#F8FAFC', framealpha=0.92, borderpad=1.0)

    # Guardar en alta resolución (300 DPI)
    print("Guardando imagen cartográfica a 300 DPI...")
    for out_path in [out_repo, out_rep, out_latex]:
        fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
        print(f"-> Mapa guardado con éxito en: {out_path}")

    plt.close(fig)
    print("Renderizado completado con éxito.")

if __name__ == "__main__":
    generate_political_map()
