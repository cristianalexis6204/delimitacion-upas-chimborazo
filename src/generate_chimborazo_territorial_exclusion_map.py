"""
GENERATE CHIMBORAZO TERRITORIAL EXCLUSION MAP (TFM UNIR)
Autor: Cristian Alexis García Pumagualle
Director: Fernando Antonio Rufo Jiménez

Genera el mapa cartográfico oficial de alta resolución (300 DPI)
de la zonificación territorial provincial de Chimborazo:
- 45 Parroquias Rurales Elegibles para UPAs agrícolas.
- 14 Parroquias Urbanas / Cabeceras Cantonales Excluidas.
- Áreas Protegidas del SNAP (Reserva Faunística Chimborazo y PN Sangay).
- Cuerpos de agua mayores (Laguna de Colta).
- Resaltado perimetral de las 4 Parroquias Piloto (Calpi, Licán, San Andrés, Columbe).
"""

import os
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

def generate_territorial_map():
    base_src = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.dirname(base_src)
    root_tfm = os.path.abspath(os.path.join(repo_dir, ".."))
    
    carto_dir = os.path.join(repo_dir, "data", "cartografia")
    parish_file = os.path.join(carto_dir, "parroquias_chimborazo_dpa.geojson")
    exclusion_file = os.path.join(carto_dir, "zonas_exclusion_chimborazo.geojson")
    
    out_png_repo = os.path.join(repo_dir, "outputs", "mapa_zonificacion_parroquial_exclusion_chimborazo.png")
    out_png_rep = os.path.join(root_tfm, "06_Codigo_y_Experimentos", "reportes", "figuras", "mapa_zonificacion_parroquial_exclusion_chimborazo.png")
    out_png_latex = os.path.join(root_tfm, "03_Fase_2_Elaboracion", "latex", "figuras", "mapa_zonificacion_parroquial_exclusion_chimborazo.png")

    for d in [os.path.dirname(out_png_repo), os.path.dirname(out_png_rep), os.path.dirname(out_png_latex)]:
        os.makedirs(d, exist_ok=True)

    print("Cargando capas cartográficas...")
    gdf_parroquias = gpd.read_file(parish_file).to_crs(epsg=32717)
    gdf_exclusiones = gpd.read_file(exclusion_file).to_crs(epsg=32717)

    fig, ax = plt.subplots(figsize=(12, 14), dpi=300)
    fig.patch.set_facecolor('#0B0F19')
    ax.set_facecolor('#0F172A')

    # 1. Plot parroquias rurales de fondo
    rurales = gdf_parroquias[gdf_parroquias["tipo"] == "Rural"]
    rurales.plot(ax=ax, color='#1E293B', edgecolor='#475569', linewidth=0.8, alpha=0.9)

    # 2. Plot parroquias urbanas excluidas
    urbanas = gdf_parroquias[gdf_parroquias["tipo"] == "Urbana"]
    urbanas.plot(ax=ax, color='#7F1D1D', edgecolor='#EF4444', linewidth=1.2, alpha=0.85, hatch='//')

    # 3. Plot SNAP Áreas Protegidas
    snap_rpf = gdf_exclusiones[gdf_exclusiones["id_exclusion"] == "EXC_SNAP_RPFCH"]
    if not snap_rpf.empty:
        snap_rpf.plot(ax=ax, color='#064E3B', edgecolor='#10B981', linewidth=1.5, alpha=0.75, hatch='\\\\')

    snap_pns = gdf_exclusiones[gdf_exclusiones["id_exclusion"] == "EXC_SNAP_PNS"]
    if not snap_pns.empty:
        snap_pns.plot(ax=ax, color='#065F46', edgecolor='#34D399', linewidth=1.5, alpha=0.75, hatch='\\\\')

    # 4. Plot Laguna de Colta
    colta_water = gdf_exclusiones[gdf_exclusiones["id_exclusion"] == "EXC_AGUA_COLTA"]
    if not colta_water.empty:
        colta_water.plot(ax=ax, color='#1E3A8A', edgecolor='#3B82F6', linewidth=1.5, alpha=0.9)

    # 5. Resaltar las 4 Parroquias Piloto de validación intensiva
    pilotos = gdf_parroquias[gdf_parroquias["parroquia"].isin(["Calpi", "Licán", "San Andrés", "Columbe"])]
    pilotos.plot(ax=ax, color='#1E3A5F', edgecolor='#38BDF8', linewidth=2.2, alpha=0.95)

    # Etiquetas de las parroquias piloto con offsets inteligentes para evitar solapes
    offsets = {
        "Calpi": (-45, 10),
        "Licán": (45, -12),
        "San Andrés": (0, 18),
        "Columbe": (0, 0)
    }

    for _, row in pilotos.iterrows():
        centroid = row.geometry.centroid
        nom = row["parroquia"]
        ha = row.get("superficie_agricola_neta_ha", row.get("area_total_ha", 0))
        ox, oy = offsets.get(nom, (0, 0))
        ax.annotate(f"{nom}\n({ha:,.0f} ha SAN)",
                    xy=(centroid.x, centroid.y),
                    xytext=(ox, oy), textcoords="offset points",
                    ha='center', va='center',
                    fontsize=9, fontweight='bold', color='#F8FAFC',
                    bbox=dict(boxstyle="round,pad=0.35", fc="#0284C7", ec="#38BDF8", alpha=0.9, lw=1.2))

    # Título y rotulado técnico
    ax.set_title("ZONIFICACIÓN TERRITORIAL Y MÁSCARAS DE EXCLUSIÓN AGRO-CATASTRAL\nPROVINCIA DE CHIMBORAZO - DPA INEC / SNAP / LADM ISO 19152 (TFM UNIR)",
                 fontsize=13, fontweight='bold', color='#F8FAFC', pad=18)

    ax.set_xlabel("Coordenada Este UTM Zona 17 Sur (m)", fontsize=10, color='#94A3B8', labelpad=10)
    ax.set_ylabel("Coordenada Norte UTM Zona 17 Sur (m)", fontsize=10, color='#94A3B8', labelpad=10)
    ax.tick_params(colors='#94A3B8', labelsize=8)
    ax.grid(True, linestyle='--', color='#334155', alpha=0.5)

    # Leyenda técnica
    legend_elements = [
        mpatches.Patch(facecolor='#1E3A5F', edgecolor='#38BDF8', linewidth=2, label='Parroquias Piloto de Inferencia (Calpi, Licán, San Andrés, Columbe)'),
        mpatches.Patch(facecolor='#1E293B', edgecolor='#475569', linewidth=0.8, label='Parroquias Rurales Elegibles para UPAs (45 Parroquias DPA)'),
        mpatches.Patch(facecolor='#064E3B', edgecolor='#10B981', linewidth=1.5, hatch='\\\\', label='Áreas Protegidas SNAP (Reserva Faunística Chimborazo y PN Sangay)'),
        mpatches.Patch(facecolor='#7F1D1D', edgecolor='#EF4444', linewidth=1.2, hatch='//', label='Zonas Urbanas Consolidadas Excluidas (14 Cabeceras Cantonales)'),
        mpatches.Patch(facecolor='#1E3A8A', edgecolor='#3B82F6', linewidth=1.5, label='Cuerpos Hídricos Mayores (Laguna de Colta)')
    ]
    
    leg = ax.legend(handles=legend_elements, loc='lower right', frameon=True,
                    facecolor='#0F172A', edgecolor='#334155', fontsize=8.5)
    for text in leg.get_texts():
        text.set_color('#F1F5F9')

    # Subtítulo con nota metodológica
    plt.figtext(0.5, 0.02,
                "Nota Metodológica: Las áreas bajo el Sistema Nacional de Áreas Protegidas (SNAP) y las huellas urbanas consolidadas son sustraídas formalmente\n"
                "mediante álgebra espacial booleana de la Superficie Agrícola Neta (SAN) para erradicar falsos positivos en zonas ecológicamente protegidas.",
                ha='center', fontsize=8, color='#94A3B8', style='italic')

    plt.tight_layout(rect=[0, 0.04, 1, 0.98])
    
    for p in [out_png_repo, out_png_rep, out_png_latex]:
        plt.savefig(p, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
        print(f"-> Guardado mapa en: {p}")
        
    plt.close()
    print("Mapa de zonificación territorial generado con éxito.")

if __name__ == "__main__":
    generate_territorial_map()
