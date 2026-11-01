import os
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as patheffects
from matplotlib.lines import Line2D

plt.switch_backend('Agg')

# Style standards: NeurIPS 2025 / Unified Institutional GIS
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 9.0
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['pdf.fonttype'] = 42

REPO_DIR = r"K:\Mi unidad\CURSOS Y CAPACITACIÓN PERSONAL\MASTER-INTELIGENCIA ARTIFICIAL\Aula Master IA\02_Segundo_Cuatrimestre\09_TFM\delimitacion-upas-chimborazo"
ROOT_TFM = r"K:\Mi unidad\CURSOS Y CAPACITACIÓN PERSONAL\MASTER-INTELIGENCIA ARTIFICIAL\Aula Master IA\02_Segundo_Cuatrimestre\09_TFM"

carto_dir = os.path.join(REPO_DIR, "data", "cartografia")
parish_file = os.path.join(carto_dir, "parroquias_chimborazo_dpa.geojson")
exclusion_file = os.path.join(carto_dir, "zonas_exclusion_chimborazo.geojson")

out_png_repo = os.path.join(REPO_DIR, "outputs", "mapa_zonificacion_parroquial_exclusion_chimborazo.png")
out_png_rep = os.path.join(ROOT_TFM, "06_Codigo_y_Experimentos", "reportes", "figuras", "mapa_zonificacion_parroquial_exclusion_chimborazo.png")
out_png_latex = os.path.join(ROOT_TFM, "03_Fase_2_Elaboracion", "latex", "figuras", "mapa_zonificacion_parroquial_exclusion_chimborazo.png")

for d in [os.path.dirname(out_png_repo), os.path.dirname(out_png_rep), os.path.dirname(out_png_latex)]:
    os.makedirs(d, exist_ok=True)

print("Cargando capas cartográficas oficiales...")
gdf_parroquias = gpd.read_file(parish_file).to_crs(epsg=32717)
gdf_exclusiones = gpd.read_file(exclusion_file).to_crs(epsg=32717)

# Canvas blanco / institucional GIS
fig, ax = plt.subplots(figsize=(11, 12), dpi=300)
fig.patch.set_facecolor('#FFFFFF')
ax.set_facecolor('#F8FAFC')

# 1. Parroquias rurales de fondo (Elegibles para UPAs)
rurales = gdf_parroquias[gdf_parroquias["tipo"] == "Rural"]
rurales.plot(ax=ax, color='#F1F5F9', edgecolor='#94A3B8', linewidth=0.7, alpha=0.95)

# 2. Parroquias urbanas excluidas (Cabeceras cantonales)
urbanas = gdf_parroquias[gdf_parroquias["tipo"] == "Urbana"]
urbanas.plot(ax=ax, color='#FEE2E2', edgecolor='#EF4444', linewidth=1.1, alpha=0.85)

# 3. SNAP Áreas Protegidas (RPF Chimborazo y PN Sangay)
snap_rpf = gdf_exclusiones[gdf_exclusiones["id_exclusion"] == "EXC_SNAP_RPFCH"]
if not snap_rpf.empty:
    snap_rpf.plot(ax=ax, color='#D1FAE5', edgecolor='#10B981', linewidth=1.2, alpha=0.80)

snap_pns = gdf_exclusiones[gdf_exclusiones["id_exclusion"] == "EXC_SNAP_PNS"]
if not snap_pns.empty:
    snap_pns.plot(ax=ax, color='#D1FAE5', edgecolor='#10B981', linewidth=1.2, alpha=0.80)

# 4. Laguna de Colta (Cuerpo de agua)
colta_water = gdf_exclusiones[gdf_exclusiones["id_exclusion"] == "EXC_AGUA_COLTA"]
if not colta_water.empty:
    colta_water.plot(ax=ax, color='#BAE6FD', edgecolor='#0284C7', linewidth=1.2, alpha=0.95)

# 5. Resaltar las 4 Parroquias Piloto (Calpi, Licán, San Andrés, Columbe)
pilotos = gdf_parroquias[gdf_parroquias["parroquia"].isin(["Calpi", "Licán", "San Andrés", "Columbe"])]
pilotos.plot(ax=ax, color='#FEF08A', edgecolor='#B45309', linewidth=2.0, alpha=0.92)

# Etiquetas con halos blancos para las 4 parroquias piloto
offsets = {
    "Calpi": (-18000, 8000),
    "Licán": (12000, 4000),
    "San Andrés": (-22000, 10000),
    "Columbe": (14000, -8000)
}

for idx, row in pilotos.iterrows():
    p_name = row['parroquia']
    pt = row['geometry'].representative_point()
    px, py = pt.x, pt.y
    dx, dy = offsets.get(p_name, (10000, 10000))
    ax.plot(px, py, 'o', color='#B45309', markersize=6.5, markeredgecolor='white', markeredgewidth=1.2, zorder=6)
    ax.annotate(f"Piloto: {p_name}", xy=(px, py), xytext=(px + dx, py + dy),
                fontsize=8.5, fontweight='bold', color='#78350F',
                arrowprops=dict(arrowstyle="->", color='#B45309', lw=1.2),
                bbox=dict(boxstyle='round,pad=0.25', facecolor='#FEF3C7', edgecolor='#D97706', lw=0.8),
                zorder=7)

# Grícula UTM Zona 17S
ax.grid(True, linestyle=':', alpha=0.45, color='#94A3B8')
ax.set_xlabel("Coordenada Este UTM Zona 17 Sur (Metros)", fontsize=9, fontweight='bold')
ax.set_ylabel("Coordenada Norte UTM Zona 17 Sur (Metros)", fontsize=9, fontweight='bold')

# Escala gráfica métrica en UTM
minx, miny, maxx, maxy = gdf_parroquias.total_bounds
ax.set_xlim(minx - 5000, maxx + 5000)
ax.set_ylim(miny - 5000, maxy + 5000)

scale_x = minx + 8000
scale_y = miny + 8000
scale_len_m = 25000 # 25 km
ax.plot([scale_x, scale_x + scale_len_m], [scale_y, scale_y], color='#1E293B', lw=4, solid_capstyle='butt')
ax.plot([scale_x, scale_x + scale_len_m/2], [scale_y, scale_y], color='#DC2626', lw=4, solid_capstyle='butt')
ax.text(scale_x + scale_len_m/2, scale_y + 2500, '25 km', ha='center', va='bottom',
        fontsize=8.5, fontweight='bold', color='#1E293B',
        path_effects=[patheffects.withStroke(linewidth=2.0, foreground='white')])

# Flecha Norte
arrow_x = minx + 12000
arrow_y = maxy - 18000
ax.annotate('N', xy=(arrow_x, arrow_y + 7000), xytext=(arrow_x, arrow_y),
            arrowprops=dict(facecolor='#1E293B', width=2.2, headwidth=7.5),
            ha='center', va='bottom', fontsize=11, fontweight='bold', color='#1E293B',
            path_effects=[patheffects.withStroke(linewidth=2.5, foreground='white')])

ax.set_title("Zonificación Agro-Catastral y Máscaras de Exclusión Territorial en Chimborazo\n(DPA INEC / SNAP MAATE / Catastro LADM ISO 19152)",
             fontsize=10.5, fontweight='bold', pad=12, color='#1E293B')

legend_elements = [
    mpatches.Patch(facecolor='#FEF08A', edgecolor='#B45309', label='Parroquias Piloto de Validación Intensiva (Calpi, Licán, San Andrés, Columbe)'),
    mpatches.Patch(facecolor='#F1F5F9', edgecolor='#94A3B8', label='Parroquias Rurales Elegibles para UPAs Agrícolas (45 Parroquias)'),
    mpatches.Patch(facecolor='#FEE2E2', edgecolor='#EF4444', label='Parroquias Urbanas Excluidas / Cabeceras Cantonales (14 Parroquias)'),
    mpatches.Patch(facecolor='#D1FAE5', edgecolor='#10B981', label='Áreas Protegidas del SNAP (Reserva Faunística Chimborazo y PN Sangay)'),
    mpatches.Patch(facecolor='#BAE6FD', edgecolor='#0284C7', label='Cuerpos de Agua Mayores (Laguna de Colta)')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=8.0, framealpha=0.94, facecolor='white', edgecolor='#CBD5E1',
          title='Zonificación y Restricciones Catastrales', title_fontsize=8.5)

plt.tight_layout()
for target in [out_png_repo, out_png_rep, out_png_latex]:
    plt.savefig(target, dpi=300, bbox_inches='tight')
    print(f"[OK] Guardado mapa unificado: {target}")

plt.close()
print("\n[ÉXITO]: Mapa de exclusiones territoriales convertido a estándar GIS institucional unificado.")
