import os
import sqlite3
import json
import geopandas as gpd
from shapely.ops import unary_union
from shapely.validation import make_valid

exp_v100_dir = r"K:\Mi unidad\CURSOS Y CAPACITACIÓN PERSONAL\MASTER-INTELIGENCIA ARTIFICIAL\Aula Master IA\02_Segundo_Cuatrimestre\09_TFM\06_Codigo_y_Experimentos\experimentos\v100.0.0"
gpkg_path = os.path.join(exp_v100_dir, "upas_san_andres_v100_a100.gpkg")

print(f"Inspeccionando GPKG via SQLite3: {gpkg_path}")
conn = sqlite3.connect(gpkg_path)
cursor = conn.cursor()
cursor.execute("SELECT table_name, data_type, identifier FROM gpkg_contents")
layers_info = cursor.fetchall()

summary = {}

for t, dt, ident in layers_info:
    try:
        gdf = gpd.read_file(gpkg_path, layer=t)
        cnt = len(gdf)
        geom_types = gdf.geometry.type.value_counts().to_dict()
        valid_cnt = int(gdf.is_valid.sum())
        empty_cnt = int(gdf.is_empty.sum())
        
        # Proyectar a UTM 17S (EPSG:32717) para calculos metricos exactos
        gdf_utm = gdf.to_crs(epsg=32717) if gdf.crs is not None else gdf
        
        total_area_ha = 0.0
        total_len_km = 0.0
        if any("Polygon" in str(k) for k in geom_types.keys()):
            total_area_ha = float(gdf_utm.geometry.area.sum() / 10000.0)
        if any("LineString" in str(k) for k in geom_types.keys()):
            total_len_km = float(gdf_utm.geometry.length.sum() / 1000.0)
            
        summary[t] = {
            "count": cnt,
            "valid": valid_cnt,
            "empty": empty_cnt,
            "geom_types": {str(k): int(v) for k, v in geom_types.items()},
            "total_area_ha": round(total_area_ha, 4),
            "total_len_km": round(total_len_km, 4)
        }
        print(f"  [OK] '{t}': {cnt} elementos | Validos: {valid_cnt}/{cnt} | Area: {total_area_ha:.2f} ha | Longitud: {total_len_km:.2f} km")
    except Exception as e:
        print(f"  [ERROR leyendo '{t}']: {e}")

# Leer agroindustria clasificada si existe
try:
    cursor.execute('SELECT clase, count(*) FROM "estructuras_agroindustriales" GROUP BY clase')
    agro_counts = cursor.fetchall()
    summary["desglose_agroindustria"] = {cl: cnt for cl, cnt in agro_counts}
    print("\nEstructuras Agroindustriales:")
    for cl, cnt in agro_counts:
        print(f"  * {cl}: {cnt}")
except Exception as e:
    print(f"Aviso lectura agroindustria: {e}")

conn.close()

# Comprobaciones deterministas LADM
print("\n--- AUDITORIA DETERMINISTA TOPOLOGICA LADM ISO 19152 ---")
gdf_upas = gpd.read_file(gpkg_path, layer="upas_agricolas_puras_circulo").to_crs(epsg=32717)
gdf_circ = gpd.read_file(gpkg_path, layer="perimetros_cuadrantes_circulares").to_crs(epsg=32717)

geoms = [make_valid(g) if not g.is_valid else g for g in gdf_upas.geometry.values]

sindex = gdf_upas.sindex
total_overlap_area = 0.0
overlap_pairs = 0

for i in range(len(geoms)):
    p_i = geoms[i]
    if p_i is None or p_i.is_empty:
        continue
    possible = list(sindex.intersection(p_i.bounds))
    for j in possible:
        if j > i:
            p_j = geoms[j]
            if p_j is None or p_j.is_empty:
                continue
            if p_i.intersects(p_j):
                inter = p_i.intersection(p_j)
                if inter.area > 0.001:
                    total_overlap_area += inter.area
                    overlap_pairs += 1

total_area = gdf_upas.geometry.area.sum()
overlap_pct = (total_overlap_area / total_area) * 100.0 if total_area > 0 else 0.0

circ_union = unary_union(gdf_circ.geometry)
upas_outside = gdf_upas.geometry.difference(circ_union)
spill_area = upas_outside.area.sum()

print(f"Total UPAs: {len(gdf_upas)}")
print(f"Pares con solape: {overlap_pairs}")
print(f"Area total de solape: {total_overlap_area:.6f} m2 ({overlap_pct:.6f}%)")
print(f"Derrame exterior fuera de R=350m: {spill_area:.6f} m2")

summary["auditoria_ladm_topologica"] = {
    "total_upas": len(gdf_upas),
    "pares_solape": overlap_pairs,
    "area_solape_m2": round(total_overlap_area, 6),
    "porcentaje_solape": round(overlap_pct, 6),
    "derrame_exterior_m2": round(spill_area, 6),
    "conformidad_ladm_iso_19152": "100.0%" if overlap_pairs == 0 and spill_area < 0.01 else "FALIBLE"
}

out_json = os.path.join(exp_v100_dir, "sonda_gis_v100_determinista.json")
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(f"\nSonda determinista guardada en: {out_json}")
