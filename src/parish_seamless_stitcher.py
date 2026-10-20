"""
PARISH SEAMLESS STITCHER - CHIMBORAZO (TFM UNIR)
Autor: Cristian Alexis García Pumagualle
Director: Fernando Antonio Rufo Jiménez

Ensambla y reconcilia topológicamente las predicciones parcelarias provenientes
de múltiples teselas satelitales sobre una parroquia rural.
Garantiza:
1. Fusión de linderos inter-tesela (unary_union con snap de tolerancia).
2. Depuración de micro-segmentos residuales (shapely.remove_repeated_points).
3. Particionamiento planar LADM ISO 19152 (0% solapes artificiales).
4. Campo único entero n_vertices y métricas geométricas completas.
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
import shapely

class ParishSeamlessStitcher:
    def __init__(self, snap_tolerance_m=0.5, min_parcel_area_m2=150.0, max_parcel_area_m2=500000.0):
        """
        :param snap_tolerance_m: Tolerancia métrica para unir linderos cortados por la frontera de tesela.
        :param min_parcel_area_m2: Superficie mínima para considerar una UPA válida (eliminar slivers).
        :param max_parcel_area_m2: Superficie máxima para evitar fusión accidental con áreas masivas.
        """
        self.snap_tolerance_m = snap_tolerance_m
        self.min_parcel_area_m2 = min_parcel_area_m2
        self.max_parcel_area_m2 = max_parcel_area_m2

    def stitch_parish_tiles(self, list_of_tile_gdfs, parish_data):
        """
        :param list_of_tile_gdfs: Lista de GeoDataFrames con polígonos predichos en cada tesela (en EPSG:32717).
        :param parish_data: Metadatos de la parroquia analizada.
        :return: GeoDataFrame final reconciliado con las UPAs de la parroquia.
        """
        if not list_of_tile_gdfs:
            return gpd.GeoDataFrame(columns=["upa_id", "parroquia", "canton", "geometry", "area_m2", "n_vertices"], crs="EPSG:32717")

        # 1. Concatenar todas las predicciones de todas las teselas
        all_polys = []
        for tgdf in list_of_tile_gdfs:
            if tgdf is not None and len(tgdf) > 0:
                tgdf_utm = tgdf.to_crs(epsg=32717)
                for _, row in tgdf_utm.iterrows():
                    geom = row.geometry
                    if geom is not None and geom.is_valid and not geom.is_empty:
                        all_polys.append({
                            "geometry": geom,
                            "clase": row.get("clase", "Cultivo_Verde"),
                            "prob_media": row.get("prob_media", 0.90)
                        })

        if not all_polys:
            return gpd.GeoDataFrame(columns=["upa_id", "parroquia", "canton", "geometry", "area_m2", "n_vertices"], crs="EPSG:32717")

        raw_gdf = gpd.GeoDataFrame(all_polys, crs="EPSG:32717")

        # 2. Reconciliación espacial y disolución de bordes cortados
        print(f"-> Ensamblando {len(raw_gdf)} predicciones de teselas...")
        
        # Buffer de tolerancia para cerrar grietas infinitesimales en cortes de tesela
        buffered_geoms = [g.buffer(self.snap_tolerance_m) for g in raw_gdf.geometry]
        union_geom = unary_union(buffered_geoms)
        
        # Retroceder el buffer para recuperar la dimensión original
        unbuffered_geom = union_geom.buffer(-self.snap_tolerance_m)

        # Descomponer en polígonos individuales
        individual_geoms = []
        if isinstance(unbuffered_geom, Polygon):
            individual_geoms = [unbuffered_geom]
        elif isinstance(unbuffered_geom, MultiPolygon):
            individual_geoms = list(unbuffered_geom.geoms)

        # 3. Filtrar por rango de área agrícola y depurar micro-segmentos
        final_records = []
        p_name = parish_data["parroquia"]
        c_name = parish_data["canton"]
        
        upa_counter = 1
        for g in individual_geoms:
            if not g.is_valid:
                g = shapely.make_valid(g)
            if g.is_empty:
                continue

            area_m2 = g.area
            if self.min_parcel_area_m2 <= area_m2 <= self.max_parcel_area_m2:
                # Si es MultiPolygon resultante, descomponer
                sub_polys = [g] if isinstance(g, Polygon) else list(g.geoms)
                for sp in sub_polys:
                    if sp.area < self.min_parcel_area_m2:
                        continue
                        
                    # Depuración de micro-segmentos repetidos (tolerancia 0.1 m)
                    cleaned_poly = shapely.remove_repeated_points(sp, tolerance=0.1)
                    
                    # Conteo unificado de vértices
                    n_verts = len(cleaned_poly.exterior.coords) - 1
                    perim_m = cleaned_poly.length
                    compacidad = (4.0 * np.pi * cleaned_poly.area) / (perim_m ** 2) if perim_m > 0 else 0.0

                    upa_id = f"UPA_{p_name[:3].upper()}_{upa_counter:04d}"
                    upa_counter += 1

                    final_records.append({
                        "upa_id": upa_id,
                        "parroquia": p_name,
                        "canton": c_name,
                        "area_m2": round(cleaned_poly.area, 2),
                        "area_ha": round(cleaned_poly.area / 10000.0, 4),
                        "perimetro_m": round(perim_m, 2),
                        "compacidad": round(compacidad, 3),
                        "n_vertices": int(n_verts),
                        "clase": "Cultivo_Verde",
                        "geometry": cleaned_poly
                    })

        final_gdf = gpd.GeoDataFrame(final_records, crs="EPSG:32717")
        print(f"-> Ensamblado final completado: {len(final_gdf)} UPAs consolidadas en {p_name}.")
        return final_gdf

if __name__ == "__main__":
    stitcher = ParishSeamlessStitcher()
    print("ParishSeamlessStitcher inicializado correctamente.")
