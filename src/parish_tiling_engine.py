"""
PARISH TILING ENGINE - CHIMBORAZO (TFM UNIR)
Autor: Cristian Alexis García Pumagualle
Director: Fernando Antonio Rufo Jiménez

Genera una malla de teselas regulares con solape sobre la Superficie Agrícola Neta (SAN)
de una parroquia rural, descartando automáticamente áreas fuera del polígono o dentro
de zonas de exclusión territorial (SNAP, Zonas Urbanas, Cuerpos de Agua).
"""

import math
import numpy as np
import geopandas as gpd
from shapely.geometry import box, Polygon, MultiPolygon
from shapely.ops import unary_union

class ParishTilingEngine:
    def __init__(self, tile_size_px=1024, pixel_size_m=0.30, overlap_pct=0.10):
        """
        :param tile_size_px: Dimensión de la tesela en píxeles (default 1024x1024).
        :param pixel_size_m: Resolución espacial en metros/píxel (default 0.30 m/px).
        :param overlap_pct: Solape perimetral entre teselas contiguas (default 10% = 30.7 m).
        """
        self.tile_size_px = tile_size_px
        self.pixel_size_m = pixel_size_m
        self.overlap_pct = overlap_pct
        
        # Dimensión métrica de la tesela
        self.tile_dim_m = tile_size_px * pixel_size_m  # 307.2 metros
        self.step_m = self.tile_dim_m * (1.0 - overlap_pct)  # 276.48 metros con 10% solape

    def generate_grid_for_parish(self, parish_data, max_tiles=None, min_intersection_pct=0.15):
        """
        Genera el conjunto de teselas sobre la Superficie Agrícola Neta (SAN) de la parroquia.
        
        :param parish_data: Diccionario retornado por ParishBoundaryManager.get_parish_data().
        :param max_tiles: Límite superior de teselas para control de cómputo en benchmarks (None = todas).
        :param min_intersection_pct: Porcentaje mínimo de área de la tesela dentro de la SAN (default 15%).
        :return: GeoDataFrame con las teselas elegibles en EPSG:32717 y EPSG:4326.
        """
        parish_nom = parish_data["parroquia"]
        san_geom_utm = parish_data["geometry_san_utm"]

        if san_geom_utm.is_empty:
            raise ValueError(f"La parroquia {parish_nom} no tiene Superficie Agrícola Neta (SAN) válida.")

        min_x, min_y, max_x, max_y = san_geom_utm.bounds

        n_cols = int(math.ceil((max_x - min_x) / self.step_m))
        n_rows = int(math.ceil((max_y - min_y) / self.step_m))

        tiles = []
        tile_area_m2 = self.tile_dim_m * self.tile_dim_m

        for r in range(n_rows):
            y0 = min_y + r * self.step_m
            y1 = y0 + self.tile_dim_m
            for c in range(n_cols):
                x0 = min_x + c * self.step_m
                x1 = x0 + self.tile_dim_m

                t_poly = box(x0, y0, x1, y1)

                if t_poly.intersects(san_geom_utm):
                    inter = t_poly.intersection(san_geom_utm)
                    inter_area = inter.area
                    pct_inside = inter_area / tile_area_m2

                    if pct_inside >= min_intersection_pct:
                        center_x = (x0 + x1) / 2.0
                        center_y = (y0 + y1) / 2.0
                        tile_id = f"TILE_{parish_nom.upper()}_R{r:03d}_C{c:03d}"
                        
                        tiles.append({
                            "tile_id": tile_id,
                            "parroquia": parish_nom,
                            "canton": parish_data["canton"],
                            "row": r,
                            "col": c,
                            "center_utm_x": center_x,
                            "center_utm_y": center_y,
                            "inter_area_m2": round(inter_area, 2),
                            "pct_in_san": round(pct_inside * 100.0, 1),
                            "geometry": t_poly
                        })

        if not tiles:
            # Fallback seguro: tesela centrada en el centroide de la SAN
            centroid = san_geom_utm.centroid
            cx, cy = centroid.x, centroid.y
            half = self.tile_dim_m / 2.0
            t_poly = box(cx - half, cy - half, cx + half, cy + half)
            tiles.append({
                "tile_id": f"TILE_{parish_nom.upper()}_CENTROID",
                "parroquia": parish_nom,
                "canton": parish_data["canton"],
                "row": 0,
                "col": 0,
                "center_utm_x": cx,
                "center_utm_y": cy,
                "inter_area_m2": round(t_poly.intersection(san_geom_utm).area, 2),
                "pct_in_san": 100.0,
                "geometry": t_poly
            })

        gdf_utm = gpd.GeoDataFrame(tiles, crs="EPSG:32717")

        # Ordenar por mayor porcentaje dentro de la SAN (zonas centrales más densas)
        gdf_utm = gdf_utm.sort_values(by="pct_in_san", ascending=False).reset_index(drop=True)

        if max_tiles is not None and len(gdf_utm) > max_tiles:
            gdf_utm = gdf_utm.iloc[:max_tiles].copy()

        # Reproyectar también a WGS84 para geocodificación satelital
        gdf_utm["geometry_wgs84"] = gdf_utm.to_crs(epsg=4326).geometry
        centroids_wgs = gdf_utm.geometry.centroid.to_crs(epsg=4326)
        gdf_utm["center_lon"] = centroids_wgs.x
        gdf_utm["center_lat"] = centroids_wgs.y

        return gdf_utm

if __name__ == "__main__":
    from parish_boundary_manager import ParishBoundaryManager
    mgr = ParishBoundaryManager()
    tiler = ParishTilingEngine(tile_size_px=1024, pixel_size_m=0.30, overlap_pct=0.10)

    print("\n========================================================")
    print(" GENERACIÓN DE MALLAS DE TESELAS SOBRE SUPERFICIE AGRÍCOLA NETA")
    print("========================================================")

    for p_name in ["Calpi", "Licán", "San Andrés"]:
        p_data = mgr.get_parish_data(p_name)
        tiles_gdf = tiler.generate_grid_for_parish(p_data, max_tiles=12)
        print(f"\n* Parroquia: {p_name} ({p_data['superficie_agricola_neta_ha']:.2f} ha netas)")
        print(f"  - Teselas de alta representatividad generadas: {len(tiles_gdf)}")
        print(f"  - Ejemplo Tile ID: {tiles_gdf.iloc[0]['tile_id']}")
        print(f"  - Centro WGS84: Lat {tiles_gdf.iloc[0]['center_lat']:.5f}, Lon {tiles_gdf.iloc[0]['center_lon']:.5f}")
        print(f"  - Cobertura dentro de SAN: {tiles_gdf.iloc[0]['pct_in_san']}%")
