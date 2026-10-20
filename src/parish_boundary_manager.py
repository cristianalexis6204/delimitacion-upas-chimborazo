"""
PARISH BOUNDARY MANAGER - CHIMBORAZO (TFM UNIR)
Autor: Cristian Alexis García Pumagualle
Director: Fernando Antonio Rufo Jiménez

Administra la cartografía oficial DPA de las parroquias de Chimborazo,
aplica las máscaras de exclusión territorial (SNAP, Zonas Urbanas, Cuerpos de Agua)
y calcula la Superficie Agrícola Neta (SAN) bajo el estándar LADM ISO 19152.
"""

import os
import json
import geopandas as gpd
import pandas as pd
from shapely.geometry import box, Polygon, MultiPolygon
from shapely.ops import unary_union
import shapely
import osmnx as ox

class ParishBoundaryManager:
    def __init__(self, carto_dir=None):
        if carto_dir is None:
            base_src = os.path.dirname(os.path.abspath(__file__))
            repo_dir = os.path.dirname(base_src)
            carto_dir = os.path.join(repo_dir, "data", "cartografia")
        self.carto_dir = carto_dir
        self.parish_file = os.path.join(carto_dir, "parroquias_chimborazo_dpa.geojson")
        self.exclusion_file = os.path.join(carto_dir, "zonas_exclusion_chimborazo.geojson")
        self.catalog_file = os.path.join(carto_dir, "chimborazo_parish_catalog.json")

        self.gdf_parroquias = None
        self.gdf_exclusiones = None
        self.catalog = None
        self._load_cartography()

    def _load_cartography(self):
        # 1. Cargar exclusiones
        if os.path.exists(self.exclusion_file):
            try:
                self.gdf_exclusiones = gpd.read_file(self.exclusion_file)
            except Exception as e:
                print(f"[Aviso] No se pudo leer {self.exclusion_file}: {e}")

        # 2. Cargar parroquias
        if os.path.exists(self.parish_file):
            try:
                self.gdf_parroquias = gpd.read_file(self.parish_file)
            except Exception as e:
                print(f"[Aviso] No se pudo leer {self.parish_file}: {e}")

        # 3. Cargar catálogo
        if os.path.exists(self.catalog_file):
            try:
                with open(self.catalog_file, "r", encoding="utf-8") as f:
                    self.catalog = json.load(f)
            except Exception as e:
                print(f"[Aviso] No se pudo leer {self.catalog_file}: {e}")

    def _get_exclusion_union_utm(self):
        if self.gdf_exclusiones is not None and len(self.gdf_exclusiones) > 0:
            excl_utm = self.gdf_exclusiones.to_crs(epsg=32717)
            return unary_union(excl_utm.geometry)
        else:
            # Fallback seguro con geocodificación al vuelo
            excl_list = []
            for q in ["Reserva de Producción Faunística Chimborazo", "Parque Nacional Sangay, Ecuador", "Riobamba, Ecuador"]:
                try:
                    g = ox.geocode_to_gdf(q).to_crs(epsg=32717)
                    excl_list.append(g.geometry.iloc[0])
                except Exception:
                    pass
            return unary_union(excl_list) if excl_list else None

    def get_parish_data(self, parish_name):
        """
        Devuelve la geometría oficial, desglose de exclusiones y Superficie Agrícola Neta (SAN)
        de una parroquia dada.
        """
        # Normalizar nombre
        p_clean = parish_name.strip()
        
        geom_wgs = None
        canton = "Riobamba"
        tipo = "Rural"
        
        # Buscar en GeoDataFrame local
        if self.gdf_parroquias is not None:
            matches = self.gdf_parroquias[self.gdf_parroquias["parroquia"].str.lower() == p_clean.lower()]
            if len(matches) > 0:
                row = matches.iloc[0]
                geom_wgs = row.geometry
                canton = row.get("canton", canton)
                tipo = row.get("tipo", tipo)
                
        # Si no se encontró en GeoDataFrame local, consultar al vuelo vía Nominatim/OSM
        if geom_wgs is None:
            queries = [
                f"{p_clean}, Chimborazo, Ecuador",
                f"{p_clean}, Ecuador"
            ]
            for q in queries:
                try:
                    g = ox.geocode_to_gdf(q)
                    geom_wgs = g.geometry.iloc[0]
                    break
                except Exception:
                    pass

        if geom_wgs is None:
            raise ValueError(f"No se pudo encontrar la geometría oficial de la parroquia: {parish_name}")

        # Reproyectar a UTM Zona 17 Sur (EPSG:32717) para cálculo exacto en metros y hectáreas
        gdf_p_utm = gpd.GeoSeries([geom_wgs], crs="EPSG:4326").to_crs(epsg=32717)
        geom_utm = gdf_p_utm.iloc[0]
        area_total_ha = geom_utm.area / 10000.0

        # Obtener máscara de exclusión
        union_excl_utm = self._get_exclusion_union_utm()

        # Calcular desgloses de exclusión específicos
        area_snap_ha = 0.0
        area_urb_ha = 0.0
        area_agua_ha = 0.0

        if self.gdf_exclusiones is not None:
            excl_utm = self.gdf_exclusiones.to_crs(epsg=32717)
            
            snap_geoms = excl_utm[excl_utm["categoria"] == "SNAP_Area_Protegida"].geometry
            if not snap_geoms.empty:
                inter_snap = geom_utm.intersection(unary_union(snap_geoms))
                area_snap_ha = inter_snap.area / 10000.0 if not inter_snap.is_empty else 0.0
                
            urb_geoms = excl_utm[excl_utm["categoria"] == "Zona_Urbana_Consolidada"].geometry
            if not urb_geoms.empty:
                inter_urb = geom_utm.intersection(unary_union(urb_geoms))
                area_urb_ha = inter_urb.area / 10000.0 if not inter_urb.is_empty else 0.0
                
            agua_geoms = excl_utm[excl_utm["categoria"] == "Cuerpo_Hidrico_Mayor"].geometry
            if not agua_geoms.empty:
                inter_agua = geom_utm.intersection(unary_union(agua_geoms))
                area_agua_ha = inter_agua.area / 10000.0 if not inter_agua.is_empty else 0.0

        # Calcular Superficie Agrícola Neta (SAN): Parroquia \ Exclusiones
        if union_excl_utm is not None:
            geom_san_utm = geom_utm.difference(union_excl_utm)
        else:
            geom_san_utm = geom_utm

        area_san_ha = geom_san_utm.area / 10000.0 if not geom_san_utm.is_empty else 0.0
        pct_san = (area_san_ha / area_total_ha * 100.0) if area_total_ha > 0 else 0.0

        # Convertir SAN a WGS84 para exportaciones KML y visores
        geom_san_wgs = gpd.GeoSeries([geom_san_utm], crs="EPSG:32717").to_crs(epsg=4326).iloc[0]

        return {
            "parroquia": p_clean,
            "canton": canton,
            "tipo": tipo,
            "area_total_ha": round(area_total_ha, 2),
            "area_total_km2": round(area_total_ha / 100.0, 2),
            "area_excl_snap_ha": round(area_snap_ha, 2),
            "area_excl_urbana_ha": round(area_urb_ha, 2),
            "area_excl_agua_ha": round(area_agua_ha, 2),
            "superficie_agricola_neta_ha": round(area_san_ha, 2),
            "superficie_agricola_neta_km2": round(area_san_ha / 100.0, 2),
            "porcentaje_agricola_neto": round(pct_san, 1),
            "porcentaje_excluido": round(100.0 - pct_san, 1),
            "geometry_wgs84": geom_wgs,
            "geometry_utm": geom_utm,
            "geometry_san_wgs84": geom_san_wgs,
            "geometry_san_utm": geom_san_utm,
            "bounds_utm": geom_san_utm.bounds,
            "bounds_wgs84": geom_san_wgs.bounds
        }

    def get_pilot_parishes_data(self):
        """
        Devuelve el análisis consolidado de las 4 parroquias piloto seleccionadas:
        Calpi, Licán, San Andrés y Columbe.
        """
        pilotos = ["Calpi", "Licán", "San Andrés", "Columbe"]
        results = []
        for p in pilotos:
            try:
                data = self.get_parish_data(p)
                results.append(data)
            except Exception as e:
                print(f"[Error evaluando {p}]: {e}")
        return results

if __name__ == "__main__":
    mgr = ParishBoundaryManager()
    print("\n========================================================")
    print(" EVALUACIÓN DE PARROQUIAS PILOTO Y SUPERFICIE AGRÍCOLA NETA")
    print("========================================================")
    pilotos = mgr.get_pilot_parishes_data()
    for p in pilotos:
        print(f"\n* Parroquia: {p['parroquia']} (Cantón {p['canton']})")
        print(f"  - Superficie Total: {p['area_total_ha']:.2f} ha ({p['area_total_km2']:.2f} km²)")
        print(f"  - Exclusión SNAP (Reserva Chimborazo / Sangay): {p['area_excl_snap_ha']:.2f} ha")
        print(f"  - Exclusión Urbana (Huella consolidada): {p['area_excl_urbana_ha']:.2f} ha")
        print(f"  - Superficie Agrícola Neta (SAN): {p['superficie_agricola_neta_ha']:.2f} ha ({p['porcentaje_agricola_neto']}%)")
