"""
BUILD PARISH AND EXCLUSION CARTOGRAPHY - CHIMBORAZO (TFM UNIR)
Autor: Cristian Alexis García Pumagualle
Director: Fernando Antonio Rufo Jiménez

Este script compila y genera:
1. data/cartografia/parroquias_chimborazo_dpa.geojson (Polígonos y atributos de las 59 parroquias)
2. data/cartografia/zonas_exclusion_chimborazo.geojson (Polígonos de exclusión: SNAP, Urbanos y Cuerpos de Agua)
3. data/cartografia/chimborazo_parish_catalog.json (Catálogo estadístico y de zonificación territorial)
"""

import os
import sys
import json
import time
import geopandas as gpd
import pandas as pd
from shapely.geometry import box, Polygon, MultiPolygon
from shapely.ops import unary_union
import shapely
import osmnx as ox

# Diccionario canónico de las 59 parroquias de Chimborazo por Cantón (INEC DPA)
CANTONES_CHIMBORAZO = {
    "Riobamba": {
        "urbanas": ["Lizarzaburu", "Maldonado", "Velasco", "Veloz", "Yaruquíes"],
        "rurales": ["Cacha", "Calpi", "Cubijíes", "Flores", "Licán", "Licto", "Pungalá", "Punín", "Químiag", "San Juan", "San Luis"]
    },
    "Guano": {
        "urbanas": ["El Rosario"],
        "rurales": ["Guanando", "Ilapo", "La Providencia", "San Andrés", "San Gerardo de Pacaicaguán", "San Isidro de Patulú", "San José del Chazo", "Santa Fe de Galán", "Valparaíso"]
    },
    "Colta": {
        "urbanas": ["Cajabamba"],
        "rurales": ["Cañi", "Columbe", "Juan de Velasco", "Santiago de Quito"]
    },
    "Alausí": {
        "urbanas": ["Alausí"],
        "rurales": ["Achupallas", "Guasuntos", "Huigra", "Multitud", "Pistishí", "Pumallacta", "Sevilla", "Sibambe", "Tixán"]
    },
    "Chambo": {
        "urbanas": ["Chambo"],
        "rurales": []
    },
    "Chunchi": {
        "urbanas": ["Chunchi"],
        "rurales": ["Capzol", "Compud", "Gonzol", "Llagos"]
    },
    "Guamote": {
        "urbanas": ["Guamote"],
        "rurales": ["Cebadas", "Palmira"]
    },
    "Pallatanga": {
        "urbanas": ["Pallatanga"],
        "rurales": []
    },
    "Penipe": {
        "urbanas": ["Penipe"],
        "rurales": ["Bilbao", "El Altar", "La Candelaria", "Matus", "Puela", "San Antonio de Bayushig"]
    },
    "Cumandá": {
        "urbanas": ["Cumandá"],
        "rurales": []
    }
}

# 4 Parroquias Piloto de validación intensiva
PILOTO_PARROQUIAS = ["Calpi", "Licán", "San Andrés", "Columbe"]

def get_cartografia_dir():
    base_src = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.dirname(base_src)
    carto_dir = os.path.join(repo_dir, "data", "cartografia")
    os.makedirs(carto_dir, exist_ok=True)
    return carto_dir

def build_exclusion_zones(carto_dir):
    print("\n========================================================")
    print(" 1. COMPILANDO ZONAS DE EXCLUSIÓN TERRITORIAL (SNAP, URBANO, AGUA)")
    print("========================================================")
    
    exclusion_file = os.path.join(carto_dir, "zonas_exclusion_chimborazo.geojson")
    
    exclusiones = [
        {
            "id": "EXC_SNAP_RPFCH",
            "nombre": "Reserva de Producción Faunística Chimborazo",
            "categoria": "SNAP_Area_Protegida",
            "query": "Reserva de Producción Faunística Chimborazo",
            "descripcion": "Páramo altoandino, glaciares y hábitat de vicuñas (>3800 msnm). Exclusión agrícola legal."
        },
        {
            "id": "EXC_SNAP_PNS",
            "nombre": "Parque Nacional Sangay",
            "categoria": "SNAP_Area_Protegida",
            "query": "Parque Nacional Sangay, Ecuador",
            "descripcion": "Cordillera oriental y ceja de montaña. Conservación estricta y bosque protector."
        },
        {
            "id": "EXC_URB_RIOBAMBA",
            "nombre": "Casco Urbano Consolidado de Riobamba",
            "categoria": "Zona_Urbana_Consolidada",
            "query": "Riobamba, Ecuador",
            "descripcion": "Mancha urbana central, predios habitacionales y viales asfaltados. Sin UPAs agrícolas."
        },
        {
            "id": "EXC_AGUA_COLTA",
            "nombre": "Laguna de Colta",
            "categoria": "Cuerpo_Hidrico_Mayor",
            "query": "Laguna de Colta, Chimborazo, Ecuador",
            "descripcion": "Cuerpo lacustre natural de la cuenca de Colta."
        }
    ]
    
    features = []
    for exc in exclusiones:
        print(f"-> Descargando / geocodificando: {exc['nombre']}...")
        try:
            gdf = ox.geocode_to_gdf(exc["query"])
            geom = gdf.geometry.iloc[0]
            if geom.geom_type not in ["Polygon", "MultiPolygon"]:
                geom = geom.buffer(0.001)
            
            # Si es zona urbana, agregar buffer de 100 m (~0.0009 grados)
            if exc["categoria"] == "Zona_Urbana_Consolidada":
                geom_utm = gpd.GeoSeries([geom], crs="EPSG:4326").to_crs(epsg=32717).iloc[0].buffer(100.0)
                geom = gpd.GeoSeries([geom_utm], crs="EPSG:32717").to_crs(epsg=4326).iloc[0]
            
            features.append({
                "id_exclusion": exc["id"],
                "nombre": exc["nombre"],
                "categoria": exc["categoria"],
                "descripcion": exc["descripcion"],
                "geometry": geom
            })
            print(f"   [OK] {exc['nombre']} ({geom.geom_type})")
        except Exception as e:
            print(f"   [AVISO] No se pudo obtener {exc['nombre']} vía OSM ({e}). Creando aproximación geométrica.")
            # Fallback seguro para Laguna de Colta si la consulta directa falla
            if "Colta" in exc["nombre"]:
                poly = box(-78.765, -1.735, -78.745, -1.710)
                features.append({
                    "id_exclusion": exc["id"],
                    "nombre": exc["nombre"],
                    "categoria": exc["categoria"],
                    "descripcion": exc["descripcion"],
                    "geometry": poly
                })
        time.sleep(1)
        
    gdf_exclusiones = gpd.GeoDataFrame(features, crs="EPSG:4326")
    gdf_exclusiones.to_file(exclusion_file, driver="GeoJSON")
    print(f"-> Guardado exitoso: {exclusion_file} ({len(gdf_exclusiones)} zonas de exclusión)")
    return gdf_exclusiones

def build_parish_cartography(carto_dir, gdf_exclusiones):
    print("\n========================================================")
    print(" 2. COMPILANDO LÍMITES PARROQUIALES DPA Y SUPERFICIE AGRÍCOLA NETA")
    print("========================================================")
    
    parish_file = os.path.join(carto_dir, "parroquias_chimborazo_dpa.geojson")
    catalog_file = os.path.join(carto_dir, "chimborazo_parish_catalog.json")
    
    # Preparar unión de exclusión en UTM 17S
    gdf_excl_utm = gdf_exclusiones.to_crs(epsg=32717)
    union_exclusion_utm = unary_union(gdf_excl_utm.geometry)
    
    # Unión separada para SNAP y Urbano para desglose analítico
    snap_geom_utm = unary_union(gdf_excl_utm[gdf_excl_utm["categoria"] == "SNAP_Area_Protegida"].geometry)
    urb_geom_utm = unary_union(gdf_excl_utm[gdf_excl_utm["categoria"] == "Zona_Urbana_Consolidada"].geometry)
    
    parish_features = []
    catalog_entries = []
    
    total_parroquias = sum(len(c["urbanas"]) + len(c["rurales"]) for c in CANTONES_CHIMBORAZO.values())
    curr_idx = 0
    
    for canton_nom, data_c in CANTONES_CHIMBORAZO.items():
        # 1. Parroquias Urbanas
        for p_urb in data_c["urbanas"]:
            curr_idx += 1
            print(f"[{curr_idx}/{total_parroquias}] Procesando Urbana: {p_urb} (Cantón {canton_nom})...")
            # Para parroquias urbanas, se busca con contexto o se asigna polígono aproximado si está dentro de la cabecera
            try:
                gdf_p = ox.geocode_to_gdf(f"{p_urb}, {canton_nom}, Chimborazo, Ecuador")
                geom = gdf_p.geometry.iloc[0]
            except Exception:
                try:
                    gdf_p = ox.geocode_to_gdf(f"{p_urb}, Chimborazo, Ecuador")
                    geom = gdf_p.geometry.iloc[0]
                except Exception:
                    geom = None
            
            parish_features.append({
                "canton": canton_nom,
                "parroquia": p_urb,
                "tipo": "Urbana",
                "elegible_upa": False,
                "motivo_exclusion": "Cabecera Cantonal / Núcleo Urbano Consolidado",
                "es_piloto": False,
                "geometry": geom if geom is not None else box(-78.68, -1.68, -78.64, -1.64)
            })
            
            catalog_entries.append({
                "canton": canton_nom,
                "parroquia": p_urb,
                "tipo": "Urbana",
                "elegible_upa": False,
                "motivo_exclusion": "Cabecera Cantonal / Núcleo Urbano Consolidado",
                "es_piloto": False,
                "area_total_ha": 0.0,
                "area_excluida_snap_ha": 0.0,
                "area_excluida_urbana_ha": 0.0,
                "superficie_agricola_neta_ha": 0.0
            })
            time.sleep(0.5)
            
        # 2. Parroquias Rurales
        for p_rur in data_c["rurales"]:
            curr_idx += 1
            is_piloto = p_rur in PILOTO_PARROQUIAS
            print(f"[{curr_idx}/{total_parroquias}] Procesando Rural: {p_rur} (Cantón {canton_nom}) {'[PILOTO]' if is_piloto else ''}...")
            
            geom = None
            for q in [f"{p_rur}, {canton_nom}, Chimborazo, Ecuador", f"{p_rur}, Chimborazo, Ecuador"]:
                try:
                    gdf_p = ox.geocode_to_gdf(q)
                    geom = gdf_p.geometry.iloc[0]
                    break
                except Exception:
                    continue
                    
            if geom is None:
                print(f"   [AVISO] Usando centroide aproximado para {p_rur}...")
                geom = box(-78.75, -1.70, -78.70, -1.65)
                
            # Reproyectar a UTM 17S para métricas exactas
            geom_utm = gpd.GeoSeries([geom], crs="EPSG:4326").to_crs(epsg=32717).iloc[0]
            area_tot_ha = geom_utm.area / 10000.0
            
            # Cálculo de áreas de exclusión
            inter_snap = geom_utm.intersection(snap_geom_utm) if snap_geom_utm and not snap_geom_utm.is_empty else None
            area_snap_ha = (inter_snap.area / 10000.0) if inter_snap and not inter_snap.is_empty else 0.0
            
            inter_urb = geom_utm.intersection(urb_geom_utm) if urb_geom_utm and not urb_geom_utm.is_empty else None
            area_urb_ha = (inter_urb.area / 10000.0) if inter_urb and not inter_urb.is_empty else 0.0
            
            # Superficie Agrícola Neta (SAN)
            diff_utm = geom_utm.difference(union_exclusion_utm)
            area_san_ha = diff_utm.area / 10000.0 if not diff_utm.is_empty else 0.0
            
            pct_excluida = ((area_tot_ha - area_san_ha) / area_tot_ha * 100.0) if area_tot_ha > 0 else 0.0
            print(f"   -> {p_rur}: Tot={area_tot_ha:.1f} ha | SNAP={area_snap_ha:.1f} ha | Urb={area_urb_ha:.1f} ha | SAN={area_san_ha:.1f} ha ({100-pct_excluida:.1f}% agro)")
            
            parish_features.append({
                "canton": canton_nom,
                "parroquia": p_rur,
                "tipo": "Rural",
                "elegible_upa": True,
                "motivo_exclusion": "Ninguno (Elegible para UPAs)",
                "es_piloto": is_piloto,
                "area_total_ha": round(area_tot_ha, 2),
                "area_excluida_snap_ha": round(area_snap_ha, 2),
                "area_excluida_urbana_ha": round(area_urb_ha, 2),
                "superficie_agricola_neta_ha": round(area_san_ha, 2),
                "geometry": geom
            })
            
            catalog_entries.append({
                "canton": canton_nom,
                "parroquia": p_rur,
                "tipo": "Rural",
                "elegible_upa": True,
                "es_piloto": is_piloto,
                "area_total_ha": round(area_tot_ha, 2),
                "area_excluida_snap_ha": round(area_snap_ha, 2),
                "area_excluida_urbana_ha": round(area_urb_ha, 2),
                "superficie_agricola_neta_ha": round(area_san_ha, 2),
                "porcentaje_agricola_neto": round(100.0 - pct_excluida, 1)
            })
            time.sleep(0.5)

    gdf_parroquias = gpd.GeoDataFrame(parish_features, crs="EPSG:4326")
    gdf_parroquias.to_file(parish_file, driver="GeoJSON")
    print(f"\n-> Guardado exitoso de límites parroquiales: {parish_file}")
    
    catalog_data = {
        "provincia": "Chimborazo",
        "codigo_dpa": "06",
        "total_cantones": len(CANTONES_CHIMBORAZO),
        "total_parroquias": total_parroquias,
        "parroquias_urbanas_excluidas": sum(len(c["urbanas"]) for c in CANTONES_CHIMBORAZO.values()),
        "parroquias_rurales_elegibles": sum(len(c["rurales"]) for c in CANTONES_CHIMBORAZO.values()),
        "parroquias_piloto_evaluadas": PILOTO_PARROQUIAS,
        "zonas_exclusion_ambientales": ["Reserva de Producción Faunística Chimborazo", "Parque Nacional Sangay", "Laguna de Colta"],
        "parroquias": catalog_entries
    }
    
    with open(catalog_file, "w", encoding="utf-8") as f:
        json.dump(catalog_data, f, indent=2, ensure_ascii=False)
    print(f"-> Guardado exitoso de catálogo: {catalog_file}")

if __name__ == "__main__":
    c_dir = get_cartografia_dir()
    excl_gdf = build_exclusion_zones(c_dir)
    build_parish_cartography(c_dir, excl_gdf)
