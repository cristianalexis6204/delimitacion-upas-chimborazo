"""
TEST PARISH BOUNDARIES AND TERRITORIAL EXCLUSION - CHIMBORAZO (TFM UNIR)
Autor: Cristian Alexis García Pumagualle
Director: Fernando Antonio Rufo Jiménez
"""

import os
import pytest
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon

@pytest.fixture
def carto_data():
    base_tests = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.dirname(base_tests)
    carto_dir = os.path.join(repo_dir, "data", "cartografia")
    
    parish_file = os.path.join(carto_dir, "parroquias_chimborazo_dpa.geojson")
    exclusion_file = os.path.join(carto_dir, "zonas_exclusion_chimborazo.geojson")
    
    assert os.path.exists(parish_file), f"No existe {parish_file}"
    assert os.path.exists(exclusion_file), f"No existe {exclusion_file}"
    
    gdf_p = gpd.read_file(parish_file)
    gdf_e = gpd.read_file(exclusion_file)
    return gdf_p, gdf_e

def test_parish_count_and_types(carto_data):
    gdf_p, _ = carto_data
    assert len(gdf_p) == 59, f"Se esperaban 59 parroquias oficiales en Chimborazo, encontradas: {len(gdf_p)}"
    
    urbanas = gdf_p[gdf_p["tipo"] == "Urbana"]
    rurales = gdf_p[gdf_p["tipo"] == "Rural"]
    
    assert len(urbanas) == 14, f"Se esperaban 14 parroquias urbanas, encontradas: {len(urbanas)}"
    assert len(rurales) == 45, f"Se esperaban 45 parroquias rurales, encontradas: {len(rurales)}"
    
    # Todas las urbanas deben ser no elegibles para UPAs
    assert all(~urbanas["elegible_upa"]), "Las parroquias urbanas no deben ser elegibles para UPAs"
    # Todas las rurales deben ser elegibles
    assert all(rurales["elegible_upa"]), "Las parroquias rurales deben ser elegibles para UPAs"

def test_parish_geometries_validity(carto_data):
    gdf_p, _ = carto_data
    for idx, row in gdf_p.iterrows():
        geom = row.geometry
        assert geom is not None and not geom.is_empty, f"Geometría vacía en parroquia {row['parroquia']}"
        assert geom.is_valid, f"Geometría inválida en parroquia {row['parroquia']}"
        assert geom.geom_type in ["Polygon", "MultiPolygon"], f"Tipo inesperado en {row['parroquia']}: {geom.geom_type}"

def test_exclusion_zones(carto_data):
    _, gdf_e = carto_data
    assert len(gdf_e) >= 4, f"Se esperaban al menos 4 zonas de exclusión, encontradas: {len(gdf_e)}"
    
    ids = set(gdf_e["id_exclusion"])
    assert "EXC_SNAP_RPFCH" in ids, "Falta la Reserva de Producción Faunística Chimborazo"
    assert "EXC_SNAP_PNS" in ids, "Falta el Parque Nacional Sangay"
    assert "EXC_URB_RIOBAMBA" in ids, "Falta el Casco Urbano de Riobamba"
    assert "EXC_AGUA_COLTA" in ids, "Falta la Laguna de Colta"

def test_san_algebra_in_pilot_parishes(carto_data):
    gdf_p, _ = carto_data
    sa = gdf_p[gdf_p["parroquia"] == "San Andrés"]
    assert not sa.empty, "No se encontró San Andrés en el catálogo"
    
    sa_row = sa.iloc[0]
    # En San Andrés, la Reserva Chimborazo debe ocupar > 6000 ha
    assert sa_row["area_excluida_snap_ha"] > 6000.0, f"Exclusión SNAP en San Andrés inesperada: {sa_row['area_excluida_snap_ha']} ha"
    assert sa_row["superficie_agricola_neta_ha"] > 8000.0, f"SAN en San Andrés inesperada: {sa_row['superficie_agricola_neta_ha']} ha"
    
    lican = gdf_p[gdf_p["parroquia"] == "Licán"].iloc[0]
    assert lican["area_excluida_urbana_ha"] > 50.0, f"Exclusión urbana en Licán esperada > 50 ha, obtenida: {lican['area_excluida_urbana_ha']} ha"
