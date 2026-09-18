"""
Suite de Pruebas Unitarias Automatizadas de Topología Catastral LADM ISO 19152
TFM UNIR - Autor: Cristian Alexis García Pumagualle
"""

import sys
import os
import pytest
from shapely.geometry import Polygon, box, LineString

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from topological_boundary_reconciler import TopologicalBoundaryReconciler

def test_zero_overlap_guarantee():
    """
    Verifica matemáticamente que dos parcelas con solape parcial
    sean reconciliadas con un área de intersección idéntica a 0.000 (estanqueidad LADM).
    """
    reconciler = TopologicalBoundaryReconciler()
    
    # Dos parcelas cuadradas con solape del 10%
    p1 = box(0.0, 0.0, 1.0, 1.0)
    p2 = box(0.9, 0.0, 1.9, 1.0) # Solape en x in [0.9, 1.0]
    
    input_list = [
        {"geometry": p1, "area_ha": 1.0, "num_vertices": 4, "clase": "Cultivo"},
        {"geometry": p2, "area_ha": 1.0, "num_vertices": 4, "clase": "Cultivo"}
    ]
    
    reconciled = reconciler.reconcile_polygons(input_list)
    assert len(reconciled) == 2, "Deben preservarse ambas parcelas"
    
    g1 = reconciled[0]["geometry"]
    g2 = reconciled[1]["geometry"]
    
    inter = g1.intersection(g2)
    assert inter.area == pytest.approx(0.0, abs=1e-7), f"El solape debe ser 0.000%, obtenido: {inter.area}"

def test_large_overlap_preservation_via_voronoi_split():
    """
    Verifica que parcelas con solape sustancial (40%) no sean descartadas,
    sino particionadas equitativamente por la mediatriz geodésica sin solapes.
    """
    reconciler = TopologicalBoundaryReconciler()
    
    # Dos parcelas cuadradas con solape del 40% (x in [0.6, 1.0])
    p1 = box(0.0, 0.0, 1.0, 1.0)
    p2 = box(0.6, 0.0, 1.6, 1.0)
    
    input_list = [
        {"geometry": p1, "area_ha": 1.0, "num_vertices": 4, "clase": "Cultivo"},
        {"geometry": p2, "area_ha": 1.0, "num_vertices": 4, "clase": "Cultivo"}
    ]
    
    reconciled = reconciler.reconcile_polygons(input_list)
    assert len(reconciled) == 2, "Ambas parcelas deben preservarse mediante partición equitativa"
    
    g1 = reconciled[0]["geometry"]
    g2 = reconciled[1]["geometry"]
    
    assert g1.intersection(g2).area == pytest.approx(0.0, abs=1e-7)
    assert 4 <= len(g1.exterior.coords) - 1 <= 10
    assert 4 <= len(g2.exterior.coords) - 1 <= 10

def test_ladm_vertex_bounds():
    """
    Comprueba que el regularizador mantenga los polígonos estrictamente
    dentro del rango registral de 4 a 10 vértices de la norma ISO 19152.
    """
    reconciler = TopologicalBoundaryReconciler()
    
    # Polígono sintético con 24 vértices (ruido o rasterización)
    coords = []
    import math
    for i in range(24):
        angle = 2.0 * math.pi * i / 24.0
        r = 1.0 + 0.1 * math.sin(6.0 * angle)
        coords.append((r * math.cos(angle), r * math.sin(angle)))
    coords.append(coords[0])
    noisy_poly = Polygon(coords)
    
    reg_poly = reconciler.regularize_ladm(noisy_poly, max_v=10, min_v=4)
    n_vert = len(reg_poly.exterior.coords) - 1
    
    assert 4 <= n_vert <= 10, f"Los vértices deben estar en [4, 10], obtenido: {n_vert}"

def test_single_polygon_validity():
    """
    Comprueba que el reconciliador garantice geometrías 100% válidas.
    """
    reconciler = TopologicalBoundaryReconciler()
    p = box(0.0, 0.0, 0.5, 0.5)
    reg = reconciler.regularize_ladm(p, max_v=8, min_v=4)
    assert reg.is_valid
    assert not reg.is_empty
    assert reg.geom_type == "Polygon"
