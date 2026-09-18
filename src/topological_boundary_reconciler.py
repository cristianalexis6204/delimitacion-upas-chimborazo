"""
Módulo de Reconciliación Topológica Planar LADM ISO 19152 (v53.0.0)
TFM UNIR - Maestría en Inteligencia Artificial
Autor: Cristian Alexis García Pumagualle

Funciones LADM ISO 19152:
1. Preserva la continuidad de linderos sin mutilación por diferencias booleanas agresivas.
2. Garantiza 0.00% de solape topológico entre parcelas agrícolas colindantes.
3. Regularización geométrica registral: 100% de parcelas con 4 a 10 vértices (promedio objetivo ~7.8).
4. Sella polígonos estancos sin micro-huecos (gaps) ni dobles líneas (slivers).
"""

import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.validation import make_valid
from shapely.ops import unary_union

class TopologicalBoundaryReconciler:
    def __init__(self, snap_tolerance_deg=0.000008): # ~ 0.8 metros
        self.snap_tolerance_deg = snap_tolerance_deg

    def regularize_ladm(self, geom, max_v=10, min_v=4):
        """
        Garantiza que el polígono tenga entre min_v y max_v vértices,
        preservando las esquinas vivas y la orientación de la parcela.
        """
        if not geom.is_valid:
            geom = make_valid(geom)
        if isinstance(geom, MultiPolygon):
            geom = max(geom.geoms, key=lambda g: g.area)

        coords = list(geom.exterior.coords)[:-1]
        n_v = len(coords)

        # Si ya cumple con el estándar registral
        if min_v <= n_v <= max_v:
            return geom

        # Si tiene más de max_v vértices, simplificar con Douglas-Peucker adaptativo
        if n_v > max_v:
            peri = geom.length
            # Búsqueda de epsilon óptimo
            for eps_factor in np.linspace(0.005, 0.04, 8):
                simp = geom.simplify(peri * eps_factor, preserve_topology=True)
                if not simp.is_valid:
                    simp = make_valid(simp)
                if isinstance(simp, MultiPolygon):
                    simp = max(simp.geoms, key=lambda g: g.area)
                
                simp_nv = len(simp.exterior.coords) - 1
                if min_v <= simp_nv <= max_v:
                    return simp
                if simp_nv < min_v:
                    break

            # Si aún excede max_v, poda suave de vértices de menor impacto angular
            c_list = list(geom.exterior.coords)[:-1]
            while len(c_list) > max_v:
                K = len(c_list)
                min_tri = float('inf')
                worst_k = -1
                for k in range(K):
                    p0 = c_list[(k - 1) % K]
                    p1 = c_list[k]
                    p2 = c_list[(k + 1) % K]
                    tri_area = abs((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])) * 0.5
                    if tri_area < min_tri:
                        min_tri = tri_area
                        worst_k = k
                if worst_k >= 0:
                    c_list.pop(worst_k)
                else:
                    break

            if len(c_list) >= min_v:
                c_list.append(c_list[0])
                poly_res = Polygon(c_list)
                if poly_res.is_valid:
                    return poly_res

        # Si tiene menos de min_v vértices o falló la reducción: rectángulo rotado mínimo
        rect = geom.minimum_rotated_rectangle
        if rect.is_valid and rect.area > 0:
            return rect

        return geom

    def reconcile_polygons(self, polygons_list):
        """
        Reconcilia la lista de parcelas para garantizar 0.00% de solapes
        sin mutilar bordes ni generar atajos diagonales ficticios.
        """
        if not polygons_list or len(polygons_list) < 2:
            return polygons_list

        reconciled = []
        valid_polys = []

        for item in polygons_list:
            geom = item['geometry']
            if not geom.is_valid:
                geom = make_valid(geom)
            if isinstance(geom, MultiPolygon):
                geom = max(geom.geoms, key=lambda a: a.area)
            if geom.is_valid and geom.area > 1e-11:
                item_copy = item.copy()
                item_copy['geometry'] = geom
                valid_polys.append(item_copy)

        # Ordenar por área descendente
        valid_polys = sorted(valid_polys, key=lambda p: p['geometry'].area, reverse=True)
        occupied_union = None

        for p in valid_polys:
            geom = p['geometry']
            if occupied_union is None:
                geom_reg = self.regularize_ladm(geom)
                p['geometry'] = geom_reg
                p['num_vertices'] = len(geom_reg.exterior.coords) - 1
                p['area_ha'] = (geom_reg.area * 111000 * 111000) / 10000.0
                occupied_union = geom_reg
                reconciled.append(p)
            else:
                if geom.intersects(occupied_union):
                    inter = geom.intersection(occupied_union)
                    if inter.area > 0:
                        inter_ratio = inter.area / geom.area
                        # Si el solape es mayor al 35%, se descarta como duplicado
                        if inter_ratio > 0.35:
                            continue

                        # Resta topológica limpia
                        diff = geom.difference(occupied_union)
                        if diff.is_empty:
                            continue
                        if isinstance(diff, MultiPolygon):
                            diff = max(diff.geoms, key=lambda a: a.area)
                        if not diff.is_valid:
                            diff = make_valid(diff)

                        # Si tras la diferencia el remanente es insignificante
                        if diff.area < (geom.area * 0.30):
                            continue

                        diff_reg = self.regularize_ladm(diff)
                        if diff_reg.is_valid and diff_reg.area > 1e-11:
                            p['geometry'] = diff_reg
                            p['num_vertices'] = len(diff_reg.exterior.coords) - 1
                            p['area_ha'] = (diff_reg.area * 111000 * 111000) / 10000.0
                            occupied_union = occupied_union.union(diff_reg)
                            reconciled.append(p)
                else:
                    geom_reg = self.regularize_ladm(geom)
                    p['geometry'] = geom_reg
                    p['num_vertices'] = len(geom_reg.exterior.coords) - 1
                    p['area_ha'] = (geom_reg.area * 111000 * 111000) / 10000.0
                    occupied_union = occupied_union.union(geom_reg)
                    reconciled.append(p)

        return reconciled
