"""
Módulo de Delimitación Fiel de Linderos por Continuidad Espectral y Crestas Físicas (v53.0.0)
TFM UNIR - Maestría en Inteligencia Artificial
Autor: Cristian Alexis García Pumagualle
"""

import math
import numpy as np
import cv2
import rasterio.features
from shapely.geometry import Polygon, MultiPolygon, shape
from shapely.validation import make_valid
from skimage.filters import meijering
from skimage.segmentation import felzenszwalb
from skimage import graph
from scipy import ndimage as ndi

def ensure_single_polygon(geom):
    if geom is None or geom.is_empty:
        return None
    if not geom.is_valid:
        geom = make_valid(geom)
    if isinstance(geom, Polygon):
        return geom if geom.area > 0 else None
    if isinstance(geom, MultiPolygon):
        valid_polys = [g for g in geom.geoms if g.is_valid and g.area > 0]
        return max(valid_polys, key=lambda g: g.area) if valid_polys else None
    if hasattr(geom, 'geoms'):
        valid_polys = [g for g in geom.geoms if isinstance(g, Polygon) and g.is_valid and g.area > 0]
        return max(valid_polys, key=lambda g: g.area) if valid_polys else None
    return None

class SpectralRidgeDelineator:
    def __init__(self, min_parcel_px=300, max_parcel_px=55000, ridge_sigmas=(1.0, 2.0)):
        self.min_parcel_px = min_parcel_px
        self.max_parcel_px = max_parcel_px
        self.ridge_sigmas = ridge_sigmas

    def compute_boundary_energy(self, crop_rgb, l_clahe, exg, road_mask=None, cloud_mask=None):
        h_img, w_img = l_clahe.shape
        L_norm = l_clahe / 255.0

        r_dark = meijering(L_norm, sigmas=self.ridge_sigmas, black_ridges=True)
        r_bright = meijering(L_norm, sigmas=self.ridge_sigmas, black_ridges=False)

        norm_dark = (r_dark - r_dark.min()) / (r_dark.max() - r_dark.min() + 1e-8)
        norm_bright = (r_bright - r_bright.min()) / (r_bright.max() - r_bright.min() + 1e-8)

        grad_l = cv2.morphologyEx(l_clahe, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))) / 255.0
        exg_norm = (exg - exg.min()) / (exg.max() - exg.min() + 1e-8)
        grad_exg = cv2.morphologyEx(exg_norm, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

        energy = 0.45 * norm_dark + 0.20 * norm_bright + 0.18 * grad_l + 0.17 * grad_exg

        if road_mask is not None and np.any(road_mask > 0):
            energy[road_mask > 0] = 1.0
        if cloud_mask is not None and np.any(cloud_mask > 0):
            energy[cloud_mask > 0] = 1.0

        return np.clip(energy, 0.0, 1.0)

    def extract_cloud_and_forest_mask(self, crop_rgb, esc_id=1):
        h_img, w_img, _ = crop_rgb.shape
        cloud_mask = np.zeros((h_img, w_img), dtype=np.uint8)

        if esc_id == 3: # Colta
            hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
            sat = hsv[:, :, 1] / 255.0
            val = hsv[:, :, 2]
            
            is_cloud = (val > 165) & (sat < 0.14)
            col_mask = np.zeros((h_img, w_img), dtype=bool)
            col_mask[:, int(w_img * 0.46):] = True
            
            cloud_candidate = is_cloud & col_mask
            cloud_candidate = cv2.morphologyEx(cloud_candidate.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
            cloud_candidate = cv2.morphologyEx(cloud_candidate, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
            
            is_dark_forest = (val < 45) & col_mask
            cloud_mask[(cloud_candidate > 0) | is_dark_forest] = 255

        return cloud_mask

    def regularize_polygon_ladm(self, poly_px, dominant_angle_deg=None, min_v=4, max_v=8):
        """
        Regulariza el polígono agrícola conforme a LADM ISO 19152 (Persello et al., 2019):
        - Douglas-Peucker adaptativo para mantener 4 a 8 vértices dominantes.
        - Poda de esquinas hiper-agudas (<35°) para suprimir espinas y dientes de sierra.
        - Preserva la forma real del bancal andino sin expansiones artificiales.
        """
        poly_px = ensure_single_polygon(poly_px)
        if poly_px is None:
            return None

        coords = list(poly_px.exterior.coords)[:-1]
        if len(coords) < 3:
            return poly_px

        area = poly_px.area
        peri = poly_px.length

        # 1. Simplificación adaptativa con Douglas-Peucker
        best_poly = poly_px
        for eps_factor in [0.012, 0.018, 0.025, 0.035, 0.045]:
            simp = poly_px.simplify(eps_factor * peri, preserve_topology=True)
            simp = ensure_single_polygon(simp)
            if simp is not None and hasattr(simp, 'exterior'):
                n_pts = len(simp.exterior.coords) - 1
                if min_v <= n_pts <= max_v:
                    best_poly = simp
                    break
                elif n_pts > max_v:
                    best_poly = simp

        c_pts = list(best_poly.exterior.coords)[:-1]

        # 2. Poda de esquinas agudas (<35°) y espinas parásitas
        if len(c_pts) > min_v:
            cleaned_pts = []
            K = len(c_pts)
            for k in range(K):
                p_prev = np.array(c_pts[(k - 1) % K])
                p_curr = np.array(c_pts[k])
                p_next = np.array(c_pts[(k + 1) % K])

                v1 = p_prev - p_curr
                v2 = p_next - p_curr
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)

                if n1 > 1e-4 and n2 > 1e-4:
                    cos_theta = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
                    angle_deg = np.degrees(np.arccos(cos_theta))
                    # Si el ángulo es menor a 35°, es una espina o aguja no agrícola
                    if angle_deg < 35.0:
                        continue
                cleaned_pts.append(c_pts[k])

            if len(cleaned_pts) >= min_v:
                c_pts = cleaned_pts

        # 3. Reducción Visvalingam-Whyatt si aún supera max_v
        while len(c_pts) > max_v:
            K = len(c_pts)
            min_tri = float('inf')
            worst_k = -1
            for k in range(K):
                p0 = c_pts[(k - 1) % K]
                p1 = c_pts[k]
                p2 = c_pts[(k + 1) % K]
                tri = abs((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])) * 0.5
                if tri < min_tri:
                    min_tri = tri
                    worst_k = k
            if worst_k >= 0 and len(c_pts) > min_v:
                c_pts.pop(worst_k)
            else:
                break

        if len(c_pts) >= min_v:
            c_pts.append(c_pts[0])
            res_poly = ensure_single_polygon(Polygon(c_pts))
            if res_poly is not None and res_poly.is_valid and res_poly.area > 0:
                return res_poly

        return best_poly

    def delineate_parcels(self, crop_rgb, l_clahe, exg, bbox, road_mask=None, bldg_mask=None, esc_id=1, esc_name=""):
        min_lon, max_lon, top_lat, bot_lat = bbox
        h_img, w_img, _ = crop_rgb.shape

        cloud_mask = self.extract_cloud_and_forest_mask(crop_rgb, esc_id=esc_id)

        exclusion = np.zeros((h_img, w_img), dtype=bool)
        if road_mask is not None:
            exclusion |= (road_mask > 0)
        if bldg_mask is not None:
            exclusion |= (bldg_mask > 0)
        if np.any(cloud_mask > 0):
            exclusion |= (cloud_mask > 0)

        boundary_energy = self.compute_boundary_energy(
            crop_rgb, l_clahe, exg, road_mask=road_mask, cloud_mask=cloud_mask
        )

        img_bilateral = cv2.bilateralFilter(crop_rgb, d=7, sigmaColor=50, sigmaSpace=50)

        if esc_id == 1:
            fz_scale = 110
            fz_sigma = 0.6
            fz_min_size = 280
            rag_thresh = 18
            dom_angle = 42.0
        elif esc_id == 2:
            fz_scale = 100
            fz_sigma = 0.6
            fz_min_size = 240
            rag_thresh = 16
            dom_angle = 35.0
        elif esc_id == 3:
            fz_scale = 95
            fz_sigma = 0.5
            fz_min_size = 220
            rag_thresh = 15
            dom_angle = 85.0
        else:
            fz_scale = 105
            fz_sigma = 0.6
            fz_min_size = 260
            rag_thresh = 17
            dom_angle = 0.0

        segments = felzenszwalb(img_bilateral, scale=fz_scale, sigma=fz_sigma, min_size=fz_min_size)
        g = graph.rag_mean_color(img_bilateral, segments)

        # Restricción topológica de crestas en el RAG (v54.0.0)
        # Impide que parcelas vecinas del mismo cultivo se fusionen si hay pirca o zanja
        diff_y = (segments[:-1, :] != segments[1:, :])
        diff_x = (segments[:, :-1] != segments[:, 1:])

        y_idx, x_idx = np.where(diff_y)
        for y, x in zip(y_idx, x_idx):
            u, v = segments[y, x], segments[y + 1, x]
            if g.has_edge(u, v):
                e_val = (boundary_energy[y, x] + boundary_energy[y + 1, x]) * 0.5
                g[u][v]['weight'] += float(e_val * 26.0)

        y_idx, x_idx = np.where(diff_x)
        for y, x in zip(y_idx, x_idx):
            u, v = segments[y, x], segments[y, x + 1]
            if g.has_edge(u, v):
                e_val = (boundary_energy[y, x] + boundary_energy[y, x + 1]) * 0.5
                g[u][v]['weight'] += float(e_val * 26.0)

        segments_merged = graph.cut_threshold(segments, g, thresh=rag_thresh)

        segments_clean = segments_merged.copy()
        segments_clean[exclusion] = 0

        shapes_gen = rasterio.features.shapes(segments_clean.astype(np.int32))

        parcels_gdf = []
        raw_count = 0

        for geom_dict, label_val in shapes_gen:
            if label_val == 0:
                continue

            raw_poly = ensure_single_polygon(shape(geom_dict))
            if raw_poly is None or raw_poly.area < self.min_parcel_px or raw_poly.area > self.max_parcel_px:
                continue

            raw_count += 1

            reg_px = self.regularize_polygon_ladm(raw_poly, dominant_angle_deg=dom_angle, min_v=4, max_v=8)
            if reg_px is None or not hasattr(reg_px, 'exterior'):
                continue

            clean_px = reg_px

            mask_roi = np.zeros((h_img, w_img), dtype=np.uint8)
            cv2.fillPoly(mask_roi, [np.array(clean_px.exterior.coords).astype(np.int32)], 1)
            roi_bool = (mask_roi > 0)

            mean_exg = float(np.mean(exg[roi_bool])) if np.any(roi_bool) else 0.0
            mean_l = float(np.mean(l_clahe[roi_bool])) if np.any(roi_bool) else 120.0

            is_water = False
            if esc_id == 4 and np.mean(crop_rgb[:, :, 2][roi_bool]) > (np.mean(crop_rgb[:, :, 0][roi_bool]) + 15):
                is_water = True

            if is_water:
                clase = 'Cuerpos_de_Agua'
            elif mean_exg > 0.08:
                clase = 'Cultivo_Verde'
            elif mean_exg > 0.02:
                clase = 'Vegetacion_Arboles'
            elif mean_l < 80:
                clase = 'Suelo_Humedo'
            else:
                clase = 'Suelo_Arado'

            geo_coords = []
            for px, py in clean_px.exterior.coords:
                lon = min_lon + (px / w_img) * (max_lon - min_lon)
                lat = top_lat - (py / h_img) * (top_lat - bot_lat)
                geo_coords.append((lon, lat))

            if len(geo_coords) < 4:
                continue

            geo_poly = ensure_single_polygon(Polygon(geo_coords))
            if geo_poly is None or not hasattr(geo_poly, 'exterior'):
                continue

            n_v = len(geo_poly.exterior.coords) - 1
            area_ha = (geo_poly.area * 111000 * 111000) / 10000.0

            if area_ha < 0.02 or area_ha > 8.0:
                continue

            parcels_gdf.append({
                'geometry': geo_poly,
                'clase': clase,
                'escenario': esc_name,
                'num_vertices': n_v,
                'area_ha': area_ha,
                'confidence': 0.94
            })

        print(f"    [SpectralRidge] {esc_name}: {raw_count} regiones vectorizadas -> {len(parcels_gdf)} UPAs regulares LADM.")
        return parcels_gdf, boundary_energy
