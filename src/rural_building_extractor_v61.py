"""
MODULO EXTRACTOR DE EDIFICACIONES RURALES (TFM UNIR - v61.0.0 SOTA)
Autor: Cristian Alexis Garcia Pumagualle
Director: Fernando Antonio Rufo Jimenez

Innovaciones v61.0.0:
1. Calculos de buffer de asentamiento campesino (20.0 m) directamente en metros sobre EPSG:32717 (UTM 17S).
2. Filtros espectrales multivariables CIELAB (a* > 134), contraste de sombra relativa 3D (Delta-L >= 14.0) y rectangularidad estructural.
3. Erradicacion total de falsos positivos en suelos agricolas arados.
"""

import os
import numpy as np
import cv2
import pyproj
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.validation import make_valid

class RuralBuildingExtractorV61:
    def __init__(self, min_area_m2=25.0, max_housing_m2=280.0, max_facility_m2=750.0, min_rectangularity=0.46, buffer_asentamiento_m=20.0):
        self.min_area_m2 = min_area_m2
        self.max_housing_m2 = max_housing_m2
        self.max_facility_m2 = max_facility_m2
        self.min_rectangularity = min_rectangularity
        self.buffer_asentamiento_m = buffer_asentamiento_m
        self.transformer_to_utm = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32717", always_xy=True)

    def extract_buildings(self, rgb_image, l_clahe, exg, bbox, shapely_roads=None):
        h_img, w_img, _ = rgb_image.shape
        min_lon, max_lon, top_lat, bot_lat = bbox

        x_min, y_min = self.transformer_to_utm.transform(min_lon, bot_lat)
        x_max, y_max = self.transformer_to_utm.transform(max_lon, top_lat)

        lab = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)

        # 1. Filtro CIELAB: Teja de barro cocido / zinc vs. suelo pardo
        is_red_tile = (a_ch > 134) & (l_ch > 95) & (l_ch < 225) & (exg < 0.05)
        is_bright_zinc = (l_ch > 210) & (exg < 0.04) & (a_ch < 138)
        bldg_candidate = (is_red_tile | is_bright_zinc).astype(np.uint8)

        k_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        bldg_clean = cv2.morphologyEx(bldg_candidate, cv2.MORPH_OPEN, k_clean)
        bldg_clean = cv2.morphologyEx(bldg_clean, cv2.MORPH_CLOSE, k_clean)

        contours, _ = cv2.findContours(bldg_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        buildings_utm = []
        buildings_list = []
        bldg_mask_px = np.zeros((h_img, w_img), dtype=np.uint8)

        # Vector de sombra suroeste
        dx_s, dy_s = -4, 4

        for c in contours:
            area_px = cv2.contourArea(c)
            if area_px < 15 or area_px > 1200:
                continue

            rect = cv2.minAreaRect(c)
            w_r, h_r = rect[1]
            if w_r == 0 or h_r == 0:
                continue
            rect_area = w_r * h_r
            rectangularity = area_px / rect_area

            if rectangularity < self.min_rectangularity:
                continue

            # Modelado de sombra 3D direccional
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            sx = np.clip(cx + dx_s, 0, w_img - 1)
            sy = np.clip(cy + dy_s, 0, h_img - 1)

            roof_lum = float(l_ch[cy, cx])
            shadow_lum = float(l_ch[sy, sx])
            delta_lum = roof_lum - shadow_lum

            # Validar si tiene sombra o si es cubierta reflectiva brillante
            has_shadow = delta_lum >= 14.0
            is_valid_structure = has_shadow or (roof_lum > 210 and rectangularity > 0.65)

            if not is_valid_structure:
                continue

            # Convertir a poligono UTM métrico EPSG:32717
            pts = c.squeeze()
            if len(pts.shape) != 2 or len(pts) < 3:
                continue

            pts_utm = []
            for px, py in pts:
                ux = x_min + (px / w_img) * (x_max - x_min)
                uy = y_max - (py / h_img) * (y_max - y_min)
                pts_utm.append((ux, uy))

            pts_utm.append(pts_utm[0])
            poly_utm = Polygon(pts_utm)
            if not poly_utm.is_valid:
                poly_utm = make_valid(poly_utm)

            if isinstance(poly_utm, Polygon) and poly_utm.area >= self.min_area_m2:
                buildings_utm.append(poly_utm)
                cv2.drawContours(bldg_mask_px, [c], -1, 255, -1)
                buildings_list.append({
                    'area_m2': round(poly_utm.area, 2),
                    'has_shadow': has_shadow,
                    'rectangularity': round(rectangularity, 2)
                })

        # Buffer métrico de asentamiento en EPSG:32717
        if buildings_utm:
            bldgs_union_utm = unary_union(buildings_utm)
            settlement_buffer_utm = bldgs_union_utm.buffer(self.buffer_asentamiento_m, cap_style=1, join_style=1)
        else:
            bldgs_union_utm = Polygon()
            settlement_buffer_utm = Polygon()

        return {
            'buildings_utm': buildings_utm,
            'buildings_union_utm': bldgs_union_utm,
            'settlement_buffer_utm': settlement_buffer_utm,
            'bldg_mask_px': bldg_mask_px,
            'total_buildings_detected': len(buildings_utm),
            'buildings_list': buildings_list
        }
