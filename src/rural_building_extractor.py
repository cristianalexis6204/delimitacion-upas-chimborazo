"""
MÓDULO EXTRACTOR DE EDIFICACIONES Y ASENTAMIENTOS RURALES (TFM UNIR - v51.0.0)
Autor: Cristian Alexis García Pumagualle

Fundamentación Científica y Catastral (LADM ISO 19152 / RENAGRO / LPIS):
- Segregación estricta entre LA_SpatialUnit_Building (Viviendas y Galpones) y LA_SpatialUnit_Parcel (UPAs Agrícolas).
- Detección Bi-Nivel adaptada a la Arquitectura Campesina Andina:
  1. Nivel 1 (Vivienda Familiar Campesina): 25 m² <= Área <= 280 m², rectangularidad >= 0.46, aspecto <= 4.0.
  2. Nivel 2 (Galpón / Tinglado Agropecuario): 280 m² < Área <= 750 m², zinc brillante reflectante en R, G, B simultáneo.
- Verificación de Sombra Solar Co-ocurrente (Sun-Azimuth Shadowing): confirmación 3D de elevación de cubierta.
- Regularización Ortogonal Registral a 90° conforme a la norma catastral ISO 19152.
- Desagregación Morfológica mediante Transformada de Distancia Euclidiana y Watershed para desarticular casas adosadas.
- Contraste perimetral Delta-L para blindaje absoluto contra suelos arados y parcelas agrícolas desnudas.
- Blindaje de Verdor (Vegetation Invariance): ExG interior < 0.02.
- Máscara Espectral Anti-Nubes y Niebla Difusa andina.
- Zonificación de Caserío Consolidado (LADM Settlement Zone): exclusión de patios urbanos no agrarios.
"""

import numpy as np
import cv2
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.ops import unary_union

class RuralBuildingExtractor:
    def __init__(self, min_area_m2=25.0, max_housing_m2=280.0, max_facility_m2=750.0, min_rectangularity=0.46):
        self.min_area_m2 = min_area_m2
        self.max_housing_m2 = max_housing_m2
        self.max_facility_m2 = max_facility_m2
        self.min_rectangularity = min_rectangularity
        self.deg_per_meter = 1.0 / 111000.0

    def extract_buildings(self, rgb_image, l_clahe, exg, bbox):
        """
        Detecta y georreferencia viviendas y galpones rurales mediante espectro cuádruple,
        desagregación por watershed, verificación de sombra solar y regularización ortogonal a 90°.
        """
        h_img, w_img, _ = rgb_image.shape
        min_lon, max_lon, top_lat, bot_lat = bbox
        
        m_per_px_x = (abs(max_lon - min_lon) * 111000.0) / w_img
        m_per_px_y = (abs(top_lat - bot_lat) * 111000.0) / h_img
        m2_per_px = m_per_px_x * m_per_px_y

        r = rgb_image[:, :, 0].astype(float)
        g = rgb_image[:, :, 1].astype(float)
        b = rgb_image[:, :, 2].astype(float)
        brightness = (r + g + b) / 3.0
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        sat = hsv[:, :, 1].astype(float) / 255.0
        val = hsv[:, :, 2].astype(float)

        # 1. MÁSCARA ESTRICTA ANTI-NUBES Y NIEBLA DIFUSA
        cloud_cand = (brightness > 165) & (sat < 0.12) & (r > 155) & (g > 155) & (b > 145)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cloud_cand.astype(np.uint8))
        cloud_mask = np.zeros((h_img, w_img), dtype=bool)
        for lbl in range(1, num_labels):
            if stats[lbl, cv2.CC_STAT_AREA] * m2_per_px > 450.0:
                cloud_mask[labels == lbl] = True
        cloud_mask_dil = cv2.dilate(cloud_mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))) > 0

        # 2. FIRMAS ESPECTRALES CON BLINDAJE DE VERDOR
        non_veg = (exg < 0.02) & (~cloud_mask_dil)

        # A) Zinc brillante / Losa clara
        mask_zinc = ((brightness > 165) | (l_clahe > 180)) & non_veg

        # B) Teja tradicional andina (Arcilla / Ladrillo cocido)
        red_ratio = (r - b) / (r + b + 1e-5)
        rg_ratio = r / (g + 1e-5)
        mask_tile = (red_ratio > 0.16) & (rg_ratio > 1.10) & (brightness > 60) & (brightness < 175) & non_veg

        # C) Chapa azul / celestes / esmaltados
        blue_ratio = (b - r) / (b + r + 1e-5)
        mask_blue = (blue_ratio > 0.05) & (b > g) & (brightness > 85) & (brightness < 185) & non_veg

        # D) Adobe tradicional / Bloque gris / Fibrocemento con relieve
        grad_mag = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        mask_gray = (sat < 0.20) & (val > 105) & (val < 175) & (grad_mag > 18) & non_veg

        combined_mask = (mask_zinc | mask_tile | mask_blue | mask_gray).astype(np.uint8) * 255
        kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        bldg_clean = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_small)
        bldg_clean = cv2.morphologyEx(bldg_clean, cv2.MORPH_OPEN, kernel_small)

        # 3. DESAGREGACIÓN MORFOLÓGICA POR WATERSHED SOBRE TECHOS ADOSADOS
        num_cc, cc_labels, cc_stats, _ = cv2.connectedComponentsWithStats(bldg_clean)
        separated_mask = np.zeros_like(bldg_clean)

        for lbl in range(1, num_cc):
            area_lbl_px = cc_stats[lbl, cv2.CC_STAT_AREA]
            area_lbl_m2 = area_lbl_px * m2_per_px
            if area_lbl_m2 < 20.0:
                continue

            roi_mask = (cc_labels == lbl).astype(np.uint8)

            if 80.0 < area_lbl_m2 <= 400.0:
                dist = cv2.distanceTransform(roi_mask, cv2.DIST_L2, 5)
                max_d = dist.max()
                if max_d > 2.5:
                    peaks = (dist > 0.44 * max_d).astype(np.uint8)
                    n_peaks, p_labels = cv2.connectedComponents(peaks)
                    if n_peaks > 2:
                        markers = p_labels.copy() + 1
                        markers[roi_mask == 0] = 0
                        dist_8u = np.clip((dist / (max_d + 1e-5)) * 255, 0, 255).astype(np.uint8)
                        dist_3ch = cv2.cvtColor(255 - dist_8u, cv2.COLOR_GRAY2BGR)
                        cv2.watershed(dist_3ch, markers)
                        for p_id in range(2, n_peaks + 1):
                            sub_mask = (markers == p_id).astype(np.uint8) * 255
                            sub_mask = cv2.morphologyEx(sub_mask, cv2.MORPH_OPEN, kernel_small)
                            separated_mask[sub_mask > 0] = 255
                        continue

            separated_mask[roi_mask > 0] = 255

        # 4. MÁSCARA DE SOMBRAS SOLARES (SHADOW VERIFICATION)
        shadow_mask = (l_clahe < 75) & (brightness < 70)

        contours, _ = cv2.findContours(separated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        building_polygons = []
        negative_prompt_px = []
        building_mask_px = np.zeros((h_img, w_img), dtype=np.uint8)

        for cnt in contours:
            area_px = cv2.contourArea(cnt)
            area_m2 = area_px * m2_per_px
            if area_m2 < self.min_area_m2:
                continue

            rect = cv2.minAreaRect(cnt)
            rect_w, rect_h = rect[1]
            if rect_w <= 0 or rect_h <= 0:
                continue

            aspect = max(rect_w, rect_h) / min(rect_w, rect_h)
            rect_area_px = rect_w * rect_h
            rectangularity = area_px / rect_area_px

            c_mask = np.zeros((h_img, w_img), dtype=np.uint8)
            cv2.drawContours(c_mask, [cnt], -1, 255, -1)

            # Blindaje de verdor interior
            mean_poly_exg = np.mean(exg[c_mask > 0])
            if mean_poly_exg >= 0.02:
                continue

            # Verificación de sombra solar en el cuadrante suroeste (dx=-3, dy=+3)
            M_shift = np.float32([[1, 0, -3], [0, 1, 3]])
            shifted_mask = cv2.warpAffine(c_mask, M_shift, (w_img, h_img))
            shadow_roi = (shifted_mask > 0) & (c_mask == 0)
            has_shadow = np.sum(shadow_roi & shadow_mask) > 3

            # Contraste perimetral Delta-L con anillo circundante exterior
            c_dil = cv2.dilate(c_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
            ring_mask = (c_dil > 0) & (c_mask == 0)
            mean_inner_l = np.mean(l_clahe[c_mask > 0])
            mean_ring_l = np.mean(l_clahe[ring_mask > 0]) if np.sum(ring_mask) > 0 else mean_inner_l
            edge_contrast = abs(mean_inner_l - mean_ring_l)

            is_building = False
            bldg_type = None

            # Nivel 1: Vivienda Campesina Tradicional (25 m² a 280 m²)
            if self.min_area_m2 <= area_m2 <= self.max_housing_m2:
                if rectangularity >= self.min_rectangularity and aspect <= 4.0:
                    if area_m2 > 180.0:
                        if (edge_contrast >= 10.0 or has_shadow) and rectangularity >= 0.54:
                            is_building = True
                            bldg_type = 'Vivienda_Campesina'
                    else:
                        is_building = True
                        bldg_type = 'Vivienda_Campesina'

            # Nivel 2: Galpón / Tinglado Agropecuario / Nave Rural (280 m² a 750 m²)
            elif self.max_housing_m2 < area_m2 <= self.max_facility_m2:
                mean_r = np.mean(r[c_mask > 0])
                mean_g = np.mean(g[c_mask > 0])
                mean_b = np.mean(b[c_mask > 0])
                is_zinc_metal = (mean_r > 165) and (mean_g > 165) and (mean_b > 160) and (mean_poly_exg < 0.0)
                if is_zinc_metal and rectangularity >= 0.52 and aspect <= 4.5:
                    is_building = True
                    bldg_type = 'Galpon_Agropecuario'

            if is_building:
                M = cv2.moments(cnt)
                if M['m00'] > 0:
                    cx_px = int(M['m10'] / M['m00'])
                    cy_px = int(M['m01'] / M['m00'])
                    negative_prompt_px.append((cx_px, cy_px))

                box = cv2.boxPoints(rect)
                cv2.fillPoly(building_mask_px, [np.int32(box)], 255)
                geo_pts = []
                for bx, by in box:
                    lon_p = min_lon + (bx / w_img) * (max_lon - min_lon)
                    lat_p = top_lat - (by / h_img) * (top_lat - bot_lat)
                    geo_pts.append((lon_p, lat_p))
                geo_pts.append(geo_pts[0])

                poly = Polygon(geo_pts)
                if poly.is_valid and poly.area > 0:
                    building_polygons.append({
                        'geometry': poly,
                        'area_m2': area_m2,
                        'centroid_px': (cx_px, cy_px),
                        'clase': 'LA_SpatialUnit_Building',
                        'subclase': bldg_type,
                        'has_shadow': has_shadow
                    })

        shapely_bldgs = [b['geometry'] for b in building_polygons]
        buildings_union = unary_union(shapely_bldgs) if shapely_bldgs else Polygon()

        return {
            'building_items': building_polygons,
            'negative_prompt_px': negative_prompt_px,
            'buildings_union': buildings_union,
            'building_mask_px': building_mask_px,
            'total_buildings': len(building_polygons)
        }

    def create_settlement_mask(self, buildings_union, buffer_m=20.0):
        if buildings_union is None or buildings_union.is_empty:
            return Polygon()
        return buildings_union.buffer(buffer_m * self.deg_per_meter)

    def filter_urban_settlement_courtyards(self, upa_polygons, settlement_geom, min_rural_upa_ha=0.065):
        if settlement_geom is None or settlement_geom.is_empty:
            return upa_polygons

        filtered_upas = []
        for item in upa_polygons:
            geom = item['geometry']
            area_ha = item.get('area_ha', (geom.area * 111000 * 111000) / 10000.0)
            
            if geom.intersects(settlement_geom):
                inter_area = geom.intersection(settlement_geom).area
                overlap_ratio = inter_area / (geom.area + 1e-12)
                
                if overlap_ratio > 0.60 and area_ha < min_rural_upa_ha:
                    continue

            filtered_upas.append(item)

        return filtered_upas

    def subtract_buildings_from_upas(self, upa_polygons, buildings_union, buffer_m=1.5):
        if buildings_union is None or buildings_union.is_empty:
            return upa_polygons

        bldg_buffer_geom = buildings_union.buffer(buffer_m * self.deg_per_meter)

        purified_upas = []
        for item in upa_polygons:
            geom = item['geometry']
            if geom.intersects(bldg_buffer_geom):
                diff_geom = geom.difference(bldg_buffer_geom)
                if diff_geom.is_empty:
                    continue
                if isinstance(diff_geom, Polygon):
                    if diff_geom.area > 1e-10:
                        it = item.copy()
                        it['geometry'] = diff_geom
                        it['area_ha'] = (diff_geom.area * 111000 * 111000) / 10000.0
                        purified_upas.append(it)
                elif isinstance(diff_geom, MultiPolygon):
                    for sub in diff_geom.geoms:
                        if sub.area > 1e-10:
                            it = item.copy()
                            it['geometry'] = sub
                            it['area_ha'] = (sub.area * 111000 * 111000) / 10000.0
                            purified_upas.append(it)
            else:
                purified_upas.append(item)

        return purified_upas
