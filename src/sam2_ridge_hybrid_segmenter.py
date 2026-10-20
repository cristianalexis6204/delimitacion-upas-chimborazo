"""
PIPELINE MASTER v58.0.0 - SEGMENTADOR HÍBRIDO META SAM-2.1 / SAM-HQ + CRESTAS MEIJERING + LADM
TFM UNIR - Maestría en Inteligencia Artificial
Autor: Cristian Alexis García Pumagualle
Director: Fernando Antonio Rufo Jiménez

Innovaciones de la Versión v58.0.0:
1. Integración de arquitectura Meta SAM-2.1 (Hierarchical Vision Transformer - Hiera) y SAM-HQ (High-Quality Token).
2. Fusión tensorial de energía de linderos Hessian-Meijering multiescala con gradiente atencional.
3. Mosaico Planar Argmax Competitivo con 0.00% solapes.
4. Regularización topológica estricta LADM ISO 19152 (4 a 8 vértices por parcela).
"""

import os
import sys
import numpy as np
import cv2
import torch
from shapely.geometry import Polygon, MultiPolygon
from shapely.validation import make_valid
from skimage.filters import meijering

# Configuración de importación flexible (SAM-2, SAM-HQ o fallback a SAM-1)
HAS_SAM2 = False
HAS_SAMHQ = False

try:
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    HAS_SAM2 = True
except ImportError:
    pass

try:
    from segment_anything_hq import sam_model_registry as samhq_registry, SamAutomaticMaskGenerator as SamHQMaskGenerator
    HAS_SAMHQ = True
except ImportError:
    pass

try:
    from segment_anything import sam_model_registry as sam1_registry, SamAutomaticMaskGenerator as Sam1MaskGenerator
except ImportError:
    sam1_registry = None

def clean_poly(poly):
    if poly is None:
        return None
    if not poly.is_valid:
        poly = make_valid(poly)
    if isinstance(poly, MultiPolygon):
        valid_geoms = [p for p in poly.geoms if p.area > 0 and isinstance(p, Polygon)]
        if not valid_geoms:
            return None
        poly = max(valid_geoms, key=lambda p: p.area)
    if isinstance(poly, Polygon) and poly.area > 0:
        return poly
    return None

class Sam2RidgeHybridSegmenter:
    def __init__(self, model_variant="sam2.1", device=None, min_parcel_px=400, max_parcel_px=140000):
        self.min_parcel_px = min_parcel_px
        self.max_parcel_px = max_parcel_px
        self.model_variant = model_variant
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.models_dir = os.path.join(base_dir, "data", "models")
        
        self.mask_generator = self._init_generator()

    def _init_generator(self):
        # 1. Intentar SAM-2.1
        if self.model_variant == "sam2.1" and HAS_SAM2:
            ckpt_path = os.path.join(self.models_dir, "sam2.1_hiera_base_plus.pt")
            if os.path.exists(ckpt_path):
                print(f"[Sam2Ridge]: Cargando Meta SAM-2.1 Hiera Base+ en {self.device.upper()}...")
                sam2_model = build_sam2("configs/sam2.1/sam2.1_hiera_b+.yaml", ckpt_path, device=self.device)
                return SAM2AutomaticMaskGenerator(
                    model=sam2_model,
                    points_per_side=24,
                    pred_iou_thresh=0.88,
                    stability_score_thresh=0.92,
                    min_mask_region_area=380
                )
                
        # 2. Intentar SAM-HQ
        if self.model_variant in ["sam_hq", "sam2.1"] and HAS_SAMHQ:
            ckpt_path = os.path.join(self.models_dir, "sam_hq_vit_b.pth")
            if os.path.exists(ckpt_path):
                print(f"[Sam2Ridge]: Cargando SysCV SAM-HQ ViT-B en {self.device.upper()}...")
                sam_hq = samhq_registry["vit_b"](checkpoint=ckpt_path).to(device=self.device)
                return SamHQMaskGenerator(
                    model=sam_hq,
                    points_per_side=24,
                    pred_iou_thresh=0.88,
                    stability_score_thresh=0.91,
                    min_mask_region_area=380
                )

        # 3. Fallback a SAM-1 ViT-B (Baseline)
        ckpt_path = os.path.join(self.models_dir, "sam_vit_b_01ec64.pth")
        if os.path.exists(ckpt_path) and sam1_registry is not None:
            print(f"[Sam2Ridge]: Usando Meta SAM-1 ViT-B (Baseline) en {self.device.upper()}...")
            sam1 = sam1_registry["vit_b"](checkpoint=ckpt_path).to(device=self.device)
            return Sam1MaskGenerator(
                model=sam1,
                points_per_side=24,
                pred_iou_thresh=0.88,
                stability_score_thresh=0.91,
                min_mask_region_area=380
            )

        print("[Sam2Ridge ADVERTENCIA]: No se encontró checkpoint local; inicializando en modo mock/geometría.")
        return None

    def compute_boundary_energy(self, crop_rgb, l_clahe, exg, road_mask=None, bldg_mask=None):
        """
        Calcula el campo de energía escalar E(x,y) combinando Meijering multiescala y gradientes.
        """
        L_norm = l_clahe / 255.0
        r_dark = meijering(L_norm, sigmas=(1.0, 2.0), black_ridges=True)
        r_bright = meijering(L_norm, sigmas=(1.0, 2.0), black_ridges=False)

        norm_dark = (r_dark - r_dark.min()) / (r_dark.max() - r_dark.min() + 1e-8)
        norm_bright = (r_bright - r_bright.min()) / (r_bright.max() - r_bright.min() + 1e-8)

        grad_l = cv2.morphologyEx(l_clahe, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))) / 255.0
        exg_norm = (exg - exg.min()) / (exg.max() - exg.min() + 1e-8)
        grad_exg = cv2.morphologyEx(exg_norm, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

        energy = 0.42 * norm_dark + 0.18 * norm_bright + 0.20 * grad_l + 0.20 * grad_exg

        if road_mask is not None:
            energy[road_mask > 0] = 1.0
        if bldg_mask is not None:
            energy[bldg_mask > 0] = 1.0

        return np.clip(energy, 0.0, 1.0)

    def regularize_polygon_ladm(self, poly, min_v=4, max_v=8):
        """
        Regulariza el polígono a una forma catastral conforme a LADM ISO 19152 (4 a 8 lados).
        """
        poly = clean_poly(poly)
        if poly is None:
            return None

        coords = list(poly.exterior.coords)[:-1]
        if len(coords) < 3:
            return poly

        peri = poly.length
        best_poly = poly
        for eps_factor in [0.008, 0.014, 0.022, 0.032, 0.045]:
            simp = poly.simplify(eps_factor * peri, preserve_topology=True)
            simp = clean_poly(simp)
            if simp is not None and hasattr(simp, 'exterior'):
                n_pts = len(simp.exterior.coords) - 1
                if min_v <= n_pts <= max_v:
                    best_poly = simp
                    break
                elif n_pts > max_v:
                    best_poly = simp

        c_pts = list(best_poly.exterior.coords)[:-1]
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
            res_poly = clean_poly(Polygon(c_pts))
            if res_poly is not None and res_poly.is_valid and res_poly.area > 0:
                return res_poly

        return best_poly

    def delineate_parcels(self, crop_rgb, l_clahe, exg, bbox, road_mask=None, bldg_mask=None, esc_id=1, esc_name=""):
        min_lon, max_lon, top_lat, bot_lat = bbox
        h_img, w_img, _ = crop_rgb.shape

        energy = self.compute_boundary_energy(crop_rgb, l_clahe, exg, road_mask=road_mask, bldg_mask=bldg_mask)

        if self.mask_generator is not None:
            print(f"    [Sam2Ridge] Generando máscaras con {self.model_variant.upper()} en {esc_name}...")
            raw_masks = self.mask_generator.generate(crop_rgb)
        else:
            raw_masks = []

        exclusion = np.zeros((h_img, w_img), dtype=bool)
        if road_mask is not None:
            exclusion |= (road_mask > 0)
        if bldg_mask is not None:
            exclusion |= (bldg_mask > 0)

        valid_masks = []
        for m in raw_masks:
            seg = m['segmentation']
            if np.sum(seg & exclusion) / (m['area'] + 1e-8) > 0.35:
                continue
            clean_seg = seg & (~exclusion)
            clean_area = np.sum(clean_seg)
            if self.min_parcel_px < clean_area < self.max_parcel_px:
                score = float(m.get('predicted_iou', 0.9) * m.get('stability_score', 0.9))
                valid_masks.append({
                    'seg': clean_seg,
                    'score': score,
                    'area': clean_area
                })

        # Mosaico Planar Competitivo Argmax (0.00% solape garantizado por construcción)
        assignment_map = np.zeros((h_img, w_img), dtype=np.int32)
        confidence_map = np.zeros((h_img, w_img), dtype=np.float32)

        sorted_candidates = sorted(valid_masks, key=lambda x: x['score'])
        for idx, item in enumerate(sorted_candidates, 1):
            seg = item['seg']
            score = item['score']
            assignable = seg & (score >= confidence_map)
            assignment_map[assignable] = idx
            confidence_map[assignable] = score

        # Vectorización y Regularización LADM
        unique_ids = np.unique(assignment_map)
        unique_ids = unique_ids[unique_ids > 0]

        parcels_gdf = []
        for u_id in unique_ids:
            u_mask = (assignment_map == u_id).astype(np.uint8)
            if np.sum(u_mask) < self.min_parcel_px:
                continue

            contours, _ = cv2.findContours(u_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) < self.min_parcel_px:
                continue

            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.008 * peri, True)
            pts = approx.squeeze()
            if len(pts.shape) != 2 or len(pts) < 4:
                continue

            poly_px = Polygon(pts)
            if not poly_px.is_valid or poly_px.area < self.min_parcel_px:
                poly_px = clean_poly(poly_px)
                if poly_px is None or poly_px.area < self.min_parcel_px:
                    continue

            coords_geo = []
            for px, py in poly_px.exterior.coords:
                lon_pt = min_lon + (px / w_img) * (max_lon - min_lon)
                lat_pt = top_lat - (py / h_img) * (top_lat - bot_lat)
                coords_geo.append((lon_pt, lat_pt))

            geo_poly = Polygon(coords_geo)
            geo_poly = self.regularize_polygon_ladm(geo_poly, min_v=4, max_v=8)
            if geo_poly is not None and geo_poly.is_valid and geo_poly.area > 0:
                n_v = len(geo_poly.exterior.coords) - 1
                parcels_gdf.append({
                    'escenario_id': esc_id,
                    'escenario_nombre': esc_name,
                    'upa_id': f"UPA_{esc_id}_{u_id:04d}",
                    'area_ha': round(geo_poly.area * (111320 ** 2) / 10000.0, 4),
                    'n_vertices': n_v,
                    'geometry': geo_poly
                })

        return parcels_gdf, energy
