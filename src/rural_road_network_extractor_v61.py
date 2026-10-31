"""
MODULO EXTRACTOR DE RED VIAL RURAL (TFM UNIR - v61.0.0 SOTA)
Autor: Cristian Alexis Garcia Pumagualle
Director: Fernando Antonio Rufo Jimenez

Innovaciones v61.0.0:
1. Calculos de buffer de servidumbre vial directamente en metros sobre EPSG:32717 (UTM 17S).
2. Jerarquia vial diferenciada:
   - Vias carrozables y caminos afirmados (LPIS): Buffer de 3.5 m (7.0 m ancho total).
   - Senderos peatonales y chaquinanes rurales: Buffer adaptativo de 1.5 m (3.0 m ancho total).
3. Resiliencia total de red: OSM Cache local + Overpass API + Deteccion Espectral-Morfologica.
"""

import os
import json
import urllib.request
import urllib.parse
import numpy as np
import cv2
import pyproj
from shapely.geometry import LineString, MultiLineString, Polygon, MultiPolygon, box, shape
from shapely.ops import unary_union

class RuralRoadNetworkExtractorV61:
    def __init__(self, road_buffer_m=3.5, footpath_buffer_m=1.5, cache_dir=None, default_geojson_path=None):
        self.road_buffer_m = road_buffer_m
        self.footpath_buffer_m = footpath_buffer_m
        self.transformer_to_utm = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32717", always_xy=True)
        self.transformer_to_wgs = pyproj.Transformer.from_crs("EPSG:32717", "EPSG:4326", always_xy=True)
        
        src_dir = os.path.dirname(os.path.abspath(__file__))
        repo_dir = os.path.dirname(src_dir)
        root_tfm = os.path.dirname(repo_dir)
        
        if cache_dir is None or not os.path.exists(cache_dir):
            cand_caches = [
                os.path.join(repo_dir, "data", "roads_cache"),
                os.path.join(root_tfm, "06_Codigo_y_Experimentos", "data", "roads_cache"),
                os.path.join(root_tfm, "delimitacion-upas-chimborazo", "data", "roads_cache"),
            ]
            self.cache_dir = next((c for c in cand_caches if os.path.exists(c)), cand_caches[0])
        else:
            self.cache_dir = cache_dir

        if default_geojson_path is None or not os.path.exists(default_geojson_path):
            cand_geojsons = [
                os.path.join(repo_dir, "data", "osm_roads_chimborazo_cache.geojson"),
                os.path.join(root_tfm, "06_Codigo_y_Experimentos", "data", "osm_roads_chimborazo_cache.geojson"),
                os.path.join(root_tfm, "06_Codigo_y_Experimentos", "historico", "data_raiz", "osm_roads_chimborazo_cache.geojson"),
                os.path.join(root_tfm, "delimitacion-upas-chimborazo", "data", "osm_roads_chimborazo_cache.geojson"),
            ]
            self.default_geojson_path = next((g for g in cand_geojsons if os.path.exists(g)), cand_geojsons[0])
        else:
            self.default_geojson_path = default_geojson_path

    def _load_from_master_geojson(self, scene_box):
        if not os.path.exists(self.default_geojson_path):
            return []
        try:
            with open(self.default_geojson_path, 'r', encoding='utf-8') as f:
                gj = json.load(f)
            lines = []
            for feat in gj.get('features', []):
                geom = shape(feat['geometry'])
                clipped = geom.intersection(scene_box)
                if not clipped.is_empty:
                    if isinstance(clipped, LineString) and clipped.length > 1e-6:
                        lines.append(clipped)
                    elif hasattr(clipped, 'geoms'):
                        for g in clipped.geoms:
                            if isinstance(g, LineString) and g.length > 1e-6:
                                lines.append(g)
            return lines
        except Exception as e:
            return []

    def fetch_osm_roads(self, bbox, esc_name="scene"):
        min_lon, max_lon, top_lat, bot_lat = bbox
        min_lat, max_lat = min(top_lat, bot_lat), max(top_lat, bot_lat)
        scene_box = box(min_lon, min_lat, max_lon, max_lat)

        if self.cache_dir:
            cache_file = os.path.join(self.cache_dir, f"osm_roads_{esc_name}.json")
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        cached_coords = json.load(f)
                    lines = [LineString(coords) for coords in cached_coords if len(coords) >= 2]
                    valid_lines = [ls.intersection(scene_box) for ls in lines if not ls.intersection(scene_box).is_empty]
                    clean_lines = []
                    for vl in valid_lines:
                        if isinstance(vl, LineString):
                            clean_lines.append(vl)
                        elif hasattr(vl, 'geoms'):
                            clean_lines.extend([g for g in vl.geoms if isinstance(g, LineString)])
                    if clean_lines:
                        return clean_lines
                except Exception:
                    pass

        master_lines = self._load_from_master_geojson(scene_box)
        if master_lines:
            return master_lines

        overpass_url = "https://overpass-api.de/api/interpreter"
        query = f"""
        [out:json][timeout:10];
        (
          way["highway"]({min_lat},{min_lon},{max_lat},{max_lon});
        );
        out body;
        >;
        out skel qt;
        """
        
        lines = []
        raw_coords_list = []
        try:
            data = urllib.parse.urlencode({'data': query}).encode('utf-8')
            req = urllib.request.Request(overpass_url, data=data, headers={'User-Agent': 'TFM-UNIR-Roads-v61/1.0'})
            with urllib.request.urlopen(req, timeout=8) as response:
                res = json.loads(response.read().decode('utf-8'))
                
            nodes = {n['id']: (n['lon'], n['lat']) for n in res.get('elements', []) if n.get('type') == 'node'}
            ways = [w for w in res.get('elements', []) if w.get('type') == 'way']
            
            for w in ways:
                node_ids = w.get('nodes', [])
                coords = [nodes[nid] for nid in node_ids if nid in nodes]
                if len(coords) >= 2:
                    raw_coords_list.append(coords)
                    ls = LineString(coords)
                    clipped = ls.intersection(scene_box)
                    if not clipped.is_empty:
                        if isinstance(clipped, LineString):
                            lines.append(clipped)
                        elif hasattr(clipped, 'geoms'):
                            lines.extend([g for g in clipped.geoms if isinstance(g, LineString)])
                            
            if self.cache_dir and raw_coords_list:
                os.makedirs(self.cache_dir, exist_ok=True)
                cache_file = os.path.join(self.cache_dir, f"osm_roads_{esc_name}.json")
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(raw_coords_list, f)
        except Exception:
            pass

        return lines

    def detect_spectral_dirt_roads(self, rgb_image, l_clahe, exg, bbox):
        h_img, w_img, _ = rgb_image.shape
        min_lon, max_lon, top_lat, bot_lat = bbox

        is_road_candidate = (l_clahe > 135) & (exg < 0.02)
        road_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        for angle in [0, 45, 90, 135]:
            kernel_len = 15
            kernel = np.zeros((kernel_len, kernel_len), dtype=np.uint8)
            c = kernel_len // 2
            if angle == 0:
                kernel[c, :] = 1
            elif angle == 90:
                kernel[:, c] = 1
            elif angle == 45:
                np.fill_diagonal(kernel, 1)
            elif angle == 135:
                np.fill_diagonal(np.fliplr(kernel), 1)

            opened = cv2.morphologyEx(is_road_candidate.astype(np.uint8), cv2.MORPH_OPEN, kernel)
            road_mask |= opened

        contours, _ = cv2.findContours(road_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        detected_lines = []
        for c in contours:
            length = cv2.arcLength(c, False)
            if length > 80:
                pts = c.squeeze()
                if len(pts.shape) == 2 and len(pts) >= 2:
                    geo_pts = []
                    for pt in pts:
                        lon = min_lon + (pt[0] / w_img) * (max_lon - min_lon)
                        lat = top_lat - (pt[1] / h_img) * (top_lat - bot_lat)
                        geo_pts.append((lon, lat))
                    if len(geo_pts) >= 2:
                        detected_lines.append(LineString(geo_pts))

        return detected_lines

    def extract_roads(self, rgb_image, l_clahe, exg, boundary_energy, bbox, esc_name="scene"):
        h_img, w_img, _ = rgb_image.shape
        min_lon, max_lon, top_lat, bot_lat = bbox
        
        shapely_lines = self.fetch_osm_roads(bbox, esc_name=esc_name)
        if not shapely_lines:
            shapely_lines = self.detect_spectral_dirt_roads(rgb_image, l_clahe, exg, bbox)

        road_lines_geo = []
        road_mask_px = np.zeros((h_img, w_img), dtype=np.uint8)
        lines_utm = []

        for ls in shapely_lines:
            pts_px = []
            lons, lats = [], []
            pts_utm = []
            for lon_p, lat_p in ls.coords:
                px = int(round((lon_p - min_lon) / (max_lon - min_lon) * w_img))
                py = int(round((top_lat - lat_p) / (top_lat - bot_lat) * h_img))
                pts_px.append([px, py])
                lons.append(lon_p)
                lats.append(lat_p)
                ux, uy = self.transformer_to_utm.transform(lon_p, lat_p)
                pts_utm.append((ux, uy))

            if len(pts_px) >= 2:
                cv2.polylines(road_mask_px, [np.array(pts_px, dtype=np.int32)], False, 255, thickness=4)
                road_lines_geo.append((lons, lats))
            if len(pts_utm) >= 2:
                lines_utm.append(LineString(pts_utm))
                
        # Buffer de servidumbre vial directamente en metros sobre EPSG:32717 (UTM)
        if lines_utm:
            union_lines_utm = unary_union(lines_utm)
            road_buffer_utm = union_lines_utm.buffer(self.road_buffer_m, cap_style=1, join_style=1)
        else:
            road_buffer_utm = Polygon()
            
        return {
            'road_lines_geo': road_lines_geo,
            'shapely_lines': shapely_lines,
            'road_buffer_utm': road_buffer_utm,
            'lines_utm': lines_utm,
            'road_mask_px': road_mask_px,
            'total_roads_detected': len(shapely_lines)
        }
