"""
MÓDULO EXTRACTOR DE RED VIAL Y VÍAS DE TERRACERÍA (TFM UNIR - v55.0.0 SOTA)
Autor: Cristian Alexis García Pumagualle

Fundamentación Científica y Estándar LPIS (Reg. UE 640/2014 & RENAGRO):
- Integra infraestructura vial oficial de alta fidelidad (OpenStreetMap con caché local offline).
- Arquitectura Resiliente Multinivel:
  1. Base vectorial local de alta resolución (GeoJSON precacheado de Chimborazo).
  2. Consulta en vivo a Overpass API con auto-almacenamiento en caché.
  3. Extractor espectral-morfológico de respaldo para caminos rurales y senderos (chaquiñanes).
- Búfer de servidumbre vial reglamentario de 3.5 metros (LPIS) para partición planar estanco.
- Cero fallos por HTTP 504 o desconexión de red: 100% de persistencia garantizada.
"""

import os
import json
import math
import urllib.request
import urllib.parse
import numpy as np
import cv2
from shapely.geometry import LineString, MultiLineString, Polygon, MultiPolygon, box, shape
from shapely.ops import unary_union


class RuralRoadNetworkExtractor:
    def __init__(self, road_buffer_m=3.5, cache_dir=None, default_geojson_path='data/osm_roads_chimborazo_cache.geojson'):
        self.road_buffer_m = road_buffer_m
        self.deg_per_meter = 1.0 / 111000.0
        self.cache_dir = cache_dir
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
            print(f"  [Aviso Cache Master GeoJSON]: {e}")
            return []

    def fetch_osm_roads(self, bbox, esc_name="scene"):
        min_lon, max_lon, top_lat, bot_lat = bbox
        min_lat, max_lat = min(top_lat, bot_lat), max(top_lat, bot_lat)
        scene_box = box(min_lon, min_lat, max_lon, max_lat)
        
        # 1. Comprobar caché master GeoJSON
        master_lines = self._load_from_master_geojson(scene_box)
        if len(master_lines) > 0:
            return master_lines

        # 2. Comprobar caché local de escena
        if self.cache_dir:
            cache_file = os.path.join(self.cache_dir, f"osm_roads_{esc_name}.json")
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        cached_data = json.load(f)
                    lines = []
                    for coords in cached_data:
                        if len(coords) >= 2:
                            ls = LineString(coords)
                            clipped = ls.intersection(scene_box)
                            if not clipped.is_empty:
                                if isinstance(clipped, LineString):
                                    lines.append(clipped)
                                elif hasattr(clipped, 'geoms'):
                                    lines.extend([g for g in clipped.geoms if isinstance(g, LineString)])
                    if lines:
                        return lines
                except Exception:
                    pass

        # 3. Descargar de Overpass API (con timeout controlado y auto-cache)
        overpass_url = 'https://overpass-api.de/api/interpreter'
        bbox_str = f"{min_lat - 0.002},{min_lon - 0.002},{max_lat + 0.002},{max_lon + 0.002}"
        query = f"""[out:json][timeout:15];
        (
          way["highway"]({bbox_str});
        );
        out body;
        >;
        out skel qt;"""
        
        lines = []
        raw_coords_list = []
        try:
            data = urllib.parse.urlencode({'data': query}).encode('utf-8')
            req = urllib.request.Request(overpass_url, data=data, headers={'User-Agent': 'TFM-UNIR-Roads/1.0'})
            with urllib.request.urlopen(req, timeout=10) as response:
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
                            
            # Guardar en caché si se obtuvieron resultados
            if self.cache_dir and raw_coords_list:
                os.makedirs(self.cache_dir, exist_ok=True)
                cache_file = os.path.join(self.cache_dir, f"osm_roads_{esc_name}.json")
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(raw_coords_list, f)
        except Exception as e:
            print(f"  [Aviso Overpass]: Falló consulta remota ({e}).")

        return lines

    def detect_spectral_dirt_roads(self, rgb_image, l_clahe, exg, bbox):
        """
        Extractor espectral y morfológico de caminos de tierra y chaquiñanes rurales.
        Se activa como respaldo si no hay datos vectoriales OSM.
        """
        h_img, w_img, _ = rgb_image.shape
        min_lon, max_lon, top_lat, bot_lat = bbox

        # Suelo desnudo compactado: alta luminancia, baja vegetación
        is_road_candidate = (l_clahe > 135) & (exg < 0.02)
        
        # Filtro morfológico direccional para estructuras lineales
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
            road_mask = cv2.bitwise_or(road_mask, opened)

        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        contours, _ = cv2.findContours(road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_lines = []
        for cnt in contours:
            if len(cnt) >= 4 and cv2.arcLength(cnt, False) > 60:
                pts_geo = []
                for pt in cnt[::3]:
                    px, py = pt[0][0], pt[0][1]
                    lon = min_lon + (px / w_img) * (max_lon - min_lon)
                    lat = top_lat - (py / h_img) * (top_lat - bot_lat)
                    pts_geo.append((lon, lat))
                if len(pts_geo) >= 2:
                    detected_lines.append(LineString(pts_geo))
                    
        return detected_lines

    def extract_roads(self, rgb_image, l_clahe, exg, boundary_energy, bbox, esc_name="scene"):
        h_img, w_img, _ = rgb_image.shape
        min_lon, max_lon, top_lat, bot_lat = bbox
        
        shapely_lines = self.fetch_osm_roads(bbox, esc_name=esc_name)
        
        # Si OSM no arrojó vías, recurrir al extractor espectral-morfológico
        if not shapely_lines:
            print(f"  [Respaldo Espectral]: Activando detección espectral-morfológica de caminos rurales...")
            shapely_lines = self.detect_spectral_dirt_roads(rgb_image, l_clahe, exg, bbox)

        road_lines_geo = []
        road_mask_px = np.zeros((h_img, w_img), dtype=np.uint8)
        
        for ls in shapely_lines:
            pts_px = []
            lons, lats = [], []
            for lon_p, lat_p in ls.coords:
                px = int(round((lon_p - min_lon) / (max_lon - min_lon) * w_img))
                py = int(round((top_lat - lat_p) / (top_lat - bot_lat) * h_img))
                pts_px.append([px, py])
                lons.append(lon_p)
                lats.append(lat_p)
            if len(pts_px) >= 2:
                cv2.polylines(road_mask_px, [np.array(pts_px, dtype=np.int32)], False, 255, thickness=4)
                road_lines_geo.append((lons, lats))
                
        buffer_deg = self.road_buffer_m * self.deg_per_meter
        if shapely_lines:
            union_lines = unary_union(shapely_lines)
            road_buffer_geom = union_lines.buffer(buffer_deg, cap_style=1, join_style=1)
        else:
            road_buffer_geom = Polygon()
            
        return {
            'road_lines_geo': road_lines_geo,
            'shapely_lines': shapely_lines,
            'road_buffer_geom': road_buffer_geom,
            'road_mask_px': road_mask_px,
            'total_roads_detected': len(shapely_lines)
        }

    def partition_and_enforce_roads(self, upa_polygons, road_buffer_geom, min_parcel_ha=0.020):
        if road_buffer_geom is None or road_buffer_geom.is_empty:
            return upa_polygons
            
        adjusted = []
        for item in upa_polygons:
            geom = item['geometry']
            if not geom.is_valid:
                geom = geom.buffer(0)
            if geom.intersects(road_buffer_geom):
                diff = geom.difference(road_buffer_geom)
                if diff.is_empty:
                    continue
                parts = []
                if isinstance(diff, Polygon):
                    parts = [diff]
                elif isinstance(diff, MultiPolygon):
                    parts = list(diff.geoms)
                for part in parts:
                    if not part.is_valid or part.area <= 0:
                        continue
                    part_ha = (part.area * 111000 * 111000) / 10000.0
                    if part_ha >= min_parcel_ha:
                        new_item = item.copy()
                        new_item['geometry'] = part
                        new_item['area_ha'] = part_ha
                        new_item['num_vertices'] = len(part.exterior.coords) - 1
                        adjusted.append(new_item)
            else:
                adjusted.append(item)
        return adjusted
