"""
MÓDULO EXTRACTOR DE RED VIAL Y VÍAS DE TERRACERÍA (TFM UNIR - v47.0.0)
Autor: Cristian Alexis García Pumagualle

Fundamentación Científica y Estándar LPIS (Reg. UE 640/2014 & RENAGRO):
- Integra infraestructura vial oficial de alta fidelidad (OpenStreetMap / Overpass API con caché local).
- Identifica carreteras pavimentadas, calles residenciales, caminos vecinales y senderos agrícolas (tracks/paths).
- Búfer de servidumbre vial de 3.5 metros (LPIS) para partición planar y exclusión estricta de UPAs.
- Cero líneas parásitas sobre techos, patios o solares interiores.
"""

import os
import json
import math
import urllib.request
import urllib.parse
import numpy as np
import cv2
from shapely.geometry import LineString, MultiLineString, Polygon, MultiPolygon, box
from shapely.ops import unary_union

class RuralRoadNetworkExtractor:
    def __init__(self, road_buffer_m=3.5, cache_dir=None):
        self.road_buffer_m = road_buffer_m
        self.deg_per_meter = 1.0 / 111000.0
        self.cache_dir = cache_dir

    def fetch_osm_roads(self, bbox, esc_name="scene"):
        min_lon, max_lon, top_lat, bot_lat = bbox
        min_lat, max_lat = min(top_lat, bot_lat), max(top_lat, bot_lat)
        scene_box = box(min_lon, min_lat, max_lon, max_lat)
        
        # 1. Comprobar caché local
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
                                elif isinstance(clipped, MultiLineString):
                                    lines.extend(clipped.geoms)
                    if lines:
                        return lines
                except Exception:
                    pass

        # 2. Descargar de Overpass API
        overpass_url = 'https://overpass-api.de/api/interpreter'
        bbox_str = f"{min_lat - 0.001},{min_lon - 0.001},{max_lat + 0.001},{max_lon + 0.001}"
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
            with urllib.request.urlopen(req, timeout=12) as response:
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
                        elif isinstance(clipped, MultiLineString):
                            lines.extend(clipped.geoms)
                            
            # Guardar en caché
            if self.cache_dir and raw_coords_list:
                cache_file = os.path.join(self.cache_dir, f"osm_roads_{esc_name}.json")
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(raw_coords_list, f)
        except Exception as e:
            print(f"  [Advertencia Overpass]: {e}")
            
        return lines

    def extract_roads(self, rgb_image, l_clahe, exg, boundary_energy, bbox, esc_name="scene"):
        h_img, w_img, _ = rgb_image.shape
        min_lon, max_lon, top_lat, bot_lat = bbox
        
        shapely_lines = self.fetch_osm_roads(bbox, esc_name=esc_name)
        
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

    def partition_and_enforce_roads(self, upa_polygons, road_buffer_geom, min_parcel_ha=0.025):
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
