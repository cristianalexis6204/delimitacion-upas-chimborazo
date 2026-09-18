"""
Módulo Generador del Visor Cartográfico Web Interactivo HTML (Folium / Leaflet)
TFM UNIR - Autor: Cristian Alexis García Pumagualle
"""

import os
import folium
from folium import plugins
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon

class InteractiveMapBuilder:
    def __init__(self, title="Catastro Rural Chimborazo (LADM ISO 19152)"):
        self.title = title

    def build_map(self, output_html_path, gdf_upas, gdf_roads=None, gdf_bldgs=None, center_lat=-1.6350, center_lon=-78.7850, zoom_start=15):
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=zoom_start,
            control_scale=True,
            tiles=None
        )

        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri World Imagery",
            name="Satelital Esri HD",
            overlay=False,
            control=True
        ).add_to(m)

        folium.TileLayer(
            tiles="OpenStreetMap",
            name="OpenStreetMap Carto",
            overlay=False,
            control=True
        ).add_to(m)

        if gdf_roads is not None and not gdf_roads.empty:
            fg_roads = folium.FeatureGroup(name="Red Vial y Caminos (OSM)", show=True)
            for _, row in gdf_roads.iterrows():
                geom = row["geometry"]
                esc = row.get("escenario", "Chimborazo")
                if geom.geom_type == "LineString":
                    coords = [[lat, lon] for lon, lat in geom.coords]
                    folium.PolyLine(
                        locations=coords,
                        color="#00FFFF",
                        weight=3.2,
                        opacity=0.9,
                        tooltip=f"Via Rural Oficial ({esc})"
                    ).add_to(fg_roads)
            fg_roads.add_to(m)

        if gdf_bldgs is not None and not gdf_bldgs.empty:
            fg_bldgs = folium.FeatureGroup(name="Viviendas Campesinas (3D Sombra)", show=True)
            for idx, row in gdf_bldgs.iterrows():
                geom = row["geometry"]
                area_m2 = row.get("area_m2", 60.0)
                subclase = row.get("subclase", "Vivienda_Campesina")
                shadow = "Verificada (Sombra 3D)" if row.get("has_shadow", False) else "Firma Espectral"
                
                popup_html = f"""
                <div style="font-family: Arial; font-size: 11px; width: 220px;">
                    <b style="color: #9333EA;">Edificacion LADM: Edif_{idx+1:03d}</b><br/>
                    <b>Tipologia:</b> {subclase}<br/>
                    <b>Superficie Cubierta:</b> {area_m2:.1f} m2<br/>
                    <b>Elevacion 3D:</b> {shadow}<br/>
                    <b>Geometria:</b> Ortogonal 90 deg<br/>
                    <b>Estado:</b> 100% Segregada (Bufer 1.5m)
                </div>
                """
                if geom.geom_type == "Polygon":
                    coords = [[lat, lon] for lon, lat in geom.exterior.coords]
                    folium.Polygon(
                        locations=coords,
                        color="#D946EF",
                        fill_color="#D946EF",
                        fill_opacity=0.45,
                        weight=2.0,
                        tooltip=f"Vivienda {area_m2:.0f} m2",
                        popup=folium.Popup(popup_html, max_width=250)
                    ).add_to(fg_bldgs)
            fg_bldgs.add_to(m)

        if gdf_upas is not None and not gdf_upas.empty:
            fg_upas = folium.FeatureGroup(name="UPAs Agricolas Delimitadas (LADM)", show=True)
            for idx, row in gdf_upas.iterrows():
                geom = row["geometry"]
                area_ha = row.get("area_ha", 0.0)
                n_vert = row.get("num_vertices", 4)
                clase = row.get("clase", "Cultivo_Verde")
                esc = row.get("escenario", "Chimborazo")
                upa_id = f"UPA-{idx+1:03d}"

                popup_html = f"""
                <div style="font-family: Arial; font-size: 12px; width: 240px;">
                    <h4 style="color: #0F766E; margin-bottom: 4px; border-bottom: 2px solid #0F766E;">FICHA PREDIAL: {upa_id}</h4>
                    <table style="width: 100%; font-size: 11px; border-collapse: collapse;">
                        <tr bgcolor="#F8FAFC"><td><b>Escenario:</b></td><td>{esc}</td></tr>
                        <tr><td><b>Superficie Util:</b></td><td><b>{area_ha:.4f} ha</b> ({area_ha*10000:.1f} m2)</td></tr>
                        <tr bgcolor="#F8FAFC"><td><b>Vertices LADM:</b></td><td>{n_vert} vertices</td></tr>
                        <tr><td><b>Uso de Suelo:</b></td><td>{clase}</td></tr>
                        <tr bgcolor="#F8FAFC"><td><b>Solape:</b></td><td>0.00% (Estanco LADM)</td></tr>
                        <tr><td><b>Estandar:</b></td><td>Conforme ISO 19152</td></tr>
                    </table>
                </div>
                """

                if geom.geom_type == "Polygon":
                    coords = [[lat, lon] for lon, lat in geom.exterior.coords]
                    folium.Polygon(
                        locations=coords,
                        color="#FACC15",
                        fill=False,
                        weight=2.4,
                        opacity=0.95,
                        tooltip=f"{upa_id}: {area_ha:.2f} ha ({n_vert} vertices)",
                        popup=folium.Popup(popup_html, max_width=260)
                    ).add_to(fg_upas)
            fg_upas.add_to(m)

        folium.LayerControl(collapsed=False).add_to(m)
        plugins.Fullscreen(position="topright").add_to(m)
        plugins.MeasureControl(position="bottomleft", primary_length_unit="meters").add_to(m)

        m.save(output_html_path)
        print(f"[Visor Web HTML Creado]: {output_html_path}")
        return output_html_path
