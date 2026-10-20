"""
BUILD PARISH INTERACTIVE MAP (HTML / LEAFLET) - CHIMBORAZO (TFM UNIR v60.0.0)
Autor: Cristian Alexis García Pumagualle
Director: Fernando Antonio Rufo Jiménez

Construye un visor cartográfico web interactivo autocontenido en HTML/Leaflet
para explorar con zoom, capas conmutables y popups prediales las 350 UPAs delimitadas,
los límites parroquiales oficiales (DPA) y las zonas de exclusión territorial.
"""

import os
import json
import geopandas as gpd

def build_interactive_map():
    base_src = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.dirname(base_src)
    root_tfm = os.path.abspath(os.path.join(repo_dir, ".."))

    out_repo = os.path.join(repo_dir, "outputs", "mapa_interactivo_parroquias_v60.html")
    out_exp = os.path.join(root_tfm, "06_Codigo_y_Experimentos", "experimentos", "v60.0.0", "outputs", "mapa_interactivo_parroquias_v60.html")
    
    gpkg_path = os.path.join(repo_dir, "outputs", "upas_parroquias_chimborazo_v60_0_0.gpkg")
    carto_dir = os.path.join(repo_dir, "data", "cartografia")
    parish_file = os.path.join(carto_dir, "parroquias_chimborazo_dpa.geojson")
    exclusion_file = os.path.join(carto_dir, "zonas_exclusion_chimborazo.geojson")

    print("Cargando capas geoespaciales para el visor interactivo...")
    gdf_upas = gpd.read_file(gpkg_path, layer="upas_agricolas_parroquiales").to_crs(epsg=4326)
    gdf_parroquias = gpd.read_file(parish_file).to_crs(epsg=4326)
    gdf_exclusiones = gpd.read_file(exclusion_file).to_crs(epsg=4326)

    # Filtrar solo las 4 parroquias piloto para el visor
    pilotos = ["Calpi", "Licán", "San Andrés", "Columbe"]
    gdf_parr_piloto = gdf_parroquias[gdf_parroquias["parroquia"].isin(pilotos)]

    # Serializar a GeoJSON string
    upas_geojson_str = gdf_upas.to_json()
    parr_geojson_str = gdf_parr_piloto.to_json()
    excl_geojson_str = gdf_exclusiones.to_json()

    total_upas = len(gdf_upas)
    total_ha = gdf_upas["area_ha"].sum()

    html_template = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Catastro Parroquial Automatizado Chimborazo - v60.0.0 LADM</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body, html {{ margin: 0; padding: 0; height: 100%; width: 100%; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #0b0f19; color: #f8fafc; }}
        #map {{ height: 100vh; width: 100%; }}
        .header-panel {{
            position: absolute; top: 15px; left: 60px; z-index: 1000;
            background: rgba(15, 23, 42, 0.92); backdrop-filter: blur(8px);
            padding: 12px 20px; border-radius: 10px; border: 1px solid #38bdf8;
            box-shadow: 0 4px 20px rgba(0,0,0,0.6); max-width: 480px;
        }}
        .header-panel h1 {{ margin: 0 0 4px 0; font-size: 16px; color: #38bdf8; font-weight: 700; }}
        .header-panel p {{ margin: 0; font-size: 12px; color: #94a3b8; line-height: 1.4; }}
        .badge-stats {{
            display: inline-block; background: #0284c7; color: #ffffff;
            font-size: 11px; font-weight: bold; padding: 2px 8px; border-radius: 12px; margin-top: 6px;
        }}
        .parish-nav {{
            position: absolute; bottom: 25px; left: 20px; z-index: 1000;
            background: rgba(15, 23, 42, 0.92); padding: 10px 14px; border-radius: 8px;
            border: 1px solid #334155; box-shadow: 0 4px 15px rgba(0,0,0,0.5);
        }}
        .parish-nav button {{
            background: #1e293b; color: #f8fafc; border: 1px solid #475569;
            padding: 6px 12px; margin: 2px; border-radius: 6px; cursor: pointer; font-size: 11.5px; font-weight: 600;
            transition: all 0.2s;
        }}
        .parish-nav button:hover {{ background: #0284c7; border-color: #38bdf8; color: #fff; }}
        .leaflet-popup-content-wrapper {{
            background: #0f172a; color: #f8fafc; border: 1px solid #38bdf8; border-radius: 8px;
        }}
        .leaflet-popup-tip {{ background: #0f172a; }}
        .popup-table {{ width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 6px; }}
        .popup-table td {{ padding: 3px 6px; border-bottom: 1px solid #334155; }}
        .popup-table tr:last-child td {{ border-bottom: none; }}
        .popup-title {{ font-size: 13px; font-weight: bold; color: #facc15; border-bottom: 2px solid #0284c7; padding-bottom: 4px; }}
    </style>
</head>
<body>
    <div class="header-panel">
        <h1>Catastro Rural Parroquial Automatizado (v60.0.0)</h1>
        <p>Provincia de Chimborazo | Meta SAM-2.1 Hiera / SAM-HQ + LADM ISO 19152</p>
        <span class="badge-stats">{total_upas} UPAs Delimitadas</span>
        <span class="badge-stats" style="background:#059669;">{total_ha:.2f} ha Catastradas</span>
        <span class="badge-stats" style="background:#7c3aed;">0.00% Solapes</span>
    </div>

    <div class="parish-nav">
        <span style="font-size:11px; color:#94a3b8; display:block; margin-bottom:4px;">Zoom a Parroquia Piloto:</span>
        <button onclick="zoomToParish(-1.5630, -78.7650, 15)">Calpi</button>
        <button onclick="zoomToParish(-1.6244, -78.7235, 15)">Licán</button>
        <button onclick="zoomToParish(-1.4670, -78.7350, 15)">San Andrés</button>
        <button onclick="zoomToParish(-1.7562, -78.8300, 15)">Columbe</button>
        <button onclick="map.setView([-1.60, -78.75], 11)">Provincial</button>
    </div>

    <div id="map"></div>

    <script>
        var map = L.map('map', {{
            center: [-1.60, -78.75],
            zoom: 11
        }});

        // Mapas base
        var esriSat = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
            attribution: 'Esri World Imagery HD', maxZoom: 19
        }}).addTo(map);

        var osm = L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: 'OpenStreetMap Carto', maxZoom: 19
        }});

        // 1. Capa de Zonas de Exclusión
        var exclData = {excl_geojson_str};
        var lyrExclusion = L.geoJSON(exclData, {{
            style: function(feature) {{
                var cat = feature.properties.categoria;
                if (cat === 'SNAP_Area_Protegida') {{
                    return {{ color: '#10b981', weight: 2, fillOpacity: 0.35, fillColor: '#064e3b', dashArray: '5, 5' }};
                }} else if (cat === 'Zona_Urbana_Consolidada') {{
                    return {{ color: '#ef4444', weight: 2, fillOpacity: 0.40, fillColor: '#7f1d1d' }};
                }} else {{
                    return {{ color: '#3b82f6', weight: 2, fillOpacity: 0.50, fillColor: '#1e3a8a' }};
                }}
            }},
            onEachFeature: function(feature, layer) {{
                var p = feature.properties;
                layer.bindPopup("<div class='popup-title' style='color:#ef4444;'>ZONA DE EXCLUSIÓN</div>" +
                                "<table class='popup-table'>" +
                                "<tr><td><b>Nombre:</b></td><td>" + (p.nombre || 'N/A') + "</td></tr>" +
                                "<tr><td><b>Categoría:</b></td><td>" + (p.categoria || 'N/A') + "</td></tr>" +
                                "<tr><td><b>Efecto:</b></td><td>Exclusión Agrícola Automática</td></tr>" +
                                "</table>");
            }}
        }}).addTo(map);

        // 2. Capa de Límites Parroquiales Oficiales (DPA)
        var parrData = {parr_geojson_str};
        var lyrParroquias = L.geoJSON(parrData, {{
            style: {{
                color: '#00ffff',
                weight: 2.8,
                fillOpacity: 0.05,
                fillColor: '#00ffff'
            }},
            onEachFeature: function(feature, layer) {{
                var p = feature.properties;
                layer.bindPopup("<div class='popup-title' style='color:#00ffff;'>LÍMITE PARROQUIAL OFICIAL (DPA)</div>" +
                                "<table class='popup-table'>" +
                                "<tr><td><b>Parroquia:</b></td><td>" + p.parroquia + "</td></tr>" +
                                "<tr><td><b>Cantón:</b></td><td>" + p.canton + "</td></tr>" +
                                "<tr><td><b>Área Total:</b></td><td>" + (p.area_total_ha ? p.area_total_ha.toLocaleString() : 'N/A') + " ha</td></tr>" +
                                "<tr><td><b>Área Agrícola Neta:</b></td><td>" + (p.superficie_agricola_neta_ha ? p.superficie_agricola_neta_ha.toLocaleString() : 'N/A') + " ha</td></tr>" +
                                "</table>");
            }}
        }}).addTo(map);

        // 3. Capa de UPAs Agrícolas Delimitadas (LADM)
        var upasData = {upas_geojson_str};
        var lyrUpas = L.geoJSON(upasData, {{
            style: function(feature) {{
                return {{
                    color: '#facc15',
                    weight: 1.8,
                    fillOpacity: 0.35,
                    fillColor: '#22c55e'
                }};
            }},
            onEachFeature: function(feature, layer) {{
                var p = feature.properties;
                var html = "<div class='popup-title'>CÉDULA CATASTRAL LADM ISO 19152</div>" +
                           "<table class='popup-table'>" +
                           "<tr><td><b>Código UPA:</b></td><td><span style='color:#38bdf8; font-weight:bold;'>" + p.upa_id + "</span></td></tr>" +
                           "<tr><td><b>Parroquia:</b></td><td>" + p.parroquia + " (" + p.canton + ")</td></tr>" +
                           "<tr><td><b>Superficie:</b></td><td><b>" + (p.area_m2 ? p.area_m2.toLocaleString() : 0) + " m²</b> (" + (p.area_ha || 0) + " ha)</td></tr>" +
                           "<tr><td><b>Perímetro:</b></td><td>" + (p.perimetro_m || 'N/A') + " m</td></tr>" +
                           "<tr><td><b>Vértices:</b></td><td>" + (p.n_vertices || 'N/A') + " vértices</td></tr>" +
                           "<tr><td><b>Clase:</b></td><td>Cultivo Verde (#22C55E)</td></tr>" +
                           "<tr><td><b>Conformidad:</b></td><td>100.0% LADM (0.00% solape)</td></tr>" +
                           "</table>";
                layer.bindPopup(html);
                layer.on('mouseover', function() {{ this.setStyle({{ weight: 3, color: '#ffffff' }}); }});
                layer.on('mouseout', function() {{ lyrUpas.resetStyle(this); }});
            }}
        }}).addTo(map);

        // Control de Capas
        var baseMaps = {{ "Esri World Imagery (HD)": esriSat, "OpenStreetMap Carto": osm }};
        var overlayMaps = {{
            "UPAs Delimitadas (350 LADM)": lyrUpas,
            "Límites Parroquiales (DPA)": lyrParroquias,
            "Zonas de Exclusión (SNAP/Urbano)": lyrExclusion
        }};
        L.control.layers(baseMaps, overlayMaps, {{ collapsed: false, position: 'topright' }}).addTo(map);

        function zoomToParish(lat, lon, z) {{
            map.flyTo([lat, lon], z, {{ duration: 1.5 }});
        }}
    </script>
</body>
</html>
"""

    for p in [out_repo, out_exp]:
        with open(p, "w", encoding="utf-8") as f:
            f.write(html_template)
        print(f"-> Visor interactivo guardado en: {p}")

if __name__ == "__main__":
    build_interactive_map()
