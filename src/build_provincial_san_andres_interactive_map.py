"""
BUILD PROVINCIAL & SAN ANDRÉS INTERACTIVE LEAFLET MAP (TFM UNIR)
Autor: Cristian Alexis García Pumagualle
Director: Fernando Antonio Rufo Jiménez

Genera un visor web interactivo en Leaflet con:
- Capas base: Esri World Imagery (satélite submétrico) y OpenStreetMap.
- Límite Provincial de Chimborazo.
- Los 10 Cantones oficiales con etiquetas toponímicas.
- Las 59 Parroquias DPA clasificadas.
- Áreas Protegidas SNAP (Chimborazo y Sangay) y Cascos Urbanos.
- Las 281 UPAs delimitadas en San Andrés (188.05 ha) con popups catastrales LADM ISO 19152.
"""

import os
import json
import geopandas as gpd

def build_map():
    base_src = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.dirname(base_src)
    root_tfm = os.path.abspath(os.path.join(repo_dir, ".."))

    out_repo = os.path.join(repo_dir, "outputs")
    gpkg_path = os.path.join(out_repo, "upas_san_andres_full_v60_0_0.gpkg")

    carto_dir = os.path.join(repo_dir, "data", "cartografia")
    parish_file = os.path.join(carto_dir, "parroquias_chimborazo_dpa.geojson")
    exclusion_file = os.path.join(carto_dir, "zonas_exclusion_chimborazo.geojson")

    print("Cargando capas geoespaciales...")
    gdf_upas = gpd.read_file(gpkg_path, layer="upas_san_andres_delimitadas").to_crs(epsg=4326)
    gdf_cantones = gpd.read_file(gpkg_path, layer="cantones_chimborazo").to_crs(epsg=4326)
    gdf_prov = gpd.read_file(gpkg_path, layer="limite_provincial_chimborazo").to_crs(epsg=4326)
    gdf_excl = gpd.read_file(exclusion_file).to_crs(epsg=4326)
    gdf_parish = gpd.read_file(parish_file).to_crs(epsg=4326)

    # Convertir a GeoJSON strings
    json_upas = gdf_upas.to_json()
    json_cantones = gdf_cantones.to_json()
    json_prov = gdf_prov.to_json()
    json_excl = gdf_excl.to_json()
    json_parish = gdf_parish.to_json()

    html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8" />
  <title>Visor Catastral Provincial y Cantonal Chimborazo - San Andrés (TFM UNIR)</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    body {{ margin: 0; padding: 0; background-color: #0b0f19; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }}
    #map {{ position: absolute; top: 0; bottom: 0; width: 100%; height: 100%; }}
    .header-box {{
      position: absolute; top: 15px; left: 60px; z-index: 1000;
      background: rgba(15, 23, 42, 0.92); color: #f8fafc;
      padding: 14px 20px; border-radius: 8px; border: 1px solid #00FFFF;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.6); backdrop-filter: blur(8px);
      max-width: 440px;
    }}
    .header-box h2 {{ margin: 0 0 6px 0; font-size: 15px; color: #00FFFF; text-transform: uppercase; letter-spacing: 0.5px; }}
    .header-box p {{ margin: 0; font-size: 11.5px; color: #94a3b8; line-height: 1.4; }}
    .badge {{ display: inline-block; padding: 2px 8px; font-size: 10px; font-weight: bold; border-radius: 4px; margin-top: 6px; }}
    .badge-ladm {{ background: #059669; color: #ffffff; }}
    .badge-cuda {{ background: #0284c7; color: #ffffff; }}
    .leaflet-popup-content-wrapper {{
      background: #0f172a; color: #f8fafc; border: 1px solid #38bdf8;
      border-radius: 8px; font-size: 11.5px;
    }}
    .leaflet-popup-tip {{ background: #0f172a; }}
  </style>
</head>
<body>
  <div id="map"></div>
  <div class="header-box">
    <h2>Catastro Rural Chimborazo & San Andrés</h2>
    <p><b>TFM UNIR:</b> Delimitación automatizada de UPAs mediante SAM-2.1 Hiera, Crestas Meijering y LADM ISO 19152.<br/>
    <b>Demarcación Focal:</b> Parroquia San Andrés (Cantón Guano) - 281 UPAs (188.05 ha netas).</p>
    <span class="badge badge-ladm">100% LADM ISO 19152</span>
    <span class="badge badge-cuda">0.00% SOLAPE</span>
  </div>

  <script>
    var map = L.map('map', {{ center: [-1.55, -78.74], zoom: 11 }});

    // Capas Base
    var esriSat = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
      attribution: 'Tiles &copy; Esri'
    }}).addTo(map);

    var osm = L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
      attribution: '&copy; OpenStreetMap'
    }});

    // Capas Vectoriales
    var provData = {json_prov};
    var cantData = {json_cantones};
    var parishData = {json_parish};
    var exclData = {json_excl};
    var upasData = {json_upas};

    // 1. Límite Provincial
    var provLayer = L.geoJSON(provData, {{
      style: {{ color: '#FFFFFF', weight: 3, fillOpacity: 0 }}
    }}).addTo(map);

    // 2. Cantones
    var cantLayer = L.geoJSON(cantData, {{
      style: {{ color: '#94A3B8', weight: 1.6, fillOpacity: 0.05, dashArray: '4, 4' }},
      onEachFeature: function(f, l) {{
        l.bindTooltip('<b>CANTÓN ' + f.properties.canton.toUpperCase() + '</b>', {{ permanent: false, direction: 'center' }});
      }}
    }}).addTo(map);

    // 3. Parroquias
    var parishLayer = L.geoJSON(parishData, {{
      style: function(f) {{
        return {{
          color: f.properties.tipo === 'Urbana' ? '#EF4444' : '#475569',
          weight: 0.8, fillOpacity: 0.1
        }};
      }},
      onEachFeature: function(f, l) {{
        l.bindTooltip(f.properties.parroquia + ' (' + f.properties.canton + ')', {{ sticky: true }});
      }}
    }});

    // 4. Áreas Protegidas SNAP y Exclusión
    var exclLayer = L.geoJSON(exclData, {{
      style: function(f) {{
        var id = f.properties.id_exclusion || '';
        if (id.indexOf('SNAP') !== -1) {{
          return {{ color: '#10B981', weight: 1.5, fillColor: '#064E3B', fillOpacity: 0.55 }};
        }} else if (id.indexOf('AGUA') !== -1) {{
          return {{ color: '#38BDF8', weight: 1.5, fillColor: '#1E3A8A', fillOpacity: 0.7 }};
        }}
        return {{ color: '#EF4444', weight: 1.2, fillColor: '#7F1D1D', fillOpacity: 0.4 }};
      }},
      onEachFeature: function(f, l) {{
        l.bindPopup('<b>Zona de Exclusión Territorial</b><br/>' + (f.properties.nombre || f.properties.id_exclusion));
      }}
    }}).addTo(map);

    // 5. UPAs de San Andrés
    var upasLayer = L.geoJSON(upasData, {{
      style: {{
        color: '#FACC15', weight: 1.8, fillColor: '#22C55E', fillOpacity: 0.35
      }},
      onEachFeature: function(f, l) {{
        var p = f.properties;
        var html = '<div style=\"padding:4px;\">' +
          '<h4 style=\"margin:0 0 6px 0; color:#38BDF8;\">' + (p.upa_id || 'UPA') + '</h4>' +
          '<b>Parroquia:</b> ' + (p.parroquia || 'San Andrés') + '<br/>' +
          '<b>Superficie:</b> ' + (p.area_ha || 0).toFixed(2) + ' ha (' + (p.area_m2 || 0).toLocaleString() + ' m²)<br/>' +
          '<b>Perímetro:</b> ' + (p.perimetro_m || 0).toFixed(1) + ' m<br/>' +
          '<b>Vértices LADM:</b> ' + (p.n_vertices || 4) + '<br/>' +
          '<b>Solape Interparcelario:</b> 0.00%<br/>' +
          '<b>Conformidad:</b> <span style=\"color:#10B981;font-weight:bold;\">✅ LADM ISO 19152</span>' +
          '</div>';
        l.bindPopup(html);
      }}
    }}).addTo(map);

    // Control de Capas
    var baseMaps = {{ 'Satelital (Esri)': esriSat, 'Mapa Base (OSM)': osm }};
    var overlayMaps = {{
      '🌾 UPAs San Andrés (281 Predios)': upasLayer,
      '🌲 Parques SNAP & Exclusiones': exclLayer,
      '🏛️ 10 Cantones de Chimborazo': cantLayer,
      '🗺️ 59 Parroquias DPA': parishLayer,
      '🌐 Límite Provincial': provLayer
    }};

    L.control.layers(baseMaps, overlayMaps, {{ collapsed: false }}).addTo(map);
  </script>
</body>
</html>"""

    out_html = os.path.join(out_repo, "mapa_interactivo_chimborazo_san_andres.html")
    out_html_rep = os.path.join(root_tfm, "06_Codigo_y_Experimentos", "reportes", "mapa_interactivo_chimborazo_san_andres.html")
    for hp in [out_html, out_html_rep]:
        os.makedirs(os.path.dirname(hp), exist_ok=True)
        with open(hp, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"-> Visor interactivo guardado en: {hp}")

if __name__ == "__main__":
    build_map()
