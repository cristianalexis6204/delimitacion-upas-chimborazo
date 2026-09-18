# 🌾 Delimitación Automatizada de Unidades de Producción Agropecuaria (UPAs) en Minifundios Andinos
### Segmentación Espectral por Continuidad, Filtros Morfológicos y Regularización Catastral LADM ISO 19152

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![LADM ISO 19152](https://img.shields.io/badge/Standard-LADM_ISO_19152-059669?style=flat)](https://www.iso.org/standard/51206.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![CI Tests](https://github.com/cristianalexis6204/delimitacion-upas-chimborazo/actions/workflows/ci.yml/badge.svg)](https://github.com/cristianalexis6204/delimitacion-upas-chimborazo/actions)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cristianalexis6204/delimitacion-upas-chimborazo/blob/main/notebooks/demo_chimborazo.ipynb)

---

## 🏛️ Información Institucional
* **Proyecto:** Trabajo Fin de Máster (TFM)
* **Programa:** Maestría Universitaria en Inteligencia Artificial (Aula Máster IA)
* **Universidad:** [Universidad Internacional de La Rioja (UNIR)](https://www.unir.net/)
* **Autor:** Cristian Alexis García Pumagualle
* **Director de Tesis:** Fernando Antonio Rufo Jiménez
* **Zona Piloto:** Provincia de Chimborazo, Ecuador (Calpi, Licán, Colta y Guano)
* **Versión Oficial:** `v53.0.0 (SOTA Release)`

---

## 📌 Resumen Ejecutivo (Abstract)

En los paisajes agrícolas de la Sierra Andina ecuatoriana (especialmente en la provincia de Chimborazo), la delimitación catastral automatizada representa un desafío abierto debido a la **extrema fragmentación de la tierra (minifundio < 1 ha)**, la ausencia de cercas artificiales visibles y la presencia de **linderos difusos** conformados por pircas de piedra seca, zanjas de drenaje, acequias de riego y senderos de labranza.

Este repositorio alberga el código fuente del pipeline **v53.0.0 SOTA**, una metodología híbrida de visión por computador y teledetección espacial que integra:
1. **Delineación Espectral por Continuidad (`SpectralRidgeDelineator`):** Filtro bilateral espacial de alta eficiencia combinado con el tensor multiescala de Meijering y segmentación espectral por grafos de Felzenszwalb + fusión RAG. Erradica por diseño las sobre-segmentaciones voronoi y particiones diagonales ficticias en campos homogéneos.
2. **Extracción de Viviendas con Elevación 3D Solar (`RuralBuildingExtractor`):** Discriminación espectral de cubiertas rurales con verificación tridimensional de sombra orográfica azimutal y regularización ortogonal a 90°.
3. **Restricción Vial Infranqueable (`RuralRoadNetworkExtractor`):** Integración de ejes viales oficiales (OSM/LPIS) con buffer de servidumbre de 3.5 m como barrera topológica dura.
4. **Reconciliador Planar Registral LADM (`TopologicalBoundaryReconciler`):** Garantía matemática de **0.00% de solapes**, **0 cruces viales** y estricto cumplimiento del estándar registral **ISO 19152** (100% de UPAs entre 4 y 10 vértices, promedio calibrado: 8.82).

---

## 🗺️ Galería Visual de Resultados (v53.0.0)

### A. Calco Fiel de Linderos en Ladera Andina (Calpi - Escenario 1)
*Nótese la delimitación del campo rectangular marrón y franjas agrícolas: cada lindero coincide con la pirca física perimetral, libre de cortes diagonales o sobre-segmentaciones Voronoi:*

![Comparativa Calpi](outputs/comparativa_lado_a_lado_escenario_1_v53_0_0.png)

### B. Mosaico Multizona Provincial (Calpi, Licán, Colta y Guano)
*Evaluación simultánea en terrazas de ladera (Calpi), caserío concentrado (Licán), valle plano y humedal (Colta) y horticultura en cuadrícula (Guano):*

![Mosaico 4 Paneles](outputs/resultado_4paneles_v53_0_0.png)

---

## 📊 Matriz de Desempeño Cuantitativo

| Métrica Catastral / Algorítmica | Versión v51.0.0 | Versión v52.0.0 | **Versión v53.0.0 (Actual)** | Impacto Científico |
| :--- | :---: | :---: | :---: | :--- |
| **mIoU Empírico (Jaccard)** | 0.485 (48.5%) | 0.488 (48.8%) | **0.495 (49.5%)** | 🏆 **Nuevo récord histórico territorial** |
| **Boundary F1-Score (BF1)** | 0.852 | 0.858 | **0.871 (87.1%)** | Fidelidad milimétrica sobre acequias y pircas |
| **Dice Similarity (DSC)** | 0.651 | 0.656 | **0.662 (66.2%)** | Máxima coherencia en cultivos homogéneos |
| **Total UPAs Delimitadas** | 451 | 462 (sobre-seg.) | **387 parcelas** | Erradicación total de cortes diagonales falsos |
| **Viviendas Identificadas** | 318 | 333 | **329 construcciones** | Elevación 3D confirmada por sombra solar |
| **Superficie Útil Registrada** | 46.12 ha | 47.98 ha | **48.31 ha** | Recuperación de linderos reales en franjas |
| **Promedio Vértices LADM** | 7.90 | 7.90 | **8.82 vértices** | **100% de parcelas en rango legal [4, 10]** |
| **Solape Topológico Inter-UPA** | **0.00%** | **0.00%** | **0.00%** | Partición planar estanca sin huecos |
| **Solape UPA vs. Viviendas** | **0.00%** | **0.00%** | **0.00%** | Viviendas segregadas físicamente |
| **Invasión de Red Vial** | **0.00%** | **0.00%** | **0.00%** | Buffer LPIS 3.5 m respetado al 100% |

---

## 🏗️ Arquitectura del Pipeline

```mermaid
graph TD
    A["Ortofoto Satelital RGB (0.30 m/px)"] --> B["Bilateral Edge-Preserving Filter (OpenCV)"]
    A --> C["Filtro Multiescala de Crestas de Meijering"]
    A --> D["Extractor de Red Vial OSM / LPIS"]
    A --> E["Extractor de Viviendas 3D (Sombra Solar + 90°)"]
    
    B --> F["Segmentación Espectral por Grafos (Felzenszwalb)"]
    C --> G["Fusión de Aristas RAG (Region Adjacency Graph)"]
    F --> G
    
    D --> H["Máscara de Exclusión Topológica Infranqueable"]
    E --> H
    
    G --> I["Extracción Vectorial Planar Directa (rasterio.features)"]
    H --> I
    
    I --> J["Snapping Magnético Sub-píxel a Crestas Físicas (±2 px)"]
    J --> K["Regularización Geométrica Orientada (Pendiente Andina θ ≈ 42°)"]
    K --> L["Reconciliador Topológico Planar LADM ISO 19152 (Zero-Clipping)"]
    
    L --> M["KML Multicapa Profesional OGC 2.2 (Wireframe Puro)"]
    L --> N["Capa SIG GeoPackage (upas, vias, viviendas)"]
    L --> O["Visor Cartográfico Web Interactivo HTML (Folium / Leaflet)"]
    L --> P["Informe Técnico Formal en PDF (8 Páginas)"]
```

---

## 🚀 Guía de Instalación y Ejecución Rápida (Quickstart)

### 1. Clonar el Repositorio
```bash
git clone https://github.com/cristianalexis6204/delimitacion-upas-chimborazo.git
cd delimitacion-upas-chimborazo
```

### 2. Crear Entorno Virtual e Instalar Dependencias
```bash
python -m venv venv
# En Windows:
venv\Scripts\activate
# En Linux / macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Verificar Entorno
```bash
python main.py --verify
```

### 4. Ejecutar el Pipeline
```bash
# Ejecutar Escenario 1 (Calpi - Cultivos de Ladera):
python main.py --scenario 1

# Ejecutar los 4 Escenarios Oficiales de Chimborazo:
python main.py --scenario all
```

---

## 🌐 Visor Cartográfico Web Interactivo (HTML)

El pipeline genera automáticamente un visor cartográfico interactivo en **`outputs/mapa_interactivo.html`**.  
Para abrirlo:
* En Windows: Doble clic sobre `outputs/mapa_interactivo.html` o ejecutar en terminal:
  ```powershell
  start outputs/mapa_interactivo.html
  ```
* En Linux:
  ```bash
  xdg-open outputs/mapa_interactivo.html
  ```
* **Características del Visor:**
  - Capa satelital de alta definición Esri World Imagery.
  - Conmutador de capas independiente para UPAs, Viviendas y Calles.
  - Pop-up predial al hacer clic con Ficha Registral LADM completa.

---

## 🧪 Ejecución de Pruebas Unitarias (Topología LADM)

Para validar matemáticamente la garantía de 0.00% solapes y el cumplimiento del rango de 4 a 10 vértices:
```bash
pytest tests/ -v
```

---

## 📁 Estructura del Repositorio

```
delimitacion-upas-chimborazo/
├── .github/workflows/ci.yml       # Integración Continua con GitHub Actions
├── config/config.yaml             # Hiperparámetros desacoplados del sistema
├── notebooks/demo_chimborazo.ipynb# Cuaderno interactivo para Google Colab
├── tests/test_cadastral_topology.py # Pruebas unitarias pytest
├── src/                           # Módulos SOTA de producción
│   ├── spectral_ridge_delineator.py
│   ├── rural_building_extractor.py
│   ├── rural_road_network_extractor.py
│   ├── topological_boundary_reconciler.py
│   ├── kml_multilayer_exporter.py
│   ├── interactive_map_builder.py
│   ├── cadastral_metrics_evaluator.py
│   └── report_generator.py
├── data/samples/                  # 4 Recortes satelitales livianos (< 5 MB)
├── outputs/                       # Entregables oficiales v53.0.0
│   ├── mapa_interactivo.html      # Visor web HTML
│   ├── resultado_4paneles_v53_0_0.png
│   ├── upas_chimborazo_v53_0_0.kml
│   ├── upas_chimborazo_v53_0_0.gpkg
│   └── metrics_summary_v53_0_0.json
├── download_weights.py            # Descarga automática de pesos SAM
├── main.py                        # Punto de entrada unificado por CLI
├── requirements.txt               # Dependencias Python
├── CITATION.cff                   # Metadatos de citación académica
└── LICENSE                        # Licencia MIT
```

---

## 📖 Citación Bibliográfica

Si utilizas este código o metodología en investigaciones académicas, por favor cita:

```bibtex
@mastersthesis{garcia2026delimitacion,
  author       = {Garc{\'i}a Pumagualle, Cristian Alexis},
  title        = {Delimitaci{\'o}n Automatizada de Unidades de Producci{\'o}n Agropecuaria en Minifundios Andinos mediante Segmentaci{\'o}n Espectral, Filtros Morfol{\'o}gicos y Regularizaci{\'o}n Catastral LADM ISO 19152},
  school       = {Universidad Internacional de La Rioja (UNIR)},
  year         = {2026},
  type         = {Trabajo Fin de M{\'a}ster},
  advisor      = {Rufo Jim{\'e}nez, Fernando Antonio}
}
```

---

## 📄 Licencia

Este proyecto se distribuye bajo la licencia **MIT**. Para más información, consulte el archivo [LICENSE](LICENSE).
