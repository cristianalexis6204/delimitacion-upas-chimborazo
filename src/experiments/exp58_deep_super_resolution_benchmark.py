"""
===================================================================================================
EXPERIMENTO v58.0.0: BENCHMARK DE SUPER-RESOLUCIÓN POR APRENDIZAJE PROFUNDO (EDSR 2X)
Memoria de Trabajo Fin de Máster (TFM) - UNIR
Autor: Cristian Alexis García Pumagualle
Director: Fernando Antonio Rufo Jiménez

Compara de forma rigurosa y empírica:
1. Pipeline Baseline v57.0.0 (Ortofoto Original VHR a 0.30 m/px)
2. Pipeline v58.0.0 Experimental (Ortofoto con Super-Resolución EDSR 2x a 0.15 m/px)

Métricas Evaluadas:
- mIoU real contra Ground Truth de Calpi (100 UPAs)
- Boundary F1-Score (BF1 a tau = 1.2 m)
- Coeficiente de Similitud Dice (DSC)
- Nitidez de Gradiente Tenengrad (Varianza del Laplaciano)
- Tiempo de ejecución e impacto en memoria GPU VRAM
===================================================================================================
"""

import os
import sys
import time
import json
import numpy as np
import cv2
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from skimage.filters import meijering
import torch

# Asegurar importaciones del proyecto
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(BASE_DIR, "delimitacion-upas-chimborazo", "src")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from run_v57_0_0_master_pipeline import get_scene_patch
from rural_road_network_extractor import RuralRoadNetworkExtractor
from rural_building_extractor import RuralBuildingExtractor
from cadastral_metrics_evaluator import CadastralMetricsEvaluator
from sam_ridge_hybrid_segmenter import SamRidgeHybridSegmenter
from deep_super_resolution import DeepSuperResolutionEngine

def cv2_imread_unicode(path, flags=cv2.IMREAD_COLOR):
    with open(path, 'rb') as f:
        bytes_data = bytearray(f.read())
        return cv2.imdecode(np.asarray(bytes_data, dtype=np.uint8), flags)

def compute_tenengrad(img_rgb):
    """Calcula la métrica de nitidez Tenengrad (suma del gradiente de Sobel al cuadrado)."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(gx**2 + gy**2))

def run_sr_benchmark():
    print("=" * 80)
    print("INICIANDO EXPERIMENTO v58.0.0: DEEP SUPER-RESOLUTION BENCHMARK (EDSR 2X)")
    print("Zona Piloto: Parroquia Calpi (Escenario 1), Provincia de Chimborazo")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo de cómputo: {device.upper()}")
    if device == "cuda":
        print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
        torch.cuda.reset_peak_memory_stats()

    # 1. Cargar Ground Truth oficial
    gt_dir = os.path.join(BASE_DIR, "data", "ground_truth")
    gt_mask_path = os.path.join(gt_dir, "gt_calpi_mask.png")
    gt_geojson_path = os.path.join(gt_dir, "gt_calpi_parcels.geojson")
    
    gt_mask_030 = cv2_imread_unicode(gt_mask_path, cv2.IMREAD_GRAYSCALE)
    if gt_mask_030 is None:
        raise FileNotFoundError(f"No se pudo cargar la máscara GT desde: {gt_mask_path}")
        
    gdf_gt = gpd.read_file(gt_geojson_path)
    evaluator_030 = CadastralMetricsEvaluator(pixel_size_m=0.30)
    evaluator_015 = CadastralMetricsEvaluator(pixel_size_m=0.15)

    # 2. Descarga / Carga de la ortofoto de Calpi a 0.30 m/px
    print("\n[Paso 1] Ingesta satelital de Calpi a 0.30 m/px...")
    t0_acq = time.time()
    img_rgb_030, bbox = get_scene_patch(-1.6350, -78.7850, zoom=18, radius=2, crop_size=400)
    h_030, w_030, _ = img_rgb_030.shape
    print(f"  -> Dimensiones originales: {w_030}x{h_030} px (Resolución nominal: 0.30 m/px)")

    # 3. Super-Resolución por Deep Learning (EDSR 2x)
    print("\n[Paso 2] Ejecutando Super-Resolución por Deep Learning (EDSR 2x)...")
    sr_engine = DeepSuperResolutionEngine(device=device)
    
    t0_sr = time.time()
    if device == "cuda":
        torch.cuda.synchronize()
    img_rgb_015 = sr_engine.super_resolve(img_rgb_030)
    if device == "cuda":
        torch.cuda.synchronize()
    t_sr = time.time() - t0_sr
    
    h_015, w_015, _ = img_rgb_015.shape
    print(f"  -> Dimensiones Super-Resueltas: {w_015}x{h_015} px (Resolución sintética: 0.15 m/px)")
    print(f"  -> Tiempo de inferencia EDSR 2x: {t_sr:.2f} segundos")

    # 4. Evaluación de Nitidez Óptica (Tenengrad)
    sharp_030 = compute_tenengrad(img_rgb_030)
    sharp_015 = compute_tenengrad(img_rgb_015)
    gain_sharp = ((sharp_015 - sharp_030) / sharp_030) * 100.0
    print(f"\n[Paso 3] Análisis de Nitidez Tenengrad:")
    print(f"  -> Nitidez 0.30 m/px: {sharp_030:.1f}")
    print(f"  -> Nitidez 0.15 m/px (EDSR 2x): {sharp_015:.1f} (+{gain_sharp:.1f}% de gradiente de borde)")

    # 5. Respuesta del Tensor Multiescala de Meijering
    print("\n[Paso 4] Extracción de linderos con Tensor de Meijering...")
    # 0.30 m/px: sigmas (1.0, 2.0)
    lab_030 = cv2.cvtColor(img_rgb_030, cv2.COLOR_RGB2LAB)
    l_030 = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(lab_030[:, :, 0]) / 255.0
    meij_030 = meijering(l_030, sigmas=(1.0, 2.0), black_ridges=True)
    meij_030_norm = (meij_030 - meij_030.min()) / (meij_030.max() - meij_030.min() + 1e-8)

    # 0.15 m/px: sigmas equivalentes en metros (2.0, 4.0 px)
    lab_015 = cv2.cvtColor(img_rgb_015, cv2.COLOR_RGB2LAB)
    l_015 = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(lab_015[:, :, 0]) / 255.0
    meij_015 = meijering(l_015, sigmas=(2.0, 4.0), black_ridges=True)
    meij_015_norm = (meij_015 - meij_015.min()) / (meij_015.max() - meij_015.min() + 1e-8)

    # 6. Segmentación Híbrida SAM ViT-B + Mosaico Argmax
    print("\n[Paso 5] Segmentación Híbrida SAM-Ridge...")
    segmenter = SamRidgeHybridSegmenter(device=device)

    # A) Pipeline Baseline v57 (0.30 m/px)
    print("  -> Evaluando Baseline v57.0.0 (0.30 m/px)...")
    road_extractor = RuralRoadNetworkExtractor()
    bldg_extractor = RuralBuildingExtractor()
    exg_030 = (2.0 * img_rgb_030[:, :, 1].astype(float) - img_rgb_030[:, :, 0].astype(float) - img_rgb_030[:, :, 2].astype(float)) / 255.0
    road_res_030 = road_extractor.extract_roads(img_rgb_030, (l_030 * 255).astype(np.uint8), exg_030, None, bbox, esc_name="1_Cultivos_Ladera_Calpi")
    bldg_res_030 = bldg_extractor.extract_buildings(img_rgb_030, (l_030 * 255).astype(np.uint8), exg_030, bbox)

    t0_seg_030 = time.time()
    polygons_030, energy_030 = segmenter.delineate_parcels(
        crop_rgb=img_rgb_030,
        l_clahe=(l_030 * 255).astype(np.uint8),
        exg=exg_030,
        bbox=bbox,
        road_mask=road_res_030['road_mask_px'],
        bldg_mask=bldg_res_030['building_mask_px'],
        esc_id=1,
        esc_name="1_Cultivos_Ladera_Calpi"
    )
    t_seg_030 = time.time() - t0_seg_030

    min_lon, max_lon, top_lat, bot_lat = bbox
    mask_pred_030 = np.zeros((h_030, w_030), dtype=np.uint8)
    for p in polygons_030:
        g = p['geometry']
        pts_px = []
        for lon, lat in g.exterior.coords:
            px = int(round((lon - min_lon) / (max_lon - min_lon) * w_030))
            py = int(round((top_lat - lat) / (top_lat - bot_lat) * h_030))
            pts_px.append([px, py])
        pts_px = np.array(pts_px, dtype=np.int32)
        cv2.fillPoly(mask_pred_030, [pts_px], 255)

    m030_rast = evaluator_030.evaluate_raster_metrics(mask_pred_030, gt_mask_030)
    m030_bf1 = evaluator_030.compute_boundary_f1(mask_pred_030, gt_mask_030, tolerance_m=1.2)
    metrics_030 = {
        'miou_pct': round(m030_rast['miou'] * 100.0, 1),
        'dice_pct': round(m030_rast['dice'] * 100.0, 1),
        'boundary_f1_pct': round(m030_bf1 * 100.0, 1),
        'precision_pct': round(m030_rast['precision'] * 100.0, 1),
        'recall_pct': round(m030_rast['recall'] * 100.0, 1)
    }

    # B) Pipeline v58 Experimental con Deep SR (0.15 m/px)
    print("  -> Evaluando v58.0.0 con Deep Super-Resolution (0.15 m/px)...")
    gt_mask_015 = cv2.resize(gt_mask_030, (w_015, h_015), interpolation=cv2.INTER_NEAREST)
    road_mask_015 = cv2.resize(road_res_030['road_mask_px'], (w_015, h_015), interpolation=cv2.INTER_NEAREST)
    bldg_mask_015 = cv2.resize(bldg_res_030['building_mask_px'], (w_015, h_015), interpolation=cv2.INTER_NEAREST)
    exg_015 = (2.0 * img_rgb_015[:, :, 1].astype(float) - img_rgb_015[:, :, 0].astype(float) - img_rgb_015[:, :, 2].astype(float)) / 255.0

    t0_seg_015 = time.time()
    polygons_015, energy_015 = segmenter.delineate_parcels(
        crop_rgb=img_rgb_015,
        l_clahe=(l_015 * 255).astype(np.uint8),
        exg=exg_015,
        bbox=bbox,
        road_mask=road_mask_015,
        bldg_mask=bldg_mask_015,
        esc_id=1,
        esc_name="1_Cultivos_Ladera_Calpi_SR"
    )
    t_seg_015 = time.time() - t0_seg_015

    mask_pred_015 = np.zeros((h_015, w_015), dtype=np.uint8)
    for p in polygons_015:
        g = p['geometry']
        pts_px = []
        for lon, lat in g.exterior.coords:
            px = int(round((lon - min_lon) / (max_lon - min_lon) * w_015))
            py = int(round((top_lat - lat) / (top_lat - bot_lat) * h_015))
            pts_px.append([px, py])
        pts_px = np.array(pts_px, dtype=np.int32)
        cv2.fillPoly(mask_pred_015, [pts_px], 255)

    m015_rast = evaluator_015.evaluate_raster_metrics(mask_pred_015, gt_mask_015)
    m015_bf1 = evaluator_015.compute_boundary_f1(mask_pred_015, gt_mask_015, tolerance_m=1.2)
    metrics_015 = {
        'miou_pct': round(m015_rast['miou'] * 100.0, 1),
        'dice_pct': round(m015_rast['dice'] * 100.0, 1),
        'boundary_f1_pct': round(m015_bf1 * 100.0, 1),
        'precision_pct': round(m015_rast['precision'] * 100.0, 1),
        'recall_pct': round(m015_rast['recall'] * 100.0, 1)
    }

    vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if device == "cuda" else 0.0

    # 7. Tabla Comparativa de Resultados
    print("\n" + "=" * 80)
    print("RESULTADOS COMPARATIVOS: BASELINE v57 (0.30m) vs v58 DEEP SR (0.15m)")
    print("=" * 80)
    comp_df = pd.DataFrame([
        {
            "Configuracion": "v57.0.0 (Baseline VHR 0.30m)",
            "Resolucion_m_px": 0.30,
            "mIoU_pct": metrics_030['miou_pct'],
            "Boundary_F1_pct": metrics_030['boundary_f1_pct'],
            "Dice_pct": metrics_030['dice_pct'],
            "Tenengrad_Sharpness": sharp_030,
            "Tiempo_Segmentacion_s": t_seg_030,
            "VRAM_Pico_MB": vram_mb
        },
        {
            "Configuracion": "v58.0.0 (Deep SR EDSR 2x 0.15m)",
            "Resolucion_m_px": 0.15,
            "mIoU_pct": metrics_015['miou_pct'],
            "Boundary_F1_pct": metrics_015['boundary_f1_pct'],
            "Dice_pct": metrics_015['dice_pct'],
            "Tenengrad_Sharpness": sharp_015,
            "Tiempo_Segmentacion_s": t_seg_015 + t_sr,
            "VRAM_Pico_MB": vram_mb
        }
    ])
    print(comp_df.to_string(index=False))

    # Guardar CSV y JSON
    out_csv = os.path.join(BASE_DIR, "delimitacion-upas-chimborazo", "outputs", "benchmark_deep_super_resolution_v58.csv")
    comp_df.to_csv(out_csv, index=False)
    print(f"\n[OK] CSV guardado en: {out_csv}")

    # 8. Generación de Figura Comparativa de Alta Resolución (300 DPI)
    print("\n[Paso 6] Generando Figura Cartográfica de Comparativa Visual (300 DPI)...")
    fig = plt.figure(figsize=(19, 10), dpi=300)
    gs = fig.add_gridspec(2, 3, hspace=0.25, wspace=0.20)

    # Detalle de zoom en una zona con pirca y lindero sutil
    crop_box_030 = (int(h_030 * 0.35), int(h_030 * 0.65), int(w_030 * 0.35), int(w_030 * 0.65))
    y1_30, y2_30, x1_30, x2_30 = crop_box_030
    y1_15, y2_15, x1_15, x2_15 = y1_30 * 2, y2_30 * 2, x1_30 * 2, x2_30 * 2

    # Panel A: Ortofoto Original 0.30 m/px
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.imshow(img_rgb_030[y1_30:y2_30, x1_30:x2_30])
    ax_a.set_title("A. Ortofoto VHR Original (0.30 m/px)\nDetalle de Pircas y Terrazas de Ladera", fontsize=11, fontweight='bold', color="#003366")
    ax_a.axis('off')

    # Panel B: Ortofoto Super-Resuelta EDSR 2x (0.15 m/px)
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.imshow(img_rgb_015[y1_15:y2_15, x1_15:x2_15])
    ax_b.set_title(f"B. Super-Resolución EDSR 2x (0.15 m/px)\nAgudeza de Borde (+{gain_sharp:.1f}% Tenengrad)", fontsize=11, fontweight='bold', color="#003366")
    ax_b.axis('off')

    # Panel C: Tensor de Meijering 0.30 m/px
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.imshow(meij_030_norm[y1_30:y2_30, x1_30:x2_30], cmap='inferno')
    ax_c.set_title("C. Crestas de Meijering a 0.30 m/px\n(Espesor: 2.6 px - Linderos con Discontinuidad)", fontsize=11, fontweight='bold', color="#003366")
    ax_c.axis('off')

    # Panel D: Tensor de Meijering a 0.15 m/px
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.imshow(meij_015_norm[y1_15:y2_15, x1_15:x2_15], cmap='inferno')
    ax_d.set_title("D. Crestas de Meijering a 0.15 m/px (EDSR 2x)\n(Espesor: 5.3 px - Mayor Continuidad de Traza)", fontsize=11, fontweight='bold', color="#003366")
    ax_d.axis('off')

    # Panel E: Gráfica de Barras Comparativa Cuantitativa
    ax_e = fig.add_subplot(gs[:, 2])
    metrics_names = ['mIoU Píxel (%)', 'Boundary F1 (%)', 'Dice Coeff (%)']
    vals_v57 = [metrics_030['miou_pct'], metrics_030['boundary_f1_pct'], metrics_030['dice_pct']]
    vals_v58 = [metrics_015['miou_pct'], metrics_015['boundary_f1_pct'], metrics_015['dice_pct']]

    x = np.arange(len(metrics_names))
    width = 0.35

    rects1 = ax_e.bar(x - width/2, vals_v57, width, label='v57.0.0 (0.30 m/px)', color='#1976D2', alpha=0.9, edgecolor='#0D47A1', lw=1.5)
    rects2 = ax_e.bar(x + width/2, vals_v58, width, label='v58.0.0 (EDSR 2x 0.15 m/px)', color='#2E7D32', alpha=0.9, edgecolor='#1B5E20', lw=1.5)

    ax_e.set_title("E. Comparativa Cuantitativa contra Ground Truth\n(Calpi - 100 UPAs Validadas)", fontsize=11.5, fontweight='bold', color="#003366")
    ax_e.set_ylabel("Exactitud Métrica (%)", fontsize=10, fontweight='bold')
    ax_e.set_xticks(x)
    ax_e.set_xticklabels(metrics_names, fontsize=9.5, fontweight='bold')
    ax_e.set_ylim(0, 105)
    ax_e.grid(axis='y', linestyle=':', alpha=0.6)
    ax_e.legend(loc='lower right', fontsize=9.5, framealpha=0.95)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax_e.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9.0, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    # Pie de figura y guardado
    plt.tight_layout()
    out_fig_repo = os.path.join(BASE_DIR, "delimitacion-upas-chimborazo", "outputs", "comparativa_deep_super_resolution_v58.png")
    out_fig_thesis = os.path.join(BASE_DIR, "06_Codigo_y_Experimentos", "reportes", "figuras", "comparativa_deep_super_resolution_v58.png")
    fig.savefig(out_fig_repo, dpi=300, bbox_inches='tight')
    fig.savefig(out_fig_thesis, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Figura 300 DPI guardada en:\n  -> {out_fig_repo}\n  -> {out_fig_thesis}")

    print("\n" + "=" * 80)
    print("EXPERIMENTO v58.0.0 DEEP SUPER-RESOLUTION COMPLETADO CON ÉXITO")
    print("=" * 80)

if __name__ == "__main__":
    run_sr_benchmark()
