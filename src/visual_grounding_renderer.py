"""
================================================================================
VISUAL GROUNDING RENDERER v3.0 - INSPECCIÓN POR CUADRANTE A ALTITUD VIRTUAL 250m
Módulo Core: Rejilla Set-of-Mark Q{i}-A1..F6, Micro-Zooms 3x y Compositor Analítico
TFM UNIR - Delimitación Automatizada de UPAs en Chimborazo
================================================================================
"""

import os
import sys
import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict, Any

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

try:
    from google.genai import types
except ImportError:
    types = None

QUADRANT_DESCRIPTIONS = {
    1: "San Andrés Centro Urbano y Agrícola",
    2: "Chocaví Laderas y Terrazas Altas",
    3: "Pulucate Piedemonte y Quebradas Riparias",
    4: "Batzacón Interfase y Planicie"
}

def cv2_imread_unicode(path: str, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """Lectura segura de imágenes en Windows con soporte para tildes y caracteres no ASCII."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            raw_b = f.read()
        arr = np.frombuffer(raw_b, dtype=np.uint8)
        return cv2.imdecode(arr, flags)
    except Exception as e:
        print(f"[cv2_imread_unicode] Error leyendo {path}: {e}")
        return None

def cv2_imwrite_unicode(path: str, img: np.ndarray, quality: int = 92) -> bool:
    """Escritura segura de imágenes en Windows con soporte Unicode."""
    try:
        ext = os.path.splitext(path)[1].lower()
        if ext in ['.jpg', '.jpeg']:
            params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        elif ext == '.png':
            params = [cv2.IMWRITE_PNG_COMPRESSION, 4]
        else:
            params = []
        is_success, buf = cv2.imencode(ext, img, params)
        if not is_success:
            return False
        with open(path, "wb") as f:
            f.write(buf.tobytes())
        return True
    except Exception as e:
        print(f"[cv2_imwrite_unicode] Error guardando {path}: {e}")
        return False

def split_atlas_dupla_into_quadrants(dupla_path: str) -> List[np.ndarray]:
    """
    Divide una lámina Atlas en Duplas (4 filas x 2 paneles, aprox 6200 x 3100 px o 2400 x 4800 px)
    en 4 recortes independientes (uno por cuadrante Q1..Q4).
    """
    im = cv2_imread_unicode(dupla_path)
    if im is None:
        return []
    h, w = im.shape[:2]
    row_h = h // 4
    quads = []
    for i in range(4):
        quad_crop = im[i * row_h:(i + 1) * row_h, :].copy()
        quads.append(quad_crop)
    return quads

def overlay_single_quadrant_grid(
    quad_im: np.ndarray,
    quad_idx: int,
    grid_size: int = 6,
    focus_a: Optional[Tuple[int, int, int, int]] = None,
    focus_b: Optional[Tuple[int, int, int, int]] = None
) -> np.ndarray:
    """
    Superpone la rejilla alfanumérica Set-of-Mark Q{idx}-A1..F6 en ambos paneles (izquierdo y derecho)
    y dibuja los recuadros indicadores de ZOOM A (oro) y ZOOM B (cian).
    """
    out = quad_im.copy()
    h, w = out.shape[:2]
    panel_w = w // 2

    col_labels = ["A", "B", "C", "D", "E", "F"][:grid_size]
    row_labels = [str(i + 1) for i in range(grid_size)]
    
    # Asumir margen superior del título si existe
    y_start = int(h * 0.08)
    eff_h = h - y_start
    step_x = panel_w / float(grid_size)
    step_y = eff_h / float(grid_size)

    for p in range(2):
        offset_x = p * panel_w

        # Líneas verticales
        for j in range(1, grid_size):
            x = int(round(offset_x + j * step_x))
            cv2.line(out, (x, y_start), (x, h), (255, 255, 255), 1, cv2.LINE_AA)

        # Líneas horizontales
        for i in range(1, grid_size):
            y = int(round(y_start + i * step_y))
            cv2.line(out, (offset_x, y), (offset_x + panel_w, y), (255, 255, 255), 1, cv2.LINE_AA)

        # Etiquetas alfanuméricas
        for i in range(grid_size):
            for j in range(grid_size):
                cell_name = f"Q{quad_idx}-{col_labels[j]}{row_labels[i]}"
                cx = int(round(offset_x + j * step_x + 6))
                cy = int(round(y_start + i * step_y + 18))
                
                # Sombra negra de contraste
                cv2.putText(out, cell_name, (cx + 1, cy + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 2, cv2.LINE_AA)
                color = (255, 255, 255) if p == 0 else (255, 230, 0)
                cv2.putText(out, cell_name, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

        # Dibujar recuadros de enfoque si se suministran
        if focus_a:
            x1, y1, x2, y2 = focus_a
            cv2.rectangle(out, (offset_x + x1, y1), (offset_x + x2, y2), (0, 215, 255), 3, cv2.LINE_AA) # Oro
            cv2.putText(out, "ZOOM A (250m)", (offset_x + x1 + 4, max(y1 - 6, y_start + 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 215, 255), 2, cv2.LINE_AA)

        if focus_b:
            x1, y1, x2, y2 = focus_b
            cv2.rectangle(out, (offset_x + x1, y1), (offset_x + x2, y2), (255, 255, 0), 3, cv2.LINE_AA) # Cian
            cv2.putText(out, "ZOOM B (250m)", (offset_x + x1 + 4, max(y1 - 6, y_start + 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 0), 2, cv2.LINE_AA)

    return out

def build_quadrant_250m_zoom_sheet(
    quad_im: np.ndarray,
    quad_idx: int,
    zoom_foci: Dict[str, Any],
    out_path: Optional[str] = None
) -> np.ndarray:
    """
    Construye una lámina analítica de alta definición (2400 x 2000 px) para un único cuadrante:
    - Mitad Superior (2400 x 1120 px): Vista completa del cuadrante R=350m con rejilla Set-of-Mark y callouts.
    - Mitad Inferior (2400 x 880 px):
      * Panel Izquierdo: ZOOM A (Altitud Virtual 250m, 110x110m) [Ortofoto 3x vs Detección 3x]
      * Panel Derecho: ZOOM B (Altitud Virtual 250m, 110x110m) [Ortofoto 3x vs Detección 3x]
    """
    h_orig, w_orig = quad_im.shape[:2]
    w_half = w_orig // 2
    left_panel = quad_im[:, :w_half]
    right_panel = quad_im[:, w_half:]

    # Parámetros del mapa dentro del panel
    cx_map = int(w_half * 0.50)
    cy_map = int(h_orig * 0.54)
    r_map = int(min(w_half, h_orig) * 0.44)

    # Coordenadas normalizadas de los focos
    fa_norm = zoom_foci.get("focus_zoom_a", {}).get("norm", (-0.3, 0.3))
    fb_norm = zoom_foci.get("focus_zoom_b", {}).get("norm", (0.3, -0.3))
    cell_a = zoom_foci.get("focus_zoom_a", {}).get("celda", "B4")
    cell_b = zoom_foci.get("focus_zoom_b", {}).get("celda", "E2")

    # Tamaño de la ventana de micro-zoom: 110m de terreno (~15% del diámetro de 700m)
    half_box = int(round(r_map * (110.0 / 700.0)))
    half_box = max(80, min(150, half_box))

    def extract_crop_box(norm_coords):
        u, v = norm_coords
        px = int(round(cx_map + u * r_map))
        py = int(round(cy_map + v * r_map))
        y1 = max(0, py - half_box)
        y2 = min(h_orig, py + half_box)
        x1 = max(0, px - half_box)
        x2 = min(w_half, px + half_box)
        return (x1, y1, x2, y2)

    box_a = extract_crop_box(fa_norm)
    box_b = extract_crop_box(fb_norm)

    # 1. Preparar la mitad superior con la rejilla y los recuadros
    quad_with_grid = overlay_single_quadrant_grid(quad_im, quad_idx, grid_size=6, focus_a=box_a, focus_b=box_b)
    
    # Redimensionar la mitad superior a 2400 x 1120
    top_section = cv2.resize(quad_with_grid, (2400, 1120), interpolation=cv2.INTER_AREA)

    # 2. Extraer recortes y magnificarlos 3x con interpolación Lanczos4
    crop_a_rgb = left_panel[box_a[1]:box_a[3], box_a[0]:box_a[2]]
    crop_a_det = right_panel[box_a[1]:box_a[3], box_a[0]:box_a[2]]
    crop_b_rgb = left_panel[box_b[1]:box_b[3], box_b[0]:box_b[2]]
    crop_b_det = right_panel[box_b[1]:box_b[3], box_b[0]:box_b[2]]

    target_zoom_w, target_zoom_h = 570, 700
    z_a_rgb = cv2.resize(crop_a_rgb, (target_zoom_w, target_zoom_h), interpolation=cv2.INTER_LANCZOS4)
    z_a_det = cv2.resize(crop_a_det, (target_zoom_w, target_zoom_h), interpolation=cv2.INTER_LANCZOS4)
    z_b_rgb = cv2.resize(crop_b_rgb, (target_zoom_w, target_zoom_h), interpolation=cv2.INTER_LANCZOS4)
    z_b_det = cv2.resize(crop_b_det, (target_zoom_w, target_zoom_h), interpolation=cv2.INTER_LANCZOS4)

    # Función para dibujar barra de escala métrica (20m en la imagen ampliada)
    def add_scale_bar_and_labels(img, title_text, is_det=False):
        res = img.copy()
        # Header del panel
        cv2.rectangle(res, (0, 0), (target_zoom_w, 42), (15, 23, 42), -1)
        color_title = (0, 229, 255) if is_det else (255, 255, 255)
        cv2.putText(res, title_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color_title, 2, cv2.LINE_AA)
        
        # Barra de escala métrica 20m en la esquina inferior izquierda
        # 110m son target_zoom_w (570px) -> 20m son ~104px
        bar_len = int(round(570.0 * (20.0 / 110.0)))
        bx, by = 18, target_zoom_h - 22
        cv2.rectangle(res, (bx - 4, by - 16), (bx + bar_len + 50, by + 10), (0, 0, 0), -1)
        cv2.line(res, (bx, by), (bx + bar_len, by), (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(res, "20 m", (bx + bar_len + 8, by + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        return res

    za_l = add_scale_bar_and_labels(z_a_rgb, "Ortofoto Satelital (3x Zoom)", is_det=False)
    za_r = add_scale_bar_and_labels(z_a_det, "Detección Vectorial (3x Zoom)", is_det=True)
    zb_l = add_scale_bar_and_labels(z_b_rgb, "Ortofoto Satelital (3x Zoom)", is_det=False)
    zb_r = add_scale_bar_and_labels(z_b_det, "Detección Vectorial (3x Zoom)", is_det=True)

    block_zoom_a = np.hstack([za_l, za_r]) # 1140 x 700
    block_zoom_b = np.hstack([zb_l, zb_r]) # 1140 x 700

    # Crear marco contenedor inferior de 2400 x 880
    bottom_section = np.full((880, 2400, 3), (15, 23, 42), dtype=np.uint8)

    # Título separador de los micro-zooms
    banner_text = "INSPECCIÓN FOCALIZADA A ALTITUD VIRTUAL DE 250 m (CAMPO DE VISIÓN: 110 m x 110 m)"
    cv2.putText(bottom_section, banner_text, (650, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (56, 189, 248), 2, cv2.LINE_AA)

    # Título específico Zoom A
    t_a = f"ZOOM A: FOCO PERI-DOMÉSTICO Y VIVIENDAS (Celda Q{quad_idx}-{cell_a})"
    cv2.putText(bottom_section, t_a, (40, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 215, 255), 2, cv2.LINE_AA)

    # Título específico Zoom B
    t_b = f"ZOOM B: FOCO MINIFUNDIO Y LINDEROS FINOS (Celda Q{quad_idx}-{cell_b})"
    cv2.putText(bottom_section, t_b, (1260, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2, cv2.LINE_AA)

    # Incrustar bloques de zoom
    # Dejar márgenes laterales
    bottom_section[85:785, 40:1180] = cv2.resize(block_zoom_a, (1140, 700))
    bottom_section[85:785, 1260:2400] = cv2.resize(block_zoom_b, (1140, 700))

    # Pie informativo
    desc_q = QUADRANT_DESCRIPTIONS.get(quad_idx, f"Cuadrante {quad_idx}")
    footer_text = f"Tribunal MoE-Kaizen v3.0 | Cuadrante {quad_idx}: {desc_q} | GSD Nativo: 0.28 m/px | Altitud Virtual Zoom: 250 m"
    cv2.putText(bottom_section, footer_text, (40, 840), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (148, 163, 184), 1, cv2.LINE_AA)

    # Ensamblar lámina final vertical de 2400 x 2000
    composite_sheet = np.vstack([top_section, bottom_section])

    if out_path:
        cv2_imwrite_unicode(out_path, composite_sheet, quality=92)

    return composite_sheet

def prepare_quadrant_zoom_parts_for_gemini(
    dupla_path: str,
    zoom_foci_dict: Dict[str, Any],
    export_dir: Optional[str] = None
) -> List[Any]:
    """
    Genera las 4 láminas analíticas con micro-zoom (una por cada cuadrante Q1..Q4)
    y retorna una lista de 4 partes `types.Part` preparadas para Gemini VLM.
    """
    quads = split_atlas_dupla_into_quadrants(dupla_path)
    if not quads:
        return []

    parts = []
    base_name = os.path.splitext(os.path.basename(dupla_path))[0]

    for idx, q_im in enumerate(quads, 1):
        q_key = f"Q{idx}"
        foci = zoom_foci_dict.get(q_key, {})
        
        out_file = None
        if export_dir:
            os.makedirs(export_dir, exist_ok=True)
            out_file = os.path.join(export_dir, f"{base_name}_{q_key}_zoom250m.jpg")

        sheet = build_quadrant_250m_zoom_sheet(q_im, quad_idx=idx, zoom_foci=foci, out_path=out_file)

        if types:
            _, buf = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
            part = types.Part.from_bytes(data=buf.tobytes(), mime_type="image/jpeg")
            parts.append(part)

    return parts

def prepare_individual_quadrant_dupla_parts(
    exp_dir: str,
    layer_num: int,
    max_dim: int = 2400
) -> List[Any]:
    """
    Carga las 4 duplas individuales desacopladas por cuadrante para la capa especificada:
    dupla_capa{layer_num:02d}_*_cuadrante_{q}.png (q=1..4).
    Retorna 4 partes types.Part preparadas para Gemini VLM con calidad óptima.
    """
    if types is None:
        return []
        
    parts = []
    cand_dirs = [os.path.join(exp_dir, "duplas_por_cuadrante"), exp_dir]
    
    for q in range(1, 5):
        prefix = f"dupla_capa{layer_num:02d}_"
        suffix = f"_cuadrante_{q}.png"
        found_path = None
        for d in cand_dirs:
            if not os.path.exists(d):
                continue
            for f in os.listdir(d):
                if f.startswith(prefix) and f.endswith(suffix):
                    found_path = os.path.join(d, f)
                    break
            if found_path:
                break
                
        if found_path and os.path.exists(found_path):
            im = cv2_imread_unicode(found_path)
            if im is not None:
                h, w = im.shape[:2]
                if max(h, w) > max_dim:
                    scale = max_dim / float(max(h, w))
                    im = cv2.resize(im, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                _, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 88])
                part = types.Part.from_bytes(data=buf.tobytes(), mime_type="image/jpeg")
                parts.append(part)
                
    return parts

def prepare_atlas_part_with_grid_for_gemini(dupla_path: str, max_dim: int = 2800) -> Optional[Any]:
    """Fallback legacy: codifica la lámina completa en un único types.Part."""
    im = cv2_imread_unicode(dupla_path)
    if im is None or types is None:
        return None
    h, w = im.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        im = cv2.resize(im, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return types.Part.from_bytes(data=buf.tobytes(), mime_type="image/jpeg")

def compute_differential_delta_map(img_prev_path: str, img_curr_path: str) -> Optional[np.ndarray]:
    """Calcula el mapa de cambio diferencial entre dos láminas consecutivas (Verde: Ganado, Rojo: Perdido)."""
    im_prev = cv2_imread_unicode(img_prev_path)
    im_curr = cv2_imread_unicode(img_curr_path)
    if im_prev is None or im_curr is None:
        return None
    if im_prev.shape != im_curr.shape:
        im_prev = cv2.resize(im_prev, (im_curr.shape[1], im_curr.shape[0]), interpolation=cv2.INTER_AREA)

    w_half = im_curr.shape[1] // 2
    right_prev = im_prev[:, w_half:].astype(np.float32)
    right_curr = im_curr[:, w_half:].astype(np.float32)
    diff = np.abs(right_curr - right_prev)
    diff_mag = np.mean(diff, axis=2)

    delta_vis = im_curr.copy()
    mask_change = diff_mag > 25.0
    curr_mag = np.mean(right_curr, axis=2)
    prev_mag = np.mean(right_prev, axis=2)
    gained = mask_change & (curr_mag > prev_mag)
    lost = mask_change & (prev_mag > curr_mag)

    delta_panel = delta_vis[:, w_half:].copy()
    delta_panel[gained] = [0, 255, 0]
    delta_panel[lost] = [0, 0, 255]
    delta_vis[:, w_half:] = cv2.addWeighted(delta_vis[:, w_half:], 0.65, delta_panel, 0.35, 0)
    return delta_vis

if __name__ == "__main__":
    test_dupla = r"K:\Mi unidad\CURSOS Y CAPACITACIÓN PERSONAL\MASTER-INTELIGENCIA ARTIFICIAL\Aula Master IA\02_Segundo_Cuatrimestre\09_TFM\06_Codigo_y_Experimentos\experimentos\v100.0.0\dupla_01_viviendas_ladm.png"
    out_dir = r"K:\Mi unidad\CURSOS Y CAPACITACIÓN PERSONAL\MASTER-INTELIGENCIA ARTIFICIAL\Aula Master IA\02_Segundo_Cuatrimestre\09_TFM\06_Codigo_y_Experimentos\experimentos\v100.0.0\quad_zooms_test"
    dummy_foci = {
        "Q1": {"focus_zoom_a": {"norm": (-0.534, 0.423), "celda": "B5"}, "focus_zoom_b": {"norm": (-0.281, 0.558), "celda": "C5"}},
        "Q2": {"focus_zoom_a": {"norm": (0.187, -0.755), "celda": "D1"}, "focus_zoom_b": {"norm": (0.694, -0.148), "celda": "F3"}},
        "Q3": {"focus_zoom_a": {"norm": (-0.374, -0.031), "celda": "B3"}, "focus_zoom_b": {"norm": (0.067, 0.027), "celda": "D4"}},
        "Q4": {"focus_zoom_a": {"norm": (-0.254, -0.579), "celda": "C2"}, "focus_zoom_b": {"norm": (-0.003, -0.743), "celda": "C1"}}
    }
    parts = prepare_quadrant_zoom_parts_for_gemini(test_dupla, dummy_foci, export_dir=out_dir)
    print(f"Generadas {len(parts)} partes con micro-zooms a altitud virtual de 250m.")
