"""
================================================================================
GUÍA DE ESTILO ACADÉMICO UNIFICADO (TFM UNIR - ESTÁNDAR NEURIPS 2025 / GIS)
================================================================================
Define paletas accesibles (Okabe-Ito, ColorBrewer), configuración de fuentes,
barras de escala métricas, rosas náuticas, marcos de coordenadas y estética.
================================================================================
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import matplotlib.ticker as ticker
from matplotlib.patches import Polygon as MplPolygon, Rectangle, FancyBboxPatch
import numpy as np

# ------------------------------------------------------------------------------
# 1. PALETAS CROMÁTICAS CIENTÍFICAS Y ACCESIBLES (COLORBLIND-SAFE)
# ------------------------------------------------------------------------------
OKABE_ITO = {
    'black': '#000000',
    'orange': '#E69F00',
    'sky_blue': '#56B4E9',
    'bluish_green': '#009E73',
    'yellow': '#F0E442',
    'blue': '#0072B2',
    'vermillion': '#D55E00',
    'reddish_purple': '#CC79A7',
    'dark_navy': '#1A365D',
    'slate_gray': '#4A5568',
    'soft_bg': '#F7FAFC',
    'border_gray': '#CBD5E0'
}

CANTON_PALETTE = {
    'Riobamba': '#EBF5FB',
    'Guano': '#E8F8F5',
    'Colta': '#FEF9E7',
    'Guamote': '#F4ECF7',
    'Alausí': '#FDEDEC',
    'Chunchi': '#FEFDE8',
    'Pallatanga': '#FDF2E9',
    'Cumandá': '#EAFAF1',
    'Chambo': '#EAF2F8',
    'Penipe': '#FBEEE6'
}

# ------------------------------------------------------------------------------
# 2. CONFIGURACIÓN TIPOGRÁFICA Y DE RENDERIZADO (300 DPI)
# ------------------------------------------------------------------------------
def set_academic_style():
    """Configura los parámetros globales de Matplotlib para estándar de publicación."""
    plt.rcParams.update({
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica', 'Liberation Sans'],
        'mathtext.fontset': 'dejavusans',
        'font.size': 8.5,
        'axes.labelsize': 8.5,
        'axes.titlesize': 9.5,
        'xtick.labelsize': 7.5,
        'ytick.labelsize': 7.5,
        'legend.fontsize': 7.5,
        'figure.titlesize': 10.5,
        'axes.edgecolor': '#4A5568',
        'axes.linewidth': 0.8,
        'grid.color': '#CBD5E0',
        'grid.linestyle': ':',
        'grid.alpha': 0.6,
        'figure.facecolor': '#FFFFFF',
        'axes.facecolor': '#FFFFFF'
    })

# ------------------------------------------------------------------------------
# 3. COMPONENTES CARTOGRÁFICOS DE BUENAS PRÁCTICAS GIS
# ------------------------------------------------------------------------------
def draw_north_arrow(ax, x, y, size=0.04, fontsize=8.0):
    """
    Dibuja una rosa náutica sobria y orientada al Norte cartográfico.
    """
    # Flecha estilizada de doble ala
    ax.annotate(
        'N', xy=(x, y + size * 0.45), xytext=(x, y - size * 0.45),
        arrowprops=dict(facecolor='#1A365D', edgecolor='#FFFFFF', width=2.2, headwidth=6.5, headlength=7.5),
        ha='center', va='bottom', fontsize=fontsize, fontweight='bold', color='#1A365D',
        path_effects=[patheffects.withStroke(linewidth=2.2, foreground='white')],
        zorder=25
    )

def draw_scale_bar(ax, x, y, length_deg, label_mid, label_end, fontsize=7.5, height_deg=None):
    """
    Dibuja una barra de escala gráfica métrica alternada en blanco y negro.
    """
    if height_deg is None:
        height_deg = length_deg * 0.045
    half = length_deg / 2.0
    
    # Segmento 1: Negro
    rect1 = Rectangle((x, y), half, height_deg, facecolor='#1A365D', edgecolor='#FFFFFF', lw=0.6, zorder=24)
    # Segmento 2: Blanco
    rect2 = Rectangle((x + half, y), half, height_deg, facecolor='#FFFFFF', edgecolor='#1A365D', lw=0.6, zorder=24)
    ax.add_patch(rect1)
    ax.add_patch(rect2)
    
    # Textos de escala
    text_kwargs = dict(fontsize=fontsize, fontweight='bold', color='#1A365D', va='bottom', ha='center',
                       path_effects=[patheffects.withStroke(linewidth=2.0, foreground='white')], zorder=25)
    ax.text(x, y + height_deg * 1.35, "0", **text_kwargs)
    ax.text(x + half, y + height_deg * 1.35, label_mid, **text_kwargs)
    ax.text(x + length_deg, y + height_deg * 1.35, label_end, **text_kwargs)

def add_neatline_ticks(ax, x_range, y_range, x_step=0.2, y_step=0.2, is_metric=False):
    """
    Aplica formato graduado y limpio al canevá cartográfico.
    """
    ax.set_xlim(x_range[0], x_range[1])
    ax.set_ylim(y_range[0], y_range[1])
    ax.grid(True, linestyle=':', alpha=0.55, color='#718096')
    
    if not is_metric:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(x_step))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(y_step))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f° W'))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f° S'))
    else:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(x_step))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(y_step))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: f"{int(val):,}".replace(',', '.')))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: f"{int(val):,}".replace(',', '.')))
