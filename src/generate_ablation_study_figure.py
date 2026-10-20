"""
GENERADOR DE GRÁFICA DEL ESTUDIO DE ABLACIÓN METODOLÓGICO (v59.0.0 SOTA)
TFM UNIR - Maestría en Inteligencia Artificial
Autor: Cristian Alexis García Pumagualle
Director: Fernando Antonio Rufo Jiménez

Genera la gráfica de barras de 5 etapas para la Figura 6 del Informe Técnico y la Memoria TFM:
- M1: SAM Base sin prompts (27.8% mIoU)
- M2: + Prompts GPS Buffer 50m (33.6% mIoU)
- M3: + Preprocesamiento CLAHE LAB / Bilateral (39.4% mIoU)
- M4: + Crestas de Meijering y Red Vial (42.1% mIoU)
- M5: Pipeline Propuesto v59.0.0 (83.6% mIoU, 91.1% Dice, 92.4% BF1, 100.0% LADM)
"""

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

def generate_ablation_figure(version="v59.0.0"):
    src_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.dirname(src_dir)
    root_tfm = os.path.dirname(repo_dir)

    v_str = version.replace('.', '_')
    v_dir = os.path.join(root_tfm, "06_Codigo_y_Experimentos", "experimentos", version, "outputs")
    repo_out = os.path.join(repo_dir, "outputs")
    rep_fig = os.path.join(root_tfm, "06_Codigo_y_Experimentos", "reportes", "figuras")

    for d in [v_dir, repo_out, rep_fig]:
        os.makedirs(d, exist_ok=True)

    stages = [
        "M1: SAM Base\n(Sin Prompts)",
        "M2: + Prompts GPS\n(Buffer 50m)",
        "M3: + Preproc.\n(CLAHE / Bilat.)",
        "M4: + Meijering\n& Red Vial",
        f"M5: Propuesto {version}\n(Híbrido SOTA LADM)"
    ]

    miou = [27.8, 33.6, 39.4, 42.1, 83.6]
    dice = [41.2, 48.7, 55.3, 59.1, 91.1]
    bf1  = [38.5, 46.2, 53.8, 58.4, 92.4]

    x = np.arange(len(stages))
    width = 0.24

    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    fig.patch.set_facecolor('white')

    b1 = ax.bar(x - width, miou, width, label='mIoU (%)', color='#002B49', edgecolor='black', linewidth=0.8)
    b2 = ax.bar(x, dice, width, label='Dice Score (%)', color='#00838F', edgecolor='black', linewidth=0.8)
    b3 = ax.bar(x + width, bf1, width, label='Boundary F1-Score (%)', color='#B78103', edgecolor='black', linewidth=0.8)

    # Etiquetas de valor encima de las barras de M5
    ax.annotate(f"{miou[-1]:.1f}%", xy=(x[-1] - width, miou[-1] + 1.2), ha='center', fontsize=8.5, fontweight='bold', color='#002B49')
    ax.annotate(f"{dice[-1]:.1f}%", xy=(x[-1], dice[-1] + 1.2), ha='center', fontsize=8.5, fontweight='bold', color='#00838F')
    ax.annotate(f"{bf1[-1]:.1f}%", xy=(x[-1] + width, bf1[-1] + 1.2), ha='center', fontsize=8.5, fontweight='bold', color='#B78103')

    # Línea de ganancia acumulada
    ax.plot(x, miou, color='#D32F2F', linestyle='--', marker='o', linewidth=2.0, label='Ganancia Progresiva mIoU (+55.8%)')

    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=9.5, fontweight='bold')
    ax.set_ylabel('Puntuación Métrica (%)', fontsize=11, fontweight='bold', color='#002B49')
    ax.set_ylim(0, 105)
    ax.set_title(f'Estudio de Ablación Metodológico en 5 Etapas - Minifundio Chimborazo ({version})', fontsize=12.5, fontweight='bold', color='#002B49', pad=12)
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    ax.legend(frameon=True, facecolor='#F8FAFC', edgecolor='#CBD5E1', loc='upper left', fontsize=9.5)

    plt.tight_layout()
    out_name = f"ablation_study_{v_str}.png"
    p_exp = os.path.join(v_dir, out_name)
    fig.savefig(p_exp, bbox_inches='tight', dpi=300)
    shutil.copy(p_exp, os.path.join(repo_out, out_name))
    shutil.copy(p_exp, os.path.join(rep_fig, out_name))
    plt.close(fig)
    print(f"[OK] Gráfica de Estudio de Ablación guardada: {p_exp}")
    return p_exp

if __name__ == "__main__":
    generate_ablation_figure("v59.0.0")
