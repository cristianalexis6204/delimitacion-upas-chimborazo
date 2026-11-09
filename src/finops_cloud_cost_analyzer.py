"""
================================================================================
MÓDULO DE ANÁLISIS FINANCIERO Y FINOPS CLOUD (TFM UNIR)
================================================================================
Auditoría económica basada en telemetría real de Compute Units de Google Colab Pro
frente a alternativas de infraestructura (GCP/AWS, Local Workstation y Terreno).
================================================================================
"""

import os
import json
import pandas as pd
import numpy as np

def run_finops_analysis(out_dir):
    """
    Calcula los modelos de costos multi-escala y exporta tabla CSV y resumen JSON.
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # --------------------------------------------------------------------------
    # Parámetros Operativos y Tarifarios
    # --------------------------------------------------------------------------
    P_CU = 9.99 / 100.0  # 0.0999 USD por Compute Unit en Google Colab Pro
    
    # Rendimiento medido en GPU NVIDIA A100-SXM4 (40 GB):
    # 568 UPAs en 207.16 s -> ~2.74 UPAs/s = 8.9 FPS en inferencia pura
    # ~25 s por clúster de 100 UPAs en inferencia pura GPU;
    # ~51.8 s por clúster incluyendo I/O, Meijering y Grafo Half-Edge en CPU.
    S_PER_CLUSTER_GPU = 25.0
    S_PER_CLUSTER_E2E = 51.8
    CU_PER_HOUR_A100 = 12.50
    CU_PER_HOUR_L4 = 5.20
    CU_PER_HOUR_T4 = 1.95
    
    scales = [
        {
            'Escala': 'Muestra Provincial Cochran (v106)',
            'UPAs': 2400,
            'Superficie_ha': 384.0,
            'Clusters': 24
        },
        {
            'Escala': 'Provincia de Chimborazo Completa',
            'UPAs': 81668,
            'Superficie_ha': 13066.88,
            'Clusters': 817
        },
        {
            'Escala': 'Nacional Ecuador (CNA RENAGRO)',
            'UPAs': 842882,
            'Superficie_ha': 134861.12,
            'Clusters': 8429
        }
    ]
    
    rows = []
    for sc in scales:
        n_upas = sc['UPAs']
        n_clust = sc['Clusters']
        
        # 1. Google Colab Pro (A100 SXM4 40GB)
        t_sec_e2e = n_clust * S_PER_CLUSTER_E2E
        t_hours = t_sec_e2e / 3600.0
        cus_a100 = t_hours * CU_PER_HOUR_A100
        cost_colab_a100 = cus_a100 * P_CU
        
        # 2. Google Colab Pro (L4 24GB - Inferencia 1.4x más lenta)
        t_hours_l4 = t_hours * 1.38
        cus_l4 = t_hours_l4 * CU_PER_HOUR_L4
        cost_colab_l4 = cus_l4 * P_CU
        
        # 3. Google Colab Pro (T4 16GB - Inferencia 2.4x más lenta)
        t_hours_t4 = t_hours * 2.35
        cus_t4 = t_hours_t4 * CU_PER_HOUR_T4
        cost_colab_t4 = cus_t4 * P_CU
        
        # 4. Google Cloud Platform dedicated (Vertex AI / Compute Engine a2-highgpu-1g)
        # On-Demand: 3.67 USD/hr | Spot: 1.10 USD/hr
        cost_gcp_ondemand = t_hours * 3.67
        cost_gcp_spot = t_hours * 1.10
        
        # 5. Workstation Local Soberana (CAPEX amortizado a 3 años)
        # Inversión: $14,500 USD (GPU 48GB VRAM + Threadripper + DDR5 + SAI)
        # Costo eléctrico: 0.65 kW * $0.092/kWh = $0.06/hr + amortización
        amort_por_hora = 14500.0 / (3 * 250 * 8)  # ~2.42 USD/hr
        cost_workstation_local = t_hours * (amort_por_hora + 0.06)
        
        # 6. Catastro Tradicional en Gabinete GIS (Digitalización manual experta)
        # Rendimiento: 6 UPAs / hora-técnico -> $12.50 USD / hora = ~2.08 USD / UPA
        cost_manual_gis = n_upas * 2.10
        hours_manual_gis = n_upas / 6.0
        
        # 7. Levantamiento Topográfico Directo en Campo (Brigadas GNSS-RTK)
        # Rendimiento: 1 UPA / hora-brigada -> $52.00 USD / UPA
        cost_field_gnss = n_upas * 52.00
        hours_field_gnss = n_upas * 1.0
        
        # Ahorro porcentual de Colab Pro A100 frente a alternativas
        ahorro_vs_manual = (1.0 - (cost_colab_a100 / cost_manual_gis)) * 100.0
        ahorro_vs_gnss = (1.0 - (cost_colab_a100 / cost_field_gnss)) * 100.0
        
        rows.append({
            'Escala Territorial': sc['Escala'],
            'UPAs Totales': n_upas,
            'Superficie (ha)': sc['Superficie_ha'],
            'Tiempo E2E A100 (horas)': round(t_hours, 2),
            'Compute Units (CUs)': round(cus_a100, 2),
            'OPEX Colab Pro A100 (USD)': round(cost_colab_a100, 2),
            'OPEX Colab Pro L4 (USD)': round(cost_colab_l4, 2),
            'OPEX Colab Pro T4 (USD)': round(cost_colab_t4, 2),
            'OPEX GCP Spot (USD)': round(cost_gcp_spot, 2),
            'OPEX GCP On-Demand (USD)': round(cost_gcp_ondemand, 2),
            'TCO Workstation Local (USD)': round(cost_workstation_local, 2),
            'Coste Gabinete GIS Manual (USD)': round(cost_manual_gis, 2),
            'Coste Campo GNSS-RTK (USD)': round(cost_field_gnss, 2),
            'Ahorro vs GIS Manual (%)': round(ahorro_vs_manual, 3),
            'Ahorro vs Campo GNSS (%)': round(ahorro_vs_gnss, 4)
        })
        
    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, 'analisis_financiero_cloud_colab_pro_v106.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8')
    
    summary_json = {
        'tarifa_colab_pro_usd_mes': 9.99,
        'compute_units_incluidas': 100,
        'costo_unitario_cu_usd': round(P_CU, 4),
        'rendimiento_gpu_a100_segundos_por_cluster': S_PER_CLUSTER_GPU,
        'rendimiento_e2e_segundos_por_cluster': S_PER_CLUSTER_E2E,
        'consumo_cu_por_hora_a100': CU_PER_HOUR_A100,
        'resultados_por_escala': rows
    }
    json_path = os.path.join(out_dir, 'telemetria_finops_colab_pro_v106.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary_json, f, indent=2, ensure_ascii=False)
        
    print(f"[FinOps] Tabla exportada: {csv_path}")
    print(f"[FinOps] JSON exportado: {json_path}")
    return df

if __name__ == '__main__':
    out = r"K:\Mi unidad\CURSOS Y CAPACITACIÓN PERSONAL\MASTER-INTELIGENCIA ARTIFICIAL\Aula Master IA\02_Segundo_Cuatrimestre\09_TFM\06_Codigo_y_Experimentos\experimentos\v106.0.0_muestra_completa_e2e\data_v106"
    run_finops_analysis(out)
