import os
import sys
import json
import datetime
import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report_generator import generate_experiment_pdf_report

def compile_v58_report():
    base_src_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.dirname(base_src_dir)
    root_tfm = os.path.abspath(os.path.join(repo_dir, ".."))
    if not os.path.exists(os.path.join(root_tfm, "06_Codigo_y_Experimentos")):
        root_tfm = "K:\\Mi unidad\\CURSOS Y CAPACITACIÓN PERSONAL\\MASTER-INTELIGENCIA ARTIFICIAL\\Aula Master IA\\02_Segundo_Cuatrimestre\\09_TFM"

    v58_dir = os.path.join(root_tfm, "06_Codigo_y_Experimentos", "experimentos", "v58.0.0")
    v58_outputs_dir = os.path.join(v58_dir, "outputs")
    repo_outputs_dir = os.path.join(repo_dir, "outputs")

    metrics_json_path = os.path.join(v58_outputs_dir, "metrics_summary_v58_0_0.json")
    with open(metrics_json_path, "r", encoding="utf-8") as f:
        metrics_summary = json.load(f)

    meta_dict = {
        'version': 'v58.0.0',
        'date_human': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    print("[1/2] Compilando INFORME_TECNICO_v58.pdf con ReportLab en v58.0.0/...")
    pdf_path = generate_experiment_pdf_report(
        meta=meta_dict,
        outputs_dir=v58_outputs_dir,
        metrics=metrics_summary
    )
    print(f"  -> Generado exitosamente: {pdf_path} ({os.path.getsize(pdf_path)} bytes)")

    # Copiar también a delimitacion-upas-chimborazo/
    repo_pdf = os.path.join(repo_dir, "INFORME_TECNICO_v58.pdf")
    shutil.copy(pdf_path, repo_pdf)
    print(f"[2/2] Copiado a repositorio GitHub: {repo_pdf} ({os.path.getsize(repo_pdf)} bytes)")

    return pdf_path

if __name__ == "__main__":
    compile_v58_report()
