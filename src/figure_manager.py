import os

def init_run_figure_dirs(run_id):
    sub_dirs = [
        "01_eda",
        "02_preprocessing",
        "03_training",
        "04_evaluation",
        "05_extended/convergence",
        "05_extended/experiments",
    ]
    for sd in sub_dirs:
        os.makedirs(f"reports/figures/student_segmentation/{run_id}/{sd}", exist_ok=True)
    
    heatmap_dirs = ["01_preprocessing", "02_elbow", "03_segmentation", "04_profile_cards"]
    for hd in heatmap_dirs:
        os.makedirs(f"reports/figures/additional_grade_heatmaps/{run_id}/{hd}", exist_ok=True)
