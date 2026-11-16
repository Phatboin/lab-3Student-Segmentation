import argparse
from src.tracking import generate_run_id
from src.figure_manager import init_run_figure_dirs
from src.eda import run_eda
from src.preprocess import run_preprocess
from src.train import run_train
from src.evaluate import run_evaluate
from src.experiments import run_experiments
from src.demo_convergence import run_convergence_demo
from src.report import generate_report

def main():
    parser = argparse.ArgumentParser(description="Student Segmentation Pipeline")
    parser.add_argument("--extended", action="store_true", help="Chạy chế độ mở rộng")
    args = parser.parse_args()

    run_id = generate_run_id()
    print(f"=== BẮT ĐẦU PIPELINE | Run_ID: {run_id} ===")
    init_run_figure_dirs(run_id)
    run_eda(run_id)
    run_preprocess(run_id)
    run_train(run_id)
    run_evaluate(run_id)

    if args.extended:
        run_experiments(run_id)
        run_convergence_demo(run_id)

    generate_report(run_id)
    print(f"=== HOÀN TẤT PIPELINE | Run_ID: {run_id} ===")

if __name__ == "__main__":
    main()
