import argparse
import sys
from src.amazon_pipeline.config import logger, _sep, OUT
from src.amazon_pipeline.data_processing import run_phase0
from src.amazon_pipeline.vader_sentiment import run_phase1
from src.amazon_pipeline.absa_sentiment import run_phase2
from src.amazon_pipeline.survival_modeling import run_phase3
from src.amazon_pipeline.uplift_modeling import run_phase4
from src.amazon_pipeline.evi_simulation import run_phase5
from src.amazon_pipeline.ablation_summary import run_phase6

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", nargs="+", type=int, default=[0,1,2,3,4,5,6])
    parser.add_argument("--skip", nargs="+", type=int, default=[])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--absa-model", action="store_true")
    parser.add_argument("--x-learner", action="store_true")
    args = parser.parse_args()

    run_phases = sorted(list(set(args.phase) - set(args.skip)))
    logger.info(f"Running phases: {run_phases}  | x-learner={args.x_learner}  absa-model={args.absa_model}")

    if 0 in run_phases: run_phase0(force=args.force)
    if 1 in run_phases: run_phase1(force=args.force)
    if 2 in run_phases: run_phase2(use_absa_model=args.absa_model, force=args.force)
    if 3 in run_phases: run_phase3(force=args.force)
    if 4 in run_phases: run_phase4(force=args.force, run_xlearner=args.x_learner)
    if 5 in run_phases: run_phase5(force=args.force)
    if 6 in run_phases: run_phase6(force=args.force)

    # Auto-generate V4 Diagnostics Markdown File
    try:
        from generate_v4_diagnostics import generate_diagnostics
        logger.info("\nGenerating V4_DIAGNOSTICS.md...")
        generate_diagnostics()
    except Exception as e:
        logger.error(f"Failed to auto-generate diagnostics: {e}")

    _sep("PIPELINE COMPLETE")
    logger.info(f"Output : {OUT}/")
