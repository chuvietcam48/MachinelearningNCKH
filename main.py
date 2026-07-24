import os
import sys
import argparse

# ── New Framework Architecture ────────────────────────────────────────────────
# main.py is preserved strictly for backwards compatibility with existing 
# execution scripts. It now delegates entirely to the Universal Framework.
# ──────────────────────────────────────────────────────────────────────────────

from run_framework import main as framework_main

def parse_args():
    parser = argparse.ArgumentParser(
        description="Legacy Orchestrator Wrapper. Redirects to Universal Framework."
    )
    parser.add_argument(
        "--dataset",
        default="uci",
        help="Dataset to run."
    )
    parser.add_argument(
        "--tau", type=int, default=0,
        help="Inactivity threshold (days). 0 = dynamic."
    )
    # Catch any old arguments and ignore them safely
    parser.add_argument('--no-shap', action='store_true')
    parser.add_argument('--sensitivity', action='store_true')
    parser.add_argument('--uplift', action='store_true')
    parser.add_argument('--no-uplift', action='store_false', dest='uplift')
    parser.add_argument('--no-mlflow', action='store_true')
    parser.add_argument('--cv', action='store_true')
    parser.add_argument('--sensitivity-penalty', action='store_true')
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--ablation', action='store_true')
    return parser.parse_args()

def main():
    print("=" * 70)
    print(" WARNING: main.py is deprecated.")
    print(" The pipeline has been upgraded to the Universal Customer Churn Decision Framework.")
    print(" Delegating execution to run_framework.py...")
    print("=" * 70)
    
    args = parse_args()
    sys.argv = [sys.argv[0], "--dataset", args.dataset]
    
    # Delegate to the framework
    framework_main()

if __name__ == "__main__":
    main()
