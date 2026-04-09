"""Diagnostic: capture full traceback from benchmark on each dataset separately."""
import sys, os, traceback, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
logging.basicConfig(level=logging.WARNING)

from src.benchmark import run_benchmark

for ds in ["tafeng", "cdnow", "x5retail", "uci"]:
    print(f"\n{'='*50}")
    print(f"  {ds.upper()}")
    print(f"{'='*50}")
    try:
        df = run_benchmark(datasets=[ds], tau=0, output_dir="outputs/benchmark_diag")
        print(f"  SUCCESS! Shape: {df.shape}")
        print(df.to_string(index=False))
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
