"""
run_benchmark.py
================
Standalone script: runs run_benchmark() across all 4 datasets and
saves outputs/benchmark/benchmark_table.md + benchmark_table.csv.

Usage:
    python run_benchmark.py
    python run_benchmark.py --datasets uci tafeng cdnow
"""
import os, sys, logging, argparse, warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

logger = logging.getLogger("run_benchmark")

# ── Parse args ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="4-Dataset Benchmark Runner")
parser.add_argument("--datasets", nargs="+", default=None,
                    help="Dataset keys to benchmark (default: all registered)")
parser.add_argument("--tau", type=int, default=0,
                    help="Fixed tau (0 = dynamic per dataset)")
parser.add_argument("--output-dir", default=os.path.join("outputs", "benchmark"),
                    help="Output directory for benchmark results")
args = parser.parse_args()

# ── Run ───────────────────────────────────────────────────────────────────────
from src.benchmark import run_benchmark

logger.info("=" * 60)
logger.info("  MULTI-DATASET BENCHMARK — ALL 4 DATASETS")
logger.info("=" * 60)

bench_df = run_benchmark(
    datasets=args.datasets,
    tau=args.tau,
    output_dir=args.output_dir,
)

print("\n" + "=" * 60)
print("  BENCHMARK COMPLETE")
print("=" * 60)
print(bench_df.to_string(index=False))
print(f"\nMarkdown report: {os.path.join(args.output_dir, 'benchmark_table.md')}")
print(f"CSV:             {os.path.join(args.output_dir, 'benchmark_table.csv')}")
