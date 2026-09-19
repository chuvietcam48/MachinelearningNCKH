"""
run_benchmark_junction.py
=========================
Benchmark runner that patches all registry data_paths to use
the C:\\MLData junction (ASCII path) so Python can open files
even though the project lives in a path with Vietnamese chars.

Usage (run from any directory):
    python C:\\MLData\\run_benchmark_junction.py
"""
import os, sys, logging, warnings
warnings.filterwarnings("ignore")

# ── Force ASCII-safe working directory ───────────────────────────────────────
os.chdir("C:\\MLData")
sys.path.insert(0, "C:\\MLData")

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
if hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

logger = logging.getLogger("benchmark_junction")

# ── Import registry AFTER path setup ──────────────────────────────────────────
from src.dataset_registry import _REGISTRY, list_datasets, register_dataset, get_dataset
from src.data_loader      import load_and_clean
from src.data_loader_tafeng import load_and_clean_tafeng
from src.data_loader_cdnow  import load_data as load_cdnow
from src.data_loader_x5     import load_data as load_x5

# ── Patch all data_paths to use C:\MLData junction ───────────────────────────
JUNCTION = "C:\\MLData"
DATA_ROOT = os.path.join(JUNCTION, "data", "raw")

PATH_OVERRIDES = {
    "uci":      os.path.join(DATA_ROOT, "Online Retail.xlsx"),
    "tafeng":   os.path.join(DATA_ROOT, "ta_feng_all_months_merged.csv"),
    "cdnow":    os.path.join(DATA_ROOT, "cdnow.csv"),
    "x5retail": os.path.join(DATA_ROOT, "x5retail", "purchases.csv"),
}

for ds_name, new_path in PATH_OVERRIDES.items():
    if ds_name in _REGISTRY:
        _REGISTRY[ds_name].data_path = new_path
        logger.info(f"[Patch] {ds_name} → {new_path}")

# ── Verify junction works on each path ────────────────────────────────────────
logger.info("\n--- Testing file access via junction ---")
for ds_name, path in PATH_OVERRIDES.items():
    try:
        size = os.path.getsize(path)
        # Real test: os.open with junction path
        fd = os.open(path, os.O_RDONLY | os.O_BINARY)
        _ = os.read(fd, 16)
        os.close(fd)
        logger.info(f"  ✅ {ds_name}: {path} ({size/1e6:.1f} MB) — readable")
    except Exception as e:
        logger.warning(f"  ❌ {ds_name}: {e}")

# ── Run benchmark ──────────────────────────────────────────────────────────────
from src.benchmark import run_benchmark

output_dir = os.path.join(JUNCTION, "outputs", "benchmark")
os.makedirs(output_dir, exist_ok=True)

logger.info("\n" + "=" * 60)
logger.info("  BENCHMARK — ALL 4 DATASETS")
logger.info("=" * 60)

bench_df = run_benchmark(
    datasets=["x5retail"],
    tau=0,           # dynamic per dataset
    output_dir=output_dir,
)

print("\n" + "=" * 60)
print("  RESULTS")
print("=" * 60)
print(bench_df.to_string(index=False))
md_path = os.path.join(output_dir, "benchmark_table.md")
print(f"\nMarkdown: {md_path}")
print(f"CSV:      {os.path.join(output_dir, 'benchmark_table.csv')}")
