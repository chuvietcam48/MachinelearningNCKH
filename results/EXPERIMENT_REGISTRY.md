# Experiment Registry

> **Codebase Frozen** after this point. No architecture changes, no refactors, no new features.

## Freeze Metadata

| Key | Value |
|-----|-------|
| Git Commit | `f56db88c2c7994ddd1e4dfd93f39280ac0b97fda` |
| Freeze Date | 2026-07-24 |
| Config Hash (SHA256) | `4D4A09D5E768A44D065D56C63565CA4D275167E34A2068DFBCC39C260DCAC9AE` |
| Python Version | 3.12.6 |
| Test Suite | 93/93 passed |

## Dataset Hashes (SHA256)

| Dataset | File | SHA256 |
|---------|------|--------|
| CDNOW | `data/raw/cdnow.csv` | `0829E4BC42E45B514C29B85F62E03D6F744B2556E545B1EEAE8B836D50F61259` |
| UCI Online Retail | `data/raw/Online Retail.xlsx` | `43465A06F2CCF7C8B5BD2892BC7DEFB52F97487934FE93B16AE4C3936424676D` |
| Ta Feng | `data/raw/ta_feng_all_months_merged.csv` | `1D575E5D0B7207D7706D22CA56C7535886FFF8175CA5537A310333A4AB7A7B67` |
| X5 Retail | `data/raw/x5retail/purchases.csv` | `A9E132E3DDE95655A0074622BB93AF0616B7C4F047E71C4CE7322E2FE6D262C1` |
| Amazon (Frozen Episodes) | `data/artifacts/semantic_labels/verified_episode_snapshots_h270_v1.parquet` | `881E0204F885DD3B68C082BCEEADDFF537916C2D938702D79626F46F8802644E` |

## Experiment Table

| ID | Experiment | Script | Dataset | Seed | Mode | Commit | Output |
|----|-----------|--------|---------|------|------|--------|--------|
| E1 | Behavioral Baseline (CDNOW) | `run_framework.py --dataset cdnow` | CDNOW | 42 | Behavioral | `f56db88` | `results/E1_cdnow/` |
| E2 | Behavioral Baseline (UCI) | `main.py` | UCI Online Retail | 42 | Behavioral | `f56db88` | `outputs/UCI_tau124/` |
| E3 | Behavioral Baseline (Ta Feng) | `main.py` | Ta Feng | 42 | Behavioral | `f56db88` | `outputs/TAFENG_tau39/` |
| E4 | Behavioral Baseline (X5 Retail) | `main.py` | X5 RetailHero | 42 | Behavioral | `f56db88` | `outputs/X5RETAIL_tau26/` |
| E5 | Semantic Validation (Amazon) | `run_framework.py --dataset amazon --enable-semantic` | Amazon CDS | 42 | Behavioral + Semantic | `f56db88` | `results/E5_amazon_semantic/` |
| E6 | Ablation: Model A/B/C × Full/Semantic | Frozen pipeline | Amazon CDS | 42 | Multi-tier | `f56db88` | `outputs/pipeline_freeze/` |

### Notes
- **E2/E3/E4**: Previously executed via old `main.py` pipeline. Results frozen in `outputs/`. Not re-executed.
- **E6**: Frozen with artifact hashes in `outputs/pipeline_freeze/experiment_registry.json`. Not re-executed.
- **E1**: Final run for export. Behavioral-only.
- **E5**: Final run for export. Requires NaN diagnostic → justify → fix → log workflow.

## Artifact Hashes (E6 — Frozen Pipeline)

| Artifact | SHA256 |
|----------|--------|
| `master_test_semantic.parquet` | `0fadbfe3c712d9b0c1412078561276a74b53a864ca4d4460d435c00a58a51506` |
| `prediction_model_c_semantic.parquet` | `dfc1550c860c85c1ec63d3f18d9098bf8dc4a526ff8b58966dd0568416f21eb9` |
| `hazard_ratios_semantic.csv` | `75c2d2968a33facec27ea0ba14d77b87b1f912036343773b6942b3add1999a9d` |
| `policy_routing_matrix.csv` | `f52bc3c51b6e95b2e89b6ef169b790c493b89be5820df8a702e333e1c7bc5f54` |
