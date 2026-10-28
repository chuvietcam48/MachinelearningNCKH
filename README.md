# Bounded Contribution of LLM-Derived Aspect Descriptors to Customer Re-Engagement Timing: An Equivalence Analysis
### *DSIT 2026 Replication Package and Reference Implementation*

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22847042.svg)](https://doi.org/10.5281/zenodo.22847042)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official replication codebase and dataset artifacts for the paper:
> **"Bounded Contribution of LLM-Derived Aspect Descriptors to Customer Re-Engagement Timing: An Equivalence Analysis"**  
> *Published / Presented at DSIT 2026*.  
> **Permanent Archival DOI:** [https://doi.org/10.5281/zenodo.22847042](https://doi.org/10.5281/zenodo.22847042)

---

## 🎯 Reproducing the DSIT 2026 Paper

All primary empirical results, two one-sided tests (TOST) for statistical equivalence, tables, and figures reported in the paper are reproducible directly from the scripts and artifacts in this repository:

### 1. Published Datasets & LLM Aspect Annotations
* **LLM Aspect Annotation File (`data/artifacts/semantic_labels/mass_results_v224_qa.jsonl`)**:
  Contains all 2,067 successfully parsed records produced by the Gemini-based ABSA extraction pipeline (prompt version v2.2.4, taxonomy v2.2.4, `gemini-2.5-flash`). Each record contains the item ID, provider extraction targets, final aspect-polarity tuples, review mixed flags, and model metadata. All personal API key hashes and raw customer review texts have been sanitized.
* **Episodic Snapshot Parquet (`data/artifacts/semantic_labels/verified_episode_snapshots_h270_v1.parquet`)**:
  Base cohort dataset of customer review episodes constructed under the 14-day anchored observation window and 270-day follow-up horizon.

### 2. Cohort Construction & Feature Engineering
* **Raw Review Episode Grouping (`src/cohort/cohort_builder.py`)**:
  Reference algorithm for clustering sequential customer reviews into engagement episodes using an anchored 14-day window ($t - t_{\text{start}} \le 14\text{ days}$).
* **Dataset Construction & Feature Extraction (`src/pipeline/02_dataset_construction.py`)**:
  Applies eligibility filters, extracts the 12 behavioral features (Model A), merges the 33 semantic aspect features (Model C), and partitions data chronologically into train / validation / test splits (768 / 164 / 159 episodes; 73 events in the test split).

### 3. Empirical Equivalence Tests & Table IV (Group Ablation)
* **Ablation Bootstrap Analysis (`scripts/ablation_harrell_cluster_bootstrap.py`)**:
  Implements the exact procedure specified in the manuscript:
  - Metric: Harrell Concordance Index (`sksurv.metrics.concordance_index_censored`).
  - Resampling: Customer-level (reviewer ID) cluster bootstrap ($B = 500$ resamples, seed = 42).
  - Models: Evaluates Model A vs. Model A + each of the 6 disjoint descriptor groups (33 total semantic variables), plus the full semantic block $\Delta C$ and $\Delta \text{AUC}_{270\text{d}}$.
  - Output: Exports all point estimates and 95% bootstrap confidence intervals directly to `results/ablation_33_groups.json`.

```bash
# Run the Table IV group ablation equivalence test
python scripts/ablation_harrell_cluster_bootstrap.py
```

### 4. Single-Panel Forest Plot (Figure 2)
* **Plotting Script (`generate_forest_plot.py` / `scripts/generate_single_panel_forest_plot.py`)**:
  Reads directly from `results/ablation_33_groups.json` and generates the publication-ready, single-panel forest plot showing $\Delta C$ equivalence bands ($\pm 0.03$), $\Delta \text{AUC}$ band ($\pm 0.04$), hollow marker for the out-of-margin Conflict group, and sparse-group dagger indicators.
  - Outputs: `images/forest_plot.pdf` (used in LaTeX `main_short.tex`), `figures/forest_plot.pdf`, and `figures/forest_plot.png`.

```bash
# Generate Figure 2 from computed results
python generate_forest_plot.py
```

---

## 🚀 Overview of the Decision Framework
This framework evaluates customer re-engagement timing and retention policy design under standard behavioral signals and optional semantic signals derived from natural language reviews:
- **Survival Engine**: Vectorized behavioral extraction and precision CoxPH/Weibull survival models.
- **Semantic Extension**: Extracts trajectory-aware semantic signals (aspect sentiment, conflict, streaks, dominance, dispersion) when textual reviews are available.
- **Equivalence Analysis**: Establishes whether rich semantic descriptors offer meaningful discrimination gains over simple behavioral baselines under pre-registered equivalence margins ($\Delta C \in [-0.03, +0.03]$, $\Delta \text{AUC} \in [-0.04, +0.04]$).

---

## 🛠️ Installation

### 1. Prerequisites
- Python 3.10+
- Git

### 2. Setup
```bash
# Clone repository
git clone https://github.com/chuvietcam48/MachinelearningNCKH.git
cd MachinelearningNCKH

# Create virtual environment (recommended)
python -m venv .venv
# Activate: Windows -> .venv\Scripts\activate | Linux/Mac -> source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 📂 Project Structure

```
MachinelearningNCKH/
├── data/
│   └── artifacts/
│       └── semantic_labels/
│           ├── mass_results_v224_qa.jsonl                 # Published LLM ABSA annotations (2,067 records)
│           └── verified_episode_snapshots_h270_v1.parquet # Episode snapshot cohort
├── src/
│   ├── cohort/
│   │   └── cohort_builder.py                              # Anchored episode clustering reference script
│   ├── pipeline/
│   │   ├── 02_dataset_construction.py                     # Episodic dataset construction & splitting
│   │   └── 03_survival_model.py                           # Baseline survival modeling
│   └── framework/
│       ├── behavior_engine.py                             # Behavioral feature extraction
│       └── survival_engine.py                             # CoxPH wrapper with penalizer=0.001
├── scripts/
│   ├── ablation_harrell_cluster_bootstrap.py              # Harrell C + customer-cluster bootstrap (Table IV)
│   ├── reproduce_ablation_33.py                           # Standalone 33-feature partition ablation verification
│   └── generate_single_panel_forest_plot.py               # Mirror of publication forest plot generator
├── results/
│   └── ablation_33_groups.json                            # Computed bootstrap results for Table IV and Fig. 2
├── images/
│   └── forest_plot.pdf                                    # Single-panel Figure 2 included by LaTeX
├── figures/
│   ├── forest_plot.pdf                                    # High-resolution vector plot
│   └── forest_plot.png                                    # 300 DPI preview image
├── generate_forest_plot.py                                # Figure 2 generator script
├── .zenodo.json                                           # Zenodo archival deposit metadata
├── README.md
└── requirements.txt
```

---

## 📊 Key Results (Amazon Case Study Summary)

* **Cohort Partitioning:** 1,091 total episodes chronologically partitioned into 768 training, 164 validation, and 159 test episodes (**73 true re-engagement events** in the test split).
* **Model Discrimination:**
  * Model A (Behavioral, 12 features): Harrell $C = 0.7540$ (95% CI $[0.693, 0.816]$), $\text{AUC}_{270\text{d}} = 0.7848$ (95% CI $[0.706, 0.858]$).
  * Model C (Behavioral + Semantic, 45 features): Harrell $C = 0.7576$ (95% CI $[0.697, 0.821]$), $\text{AUC}_{270\text{d}} = 0.7806$ (95% CI $[0.706, 0.852]$).
* **Paired Incremental Gains:**
  * $\Delta C = +0.0036$ (95% CI $[-0.0216, +0.0282]$, rounds to $[-0.022, +0.028]$). Falls entirely within the equivalence margin $[-0.03, +0.03]$.
  * $\Delta \text{AUC} = -0.0042$ (95% CI $[-0.0362, +0.0281]$, rounds to $[-0.036, +0.028]$). Falls entirely within the equivalence margin $[-0.04, +0.04]$.
* **Descriptor Group Ablation (Table IV):**
  Five of six groups fall strictly inside the $\pm 0.03$ margin. Only the sparse Conflict group (3 features) exceeds the margin ($[-0.035, +0.003]$), driven by extreme sparsity rather than predictive utility.

---

## 📝 Reference Notes

### 1. Cohort Construction Alignment (Anchored Episode Window)
* **Episode Clustering Logic (`src/cohort/cohort_builder.py`)**: Synchronized to the exact anchored window rule used in the published snapshot dataset (`verified_episode_snapshots_h270_v1.parquet`). Reviews by the same customer occurring within 14 days of the episode's initial review date ($t - t_{\text{start}} \le 14\text{ days}$) are aggregated into a single engagement episode, capping `episode_duration` at 14.0 days by construction.

### 2. Artifact Audit & Model Serialization Notice
* **Gate 10 Baseline Artifacts (`outputs/amazon_v5_rebuild/gate10_baseline/`)**: Explanatory note added (`README_MODELS.md`) clarifying that legacy serialized `.pkl` files reflect exploratory ElasticNet (`CoxnetSurvivalAnalysis`) checkpoints, whereas all official reported tables and ablation tests utilize `lifelines.CoxPHFitter` with pre-registered small ridge penalties ($\lambda = 0.001$).
