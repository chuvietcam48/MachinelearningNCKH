# Execution Timeline

This timeline dictates the sequential dependencies of the experimental phase. 

```text
Phase 1: Environment Freeze
  └── Goal: Lock dataset, random seeds, and Universal Framework architecture.
  └── Status: ✅ Completed.

Phase 2: Baseline Reconstruction (E0)
  └── Goal: Code and execute the Original Framework strictly on the Amazon dataset.
  └── Output: `experiments/E0_original_framework/` artifacts.
  └── Status: ⏳ Pending.

Phase 3: Universal Prediction Ablation (E1, E2, E3)
  └── Goal: Run the Universal Pipeline to extract behavioral, VADER, and LLM predictive limits.
  └── Output: `model_metrics.json`, `hazard_ratios.csv`, `lrt.csv`.
  └── Status: ⏳ Pending.

Phase 4: Universal Decision Support (E4)
  └── Goal: Run the EVI simulations using the predictions generated in Phase 3.
  └── Output: Policy routing and ROI artifacts.
  └── Status: ⏳ Pending.

Phase 5: Statistical Consolidation
  └── Goal: Aggregate artifacts from E0-E4 into the finalized CSV tables (Tables 1-5).
  └── Output: `experiments/tables/` populated.
  └── Status: ⏳ Pending.

Phase 6: Manuscript Integration
  └── Goal: Inject finalized tables and plot figures (KM Curves, Forest Plots) directly into the LaTeX/Word manuscript.
  └── Status: ⏳ Pending.
```

**Dependency Note**: Phase 4 (Decision Support) strictly depends on the completion of Phase 3 (Prediction). However, Phase 2 (E0 Baseline) is entirely independent and can be executed in parallel with Phase 3.
