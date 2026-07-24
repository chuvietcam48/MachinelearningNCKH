# DEPRECATED

This entire directory has been deprecated and its functionality merged into `src/pipeline/`.

**Reason for Deprecation:**
The experimental scripts in this directory (formerly `amazon_v5_rebuild/`) represented the iterative development of the Semantic Trajectory Pipeline (Gates 1 through 11). To prepare for publication (Gate 16), these scripts have been refactored, consolidated, and frozen into a strict 6-step workflow-centric pipeline located at `src/pipeline/`.

Please use `src/pipeline/run_pipeline.py` to execute the end-to-end framework. This folder is kept strictly for historical reference and should not be modified or executed.
