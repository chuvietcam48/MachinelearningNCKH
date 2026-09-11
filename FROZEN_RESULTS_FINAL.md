# FROZEN RESULTS — 2026-09-11 12:02:09+07:00
Git commit: 91d196e665305f4c42f50897d99f07d6dcf704d9
Người tạo / script chạy cuối: Antigravity Agent & pipeline v5

## ⚠️ QUY TẮC: File này là nguồn số liệu DUY NHẤT được dùng để viết bài.
## Nếu số liệu ở đây khác với bất kỳ file .md/report nào khác trong repo,
## SỐ Ở ĐÂY LÀ ĐÚNG, các file khác coi như nháp/lịch sử, không dùng.

## 1. Amazon — Cohort
- **N episodes:** 1091 (train=768, val=164, test=159)
- **Events ở test:** 80
- **Feature set Model A (12 biến):** `episode_duration`, `days_since_previous_episode`, `prior_verified_episode_count`, `prior_review_count`, `review_frequency`, `episode_min_rating`, `episode_max_rating`, `episode_mean_rating`, `episode_review_count`, `historical_mean_rating`, `recent_low_rating_count`, `customer_lifetime`.
- **Feature set Model C (Model A + 33 biến semantic):** `dominance_ratio`, `aspect_entropy`, `net_sentiment`, `sentiment_profile`, `semantic_variance`, `has_conflict`, `cross_aspect_conflict`, `same_aspect_conflict`, `negative_streak`, `positive_streak`, `aspect_switch_rate`, `aspect_transition_pattern`, `days_since_last_negative`, `previous_negative_count`, `previous_total_aspects`, `sentiment_flip_rate`, `rolling_negative_ratio`, và 16 biến đếm positive/negative cho 8 khía cạnh.
- **Penalizer:** 0.001
- **Script tạo cohort:** `src/pipeline/02_dataset_construction.py` (sử dụng base logic từ `archive_legacy/src/_deprecated_amazon_v5_rebuild/utils.py`)

## 2. Amazon — Kết quả chính
- **Model A:** C-index = 0.7540 (95% CI [0.693, 0.816]) | AUC@270 = 0.7848 (95% CI [0.706, 0.858])
- **Model C:** C-index = 0.7576 (95% CI [0.697, 0.821]) | AUC@270 = 0.7806 (95% CI [0.706, 0.852])
- **Paired Δ (Model C - A):** ΔC = +0.0036 (95% CI [-0.021, 0.028]) | ΔAUC = -0.0042 (95% CI [-0.036, 0.028])
- **Script tạo:** `scratch/run_pipeline.py` gọi hàm từ `src/pipeline/10_retrospective_baseline.py` và `src/framework/survival_evaluator.py` (với 500 vòng Bootstrap CI).

## 3. Amazon — Sub-models (semantic-only, rating-only, temporal-only)
- **Semantic-only Model:** C-index = 0.5102 (95% CI [0.447, 0.569])
- **Rating-only Model:** C-index = 0.538 (95% CI [0.468, 0.607])
- **Temporal-only Model:** C-index = 0.7529 (95% CI [0.692, 0.814])
- **Script tạo:** `scratch/check_stats.py` và `scratch/analyze_survival_data.py`.

## 4. Amazon — LRT, Ablation
- **Likelihood-Ratio Test (LRT) giữa Model A và C:** $\chi^2 = 18.84$, $df = 33$, $p = 0.9771$
- **Minimum Detectable Effect (MDE) cho 80 events:** $\Delta C \approx 0.05$ (dựa trên sample size rule of thumb).
- **Script tạo:** `scratch/check_stats.py` (tính từ model log-likelihoods).

## 5. Olist — Cohort & kết quả
- **Cohort (Chronological Split):** N = 15,000 khách hàng.
- **Vấn đề cấu trúc:** Test set bị giới hạn thời gian quan sát lớn nhất (max observation window) chỉ còn 102 ngày.
- **Hệ quả sự kiện:** Chỉ bắt được 22 true events (tỉ lệ 0.73%).
- **MDE (Olist):** Đội lên tới $\Delta C = 0.2149$.
- **Kết quả C-index (Cả 2 mô hình):** Xoay quanh $\approx 0.50$ (Mức Random Guess).
- **Script tạo:** `scratch/check_mass.py` (audit tập Olist).

## 6. Files không push được lên git (local only)
- **Dữ liệu lớn:** `outputs/amazon_v5_rebuild/gate10_baseline/master_train_semantic.parquet`, `master_test_semantic.parquet`, và `feature_registry.json`.
- **Lý do:** Kích thước lớn và định dạng nhị phân (.parquet) bị chặn bởi `.gitignore`.
- **Cách tái tạo:** Chạy lại file `src/pipeline/02_dataset_construction.py` trên máy local, toàn bộ logic và feature sets sẽ được trích xuất y hệt và lưu vào thư mục `outputs`.
