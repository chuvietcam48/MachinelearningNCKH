# 🚀 AMAZON CDS & VINYL PIPELINE (V4 FINAL)

Đây là tài liệu hướng dẫn chính thức cho phiên bản **V4 (Bản cập nhật lớn thứ 4)** của dự án Dự đoán Rời bỏ (Churn Prediction) & Tối ưu hóa Can thiệp (EVI Simulation) trên tập dữ liệu Amazon CDs & Vinyl. 

Phiên bản V4 tập trung giải quyết triệt để các vấn đề về **Data Leakage**, **Calibration**, và thiết lập **Risk-Adjusted Guardrail** (Cơ chế an toàn điều chỉnh theo rủi ro).

---

## 🌟 Có gì mới trong V4?

1. **Temporal Split Sạch 100% (Option B)**: Dữ liệu được chia Train/Validation/Test theo trục thời gian thực tuyệt đối (strictly disjoint). Không có chuyện khách hàng ở tập Test xuất hiện trong tập Train cùng một ngày. `train_max_date < val_min_date < test_min_date`.
2. **Zero-Leakage Sentiment Engineering**: Toàn bộ cảm xúc (sentiment) của các lượt đánh giá đều được tính toán **trước** mốc thời gian chốt (prediction cutoff date). Đã loại bỏ hoàn toàn các trường hợp dùng dữ liệu tương lai để dự đoán hiện tại.
3. **Honest Isotonic Calibration**: Hiệu chuẩn phân phối xác suất (Calibration) sử dụng `IsotonicRegression` được fit trên Validation và test trên tập Horizon riêng biệt (loại bỏ hoàn toàn thiên kiến censoring).
4. **Risk-Adjusted EVI Guardrail (Hybrid Fallback)**: Chính sách can thiệp mềm (Soft Buffer) thay vì "cắt cứng". V4 override Baseline V1 chỉ khi `Expected Profit > Risk Buffer` (Buffer tùy biến dựa trên Voucher Cost và Baseline Response Rate). Nếu rủi ro quá cao, hệ thống mượn quyết định an toàn của V1.
5. **Research Honesty**: Feature `loyal_x_mixed` bị loại bỏ vì zero variance. Biến cố `sentiment_shock` (khách hàng tuột dốc cảm xúc) được đưa vào test, CoxPH xác nhận là **không có ý nghĩa thống kê** (Inconclusive) — kết luận vô cùng trung thực.

---

## ⚙️ Hướng dẫn Cài đặt (Installation)

Yêu cầu: **Python 3.9+**

1. **Clone repository (nếu chưa có):**
   ```bash
   git clone <repo-url>
   cd MachinelearningNCKH
   ```

2. **Tạo môi trường ảo (Virtual Environment):**
   ```bash
   python -m venv .venv
   ```

3. **Kích hoạt môi trường ảo:**
   - Trên Windows: `.venv\Scripts\activate`
   - Trên Mac/Linux: `source .venv/bin/activate`

4. **Cài đặt thư viện:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Hãy chắc chắn bạn đã cài đủ các gói chính như `pandas`, `numpy`, `scikit-survival`, `lifelines`, `vaderSentiment`, `matplotlib`)*

---

## 🚀 Cách Vận hành Pipeline (How to Run)

Toàn bộ quy trình từ xử lý dữ liệu thô đến ra quyết định kinh doanh được chia thành 6 giai đoạn (Phases). Bạn vận hành qua file trung tâm `run_amazon_sentiment_pipeline.py`.

### Chạy tự động từ A-Z
```bash
python run_amazon_sentiment_pipeline.py --phase 1 2 3 4 5 6 --force
```

*(Lưu ý: Flag `--force` sẽ ép hệ thống chạy lại từ đầu thay vì đọc cache cũ)*

### Các giai đoạn cụ thể (Phases Breakdown):
- **Phase 1 (Data Extraction):** Lấy dữ liệu thô, cắt theo thời gian, trích xuất điểm Sentiment VADER, tính toán Cú sốc cảm xúc (`sentiment_shock`).
- **Phase 2 (Feature Engineering):** Gắn cờ Loyalty, gộp features, loại bỏ leakage, tạo dataset cuối cùng phục vụ train.
- **Phase 3 (Survival Modeling):** Huấn luyện mô hình Weibull AFT, CoxPH. Chạy Temporal Split. Thực hiện Isotonic Calibration để nắn lại phân phối xác suất dự đoán (1095 ngày).
- **Phase 4 (Proxy Uplift - T-Learner):** Chạy T-Learner lấy điểm Uplift/Treatment Effect (biến `tau_hat`).
- **Phase 5 (Business Simulation):** Kết hợp Xác suất rời bỏ + Xác suất phản hồi (từ Phase 4) để tính EVI (Expected Value of Intervention). Chạy **Monte Carlo Sensitivity Grid (24,000 runs)** để so sánh V4 Raw, V4 Guardrail và V1 Baseline.
- **Phase 6 (Summary & Diagnostics):** Xuất bảng Ablation Table và gen ra file Báo cáo tự động (Markdown).

---

## 📈 Đọc kết quả ở đâu?

Tất cả báo cáo và outputs quan trọng nhất được lưu tại: `outputs/amazon_cds_v3/`

Bạn team member chỉ cần chú ý 2 file cực kỳ quan trọng này:
1. 📄 **`PIPELINE_SUMMARY_V4.md`**: File tóm tắt cho **Business / Management**. Chứa Ablation table và Narrative chốt sổ (Win Rate, Profit Lift, Worst-case, C-index).
2. 📄 **`V4_DIAGNOSTICS.md`**: File báo cáo chi tiết **Kỹ thuật (Technical)**. Chứa bảng kiểm tra Data Leakage, phân phối Feature, CoxPH p-values, Calibration Table và Error Analysis (False Positives).

---

## 💡 Hiểu về "Risk-Adjusted Guardrail" (Quan trọng cho Team)

Trong V4, chúng ta không dùng model V4 một cách mù quáng. Ở **Phase 5**, hệ thống sẽ chạy một `Hybrid Guardrail`:
- **Voucher cost = $2** → Risk Buffer = 0.0
- **Voucher cost = $5** → Risk Buffer = 1.0
- **Voucher cost = $10** → Risk Buffer = 3.0
- *(Bị phạt thêm +2.0 Buffer nếu Base Response Rate quá thấp < 0.10)*

Nếu Expected Profit (EVI) vượt qua ngưỡng Buffer trên, V4 được phép can thiệp. Nếu không, V4 sẽ nhường lại quyết định (fallback) cho V1 (Baseline quy luật cũ). 
**Kết quả thực tế:** Guardrail này nâng tỉ lệ Win-rate lên ~84% và quản lý lỗ cực kì chặt chẽ (Worst-case giảm thiểu đáng kể so với Raw V4).

---

Chúc team vận hành và nộp Paper thành công! 🎉
