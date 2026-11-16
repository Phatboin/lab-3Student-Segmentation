## LAB03 – STUDENT SEGMENTATION

## 1. Giới thiệu

Lab03 thực hiện bài toán **Student Segmentation (Phân cụm học sinh theo kết quả học tập)** bằng thuật toán K-means tự cài đặt với NumPy.

Mục tiêu là phân nhóm học sinh dựa trên các đặc trưng về điểm số các môn học (Toán, Đọc hiểu, Viết). Qua đó, giáo viên và nhà trường có thể phân tích năng lực học tập, phát hiện nhóm học sinh thiên lệch môn học hoặc có nguy cơ hổng kiến thức để đưa ra lộ trình bồi dưỡng phù hợp.

### Đặc trưng sử dụng

| Đặc trưng | Vai trò |
|---|---|
| `Math_Score` | Điểm môn Toán (thang 0 – 100) |
| `Reading_Score` | Điểm kỹ năng Đọc hiểu (thang 0 – 100) |
| `Writing_Score` | Điểm kỹ năng Viết (thang 0 – 100) |
| `Student_ID` | Chỉ dùng để định danh học sinh, không đưa vào mô hình |
| `Gender` | Giới tính – dùng để mô tả và phân tích đặc điểm cụm |
| `Test_Prep_Course` | Trạng thái hoàn thành khóa ôn tập – dùng để phân tích đặc điểm cụm |

Mô hình cơ sở sử dụng ba đặc trưng số là `Math_Score`, `Reading_Score` và `Writing_Score`. Các thuộc tính phân loại như `Gender` và `Test_Prep_Course` không tham gia tính khoảng cách trong mô hình cơ sở.

---

## 2. Cấu trúc dự án

```text
Lab03_ML/
├── agents/                         # Mô tả bài toán, quy tắc và hướng dẫn
├── app.py                          # Streamlit inference dự đoán nhóm học lực học sinh mới
├── config.yaml                     # Cấu hình dữ liệu điểm số và tham số K-means
├── data/
│   ├── raw/                        # Dữ liệu gốc (students_performance.csv)
│   └── processed/                  # Dữ liệu điểm đã làm sạch, chuẩn hóa và gán nhãn cụm
├── docs/
│   ├── images/                     # Hình ảnh trực quan hóa
│   ├── experiment_logs.csv         # Lịch sử thực nghiệm theo Run_ID
│   └── results_summary.md          # Tóm tắt kết quả phân cụm học sinh
├── logs/                           # Nhật ký thực thi Pipeline
├── models/                         # Mô hình K-means (kmeans.pkl) và Scaler (scaler.pkl) đã huấn luyện
├── notebooks/                      # Notebook thực nghiệm
├── progress_results/               # Kết quả tracking theo Run_ID
├── reports/
│   ├── figures/
│   │   ├── student_segmentation/
│   │   │   └── <Run_ID>/
│   │   │       ├── 01_eda/
│   │   │       ├── 02_preprocessing/
│   │   │       ├── 03_training/
│   │   │       ├── 04_evaluation/
│   │   │       └── 05_extended/
│   │   │           ├── convergence/
│   │   │           └── experiments/
│   │   └── additional_grade_heatmaps/
│   │       └── <Run_ID>/
│   │           ├── 01_preprocessing/
│   │           ├── 02_elbow/
│   │           ├── 03_segmentation/
│   │           └── 04_profile_cards/
│   ├── archive/                    # Báo cáo cũ
│   ├── inference_history.csv       # Lịch sử dự đoán học sinh mới
│   ├── metrics_summary.csv         # Tổng hợp chỉ số đánh giá
│   └── report_<Run_ID>.{md,html,pdf}
├── src/
│   ├── models/
│   │   └── kmeans.py               # K-means tự cài đặt bằng NumPy
│   ├── eda.py                      # Khám phá phân phối điểm và tương quan môn học
│   ├── preprocess.py               # Làm sạch dữ liệu thiếu/ngoại lai và chuẩn hóa
│   ├── train.py                    # Huấn luyện và lựa chọn số cụm học lực K
│   ├── evaluate.py                 # Đánh giá chỉ số và diễn giải đặc điểm nhóm học sinh
│   ├── experiments.py              # So sánh không gian đặc trưng (2D/3D/4D) và Grid Search
│   ├── demo_convergence.py         # Minh họa quá trình hội tụ tâm cụm điểm số
│   ├── predict.py                  # Dự đoán nhóm học lực cho học sinh mới
│   ├── tracking.py                 # Quản lý Run_ID và tracking
│   ├── figure_manager.py           # Thu thập và phân loại hình ảnh
│   ├── report.py                   # Sinh báo cáo MD, HTML, PDF
│   └── pipeline.py                 # Điều phối toàn bộ quy trình
├── tests/                          # Unit tests cho K-means và tiền xử lý
├── requirements.txt
└── README.md
```

---

## 3. Cài đặt môi trường

Yêu cầu Python ($\ge 3.9$) và các thư viện được khai báo trong `requirements.txt` (`numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `pyyaml`, `streamlit`).

Mở PowerShell tại thư mục gốc `Lab03_ML` và thực hiện:

```powershell
# Tạo môi trường ảo
python -m venv .venv

# Kích hoạt môi trường ảo
.\.venv\Scripts\Activate.ps1

# Cài đặt thư viện
pip install -r requirements.txt
```

---

## 4. Chạy toàn bộ Pipeline

Đứng tại thư mục gốc của dự án.

### 4.1. Pipeline cơ bản

```powershell
python -m src.pipeline
```

Pipeline cơ bản thực hiện các giai đoạn:

1. **EDA:** Khám phá dữ liệu điểm số, phân tích phân phối điểm từng môn, phát hiện điểm ngoại lai (ví dụ điểm 0 do bỏ thi) và mối tương quan giữa Toán, Đọc, Viết.
2. **Preprocessing:** Kiểm tra dữ liệu thiếu, loại bỏ `Student_ID`, tách biến phân loại và chuẩn hóa các cột điểm bằng `StandardScaler`.
3. **Training:** Thử nghiệm các giá trị $K \in [2, 8]$ và lựa chọn mô hình K-means tối ưu (thường $K=4$ nhóm học lực).
4. **Evaluation:** Đánh giá chất lượng phân cụm qua các chỉ số nội tại và phân tích hồ sơ năng lực (Radar chart, trung bình điểm, tỷ lệ ôn tập) của từng cụm.
5. **Visualization:** Thu thập, phân loại và lưu hình ảnh theo `Run_ID`.
6. **Report:** Tổng hợp kết quả thành báo cáo Markdown, HTML và PDF.

Mỗi lần chạy tạo một `Run_ID` riêng để phân biệt và theo dõi kết quả thực nghiệm.

### 4.2. Pipeline mở rộng

```powershell
python -m src.pipeline --extended
```

Ngoài các giai đoạn cơ bản, chế độ mở rộng thực hiện thêm:

- **So sánh các không gian đặc trưng (Feature Space Comparison):** So sánh kết quả phân cụm giữa không gian 2D (`Math_Score`, `Reading_Score`), 3D (`Math_Score`, `Reading_Score`, `Writing_Score`) và 4D (bổ sung biến mã hóa `Test_Prep_Course` hoặc tỷ lệ chuyên cần).
- **Grid Search:** Khảo sát các cấu hình K-means như phương pháp khởi tạo tâm cụm (`random` vs `k-means++`), `max_iters` và `tol`.
- **Minh họa lịch sử hội tụ (K-means Convergence):** Trực quan hóa sự dịch chuyển của các tâm cụm điểm số qua từng vòng lặp.

---

## 5. Chạy riêng từng giai đoạn

Có thể chạy từng module độc lập từ thư mục gốc:

```powershell
# Khám phá dữ liệu điểm học sinh
python -m src.eda

# Tiền xử lý và chuẩn hóa điểm số
python -m src.preprocess

# Huấn luyện mô hình K-means
python -m src.train

# Đánh giá và phân tích nhóm học lực
python -m src.evaluate

# Thực nghiệm mở rộng (Feature Space & Grid Search)
python -m src.experiments

# Minh họa quá trình hội tụ
python -m src.demo_convergence
```

**Lưu ý:** Khi chạy từng module riêng lẻ, cần đảm bảo dữ liệu đầu vào và các kết quả của giai đoạn trước đã tồn tại trong `data/processed/` hoặc `models/`. Việc chạy riêng không tự động sinh đầy đủ tracking và báo cáo tổng hợp như khi chạy toàn bộ Pipeline.

---

## 6. Tracking và quản lý kết quả

Dự án sử dụng `Run_ID` để theo dõi từng lần thực nghiệm. Các kết quả được lưu riêng nhằm hỗ trợ kiểm tra, so sánh và tái lập quá trình chạy.

| Thành phần | Đường dẫn | Chức năng |
|---|---|---|
| Progress | `progress_results/progress_<Run_ID>.json` | Trạng thái, đầu vào, đầu ra, thời gian và lỗi của từng giai đoạn |
| Log | `logs/pipeline_<Run_ID>.log` | Nhật ký thực thi Pipeline theo thời gian |
| Experiment Logs | `docs/experiment_logs.csv` | Lịch sử thử nghiệm $K$ và thông tin `Run_ID` |
| Metrics Summary | `reports/metrics_summary.csv` | Tổng hợp các chỉ số đánh giá theo `Run_ID` |
| Figures | `reports/figures/` | Hình ảnh được phân loại theo Pipeline và `Run_ID` |
| Reports | `reports/report_<Run_ID>` | Báo cáo tổng hợp của từng lần chạy |

### 6.1. Tổ chức hình ảnh

Hình ảnh được thu thập và phân loại tự động bằng `src/figure_manager.py`.

Đối với Student Segmentation:

```text
student_segmentation/
└── <Run_ID>/
    ├── 01_eda/
    ├── 02_preprocessing/
    ├── 03_training/
    ├── 04_evaluation/
    └── 05_extended/
        ├── convergence/
        └── experiments/
```

Ý nghĩa các thư mục:

| Thư mục | Nội dung |
|---|---|
| `01_eda` | Histogram phân phối điểm Toán/Đọc/Viết, Boxplot theo giới tính & khóa ôn tập, Heatmap tương quan |
| `02_preprocessing` | Phân phối dữ liệu trước và sau chuẩn hóa Z-score (`StandardScaler`) |
| `03_training` | Biểu đồ Elbow Method (Inertia) và Silhouette Score theo từng giá trị $K$ |
| `04_evaluation` | Biểu đồ phân tán 3D cụm học sinh, Radar chart năng lực từng nhóm, thống kê nhân khẩu học cụm |
| `05_extended/convergence` | Quá trình dịch chuyển tâm cụm qua từng iteration của K-means |
| `05_extended/experiments` | Biểu đồ so sánh các không gian đặc trưng (2D vs 3D vs 4D) và kết quả Grid Search |

### 6.2. Tạo lại báo cáo

Có thể tạo lại báo cáo cho một `Run_ID` đã tồn tại bằng lệnh:

```powershell
python -m src.report --run-id <RUN_ID>
```

Thay `<RUN_ID>` bằng mã thực nghiệm cần tạo báo cáo. Các định dạng xuất ra bao gồm:
- `.md`: Báo cáo Markdown.
- `.html`: Báo cáo HTML trực quan.
- `.pdf`: Báo cáo PDF phục vụ lưu trữ và in ấn.

---

## 7. Inference – Dự đoán nhóm học lực cho học sinh mới

Sau khi hoàn tất huấn luyện, giáo viên hoặc cố vấn học tập có thể sử dụng mô hình đã lưu để phân nhóm cho học sinh mới.

### 7.1. Chạy inference bằng dòng lệnh

```powershell
python -m src.predict
```

### 7.2. Chạy giao diện Streamlit

```powershell
python -m streamlit run app.py
```

### 7.3. Quy trình dự đoán

Inference nhận ba thông tin điểm số đầu vào của học sinh:
- `Math_Score`
- `Reading_Score`
- `Writing_Score`

Quy trình xử lý:

1. Nhận thông tin điểm số của học sinh mới (từ CLI hoặc giao diện web Streamlit).
2. Đọc đối tượng `scaler.pkl` đã được lưu từ giai đoạn Preprocessing.
3. Chuẩn hóa bộ điểm mới bằng đúng trung bình ($\mu$) và độ lệch chuẩn ($\sigma$) đã học từ tập huấn luyện.
4. Đưa vector điểm đã chuẩn hóa vào `model.predict()`.
5. Xác định cụm học lực tương ứng (ví dụ: *Xuất sắc toàn diện*, *Thiên về Tự nhiên*, *Thiên về Xã hội*, *Cần hỗ trợ*).
6. Hiển thị kết quả, khuyến nghị học tập và lưu lịch sử dự đoán.

Lịch sử inference được lưu tại:

```text
reports/inference_history.csv
```

**Lưu ý:** Tuyệt đối không được `fit` lại scaler trên dữ liệu học sinh mới. Phải sử dụng các tham số chuẩn hóa được lưu từ tập huấn luyện để đảm bảo điểm số mới nằm trong cùng không gian tọa độ với dữ liệu huấn luyện.

---

## 8. Kiểm thử

Chạy bộ unit tests bằng lệnh:

```powershell
python -m unittest discover -s tests -v
```

Các kịch bản kiểm thử bao gồm:
- Kiểm tra tính đúng đắn của phép tính khoảng cách Euclidean và cập nhật tâm cụm trong `src/models/kmeans.py`.
- Kiểm tra điều kiện dừng hội tụ của thuật toán.
- Kiểm tra tính nhất quán của `StandardScaler` và định dạng đầu ra của hàm `predict()`.

---

## 9. Phương pháp và lưu ý đánh giá

### 9.1. Thuật toán K-means

K-means là thuật toán học không giám sát (Unsupervised Learning). Thuật toán phân chia tập học sinh thành $K$ nhóm dựa trên khoảng cách Euclidean giữa vector điểm số của từng học sinh và tâm cụm (Centroid).

Quy trình cơ bản:

1. Khởi tạo $K$ tâm cụm ban đầu (ngẫu nhiên hoặc theo chiến lược `k-means++`).
2. Tính khoảng cách từ từng học sinh đến các tâm cụm.
3. Gán mỗi học sinh vào cụm có tâm gần nhất.
4. Cập nhật lại tâm cụm bằng trung bình cộng điểm số của tất cả học sinh thuộc cụm đó.
5. Lặp lại bước 3 và 4 cho đến khi tâm cụm không thay đổi (hội tụ) hoặc đạt số vòng lặp tối đa (`max_iters`).

Trong dự án này, thuật toán K-means được tự cài đặt hoàn toàn bằng thư viện NumPy.

### 9.2. Các chỉ số đánh giá

| Chỉ số | Ý nghĩa trong bài toán phân cụm học sinh | Hướng đánh giá |
|---|---|---|
| **Inertia (WCSS)** | Tổng bình phương khoảng cách điểm số từ học sinh đến tâm cụm của mình | Thấp hơn thường tốt hơn |
| **Silhouette Score** | Đo lường mức độ đồng đều học lực trong cùng một nhóm và sự tách biệt giữa các nhóm | Cao hơn (gần $1$) tốt hơn |
| **Davies-Bouldin Index** | Đánh giá tỷ lệ phân tán nội cụm so với khoảng cách giữa các tâm nhóm học lực | Thấp hơn (gần $0$) tốt hơn |
| **Calinski-Harabasz Index** | Tỷ số phương sai giữa các nhóm học lực và phương sai nội bộ từng nhóm | Cao hơn thường tốt hơn |

Không nên lựa chọn $K$ chỉ dựa trên một chỉ số toán học duy nhất. Cần kết hợp các chỉ số đánh giá với quy mô từng cụm và tính ứng dụng thực tế trong sư phạm (ví dụ: chia thành 4 nhóm năng lực rõ ràng để dễ xếp lớp phụ đạo).

### 9.3. Những điểm cần lưu ý

- K-means là học không giám sát nên không có nhãn "đúng/sai" cố định để tính Accuracy như bài toán phân loại.
- Inertia luôn giảm đơn điệu khi tăng $K$, vì vậy cần tìm điểm khuỷu tay (Elbow) kết hợp với Silhouette Score thay vì chọn $K$ lớn nhất.
- `Student_ID` không được đưa vào mô hình vì chỉ mang ý nghĩa định danh.
- `Gender` và `Test_Prep_Course` chỉ dùng để hậu kiểm (post-hoc analysis) nhằm giải thích nguyên nhân chênh lệch điểm số giữa các cụm (ví dụ: cụm điểm cao có tỷ lệ hoàn thành khóa ôn tập cao hơn hẳn).
- Các thực nghiệm mở rộng đưa biến phân loại (đã mã hóa One-Hot/Binary) vào tính khoảng cách chỉ mang tính khảo sát, vì biến nhị phân $0/1$ có thể làm sai lệch cấu trúc khoảng cách Euclidean liên tục của điểm số.
- Nhãn số nguyên của cụm (`Cluster 0`, `Cluster 1`...) chỉ là định danh do thuật toán sinh ra ngẫu nhiên theo lần khởi tạo, không tự mang thứ bậc cao/thấp nếu chưa đối chiếu với giá trị điểm trung bình của tâm cụm.

---

## 10. Kết luận

Lab03 xây dựng một quy trình phân cụm học sinh theo kết quả học tập hoàn chỉnh từ đầu đến cuối: khám phá dữ liệu điểm số, tiền xử lý, tự cài đặt thuật toán K-means bằng NumPy, huấn luyện chọn $K$, đánh giá đa chỉ số, trực quan hóa và triển khai ứng dụng dự đoán cho học sinh mới.

Bên cạnh mô hình cơ sở trên 3 môn học chính, dự án còn tích hợp các thực nghiệm mở rộng nhằm khảo sát tác động của không gian đặc trưng và quá trình hội tụ của thuật toán.

Hệ thống tracking theo `Run_ID`, quản lý hình ảnh phân cấp và tự động xuất báo cáo đa định dạng giúp toàn bộ quá trình nghiên cứu và phân tích dữ liệu giáo dục được lưu trữ khoa học, minh bạch và dễ dàng tái lập.
