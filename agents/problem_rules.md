# Quy tắc bài toán Student Segmentation
- Sử dụng K-means tự cài đặt bằng NumPy (`src/models/kmeans.py`).
- Không đưa `Student_ID` vào tính khoảng cách.
- Không fit lại `StandardScaler` ở bước inference (`src/predict.py`, `app.py`).
