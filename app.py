import streamlit as st

st.set_page_config(page_title="Student Segmentation", layout="centered")
st.title("Phân cụm học sinh theo kết quả học tập")

math_score = st.number_input("Điểm Toán (Math Score)", 0.0, 100.0, 75.0)
reading_score = st.number_input("Điểm Đọc hiểu (Reading Score)", 0.0, 100.0, 75.0)
writing_score = st.number_input("Điểm Viết (Writing Score)", 0.0, 100.0, 75.0)

if st.button("Dự đoán nhóm học lực"):
    st.info(f"Đã nhận điểm: Toán={math_score}, Đọc={reading_score}, Viết={writing_score}. Hãy huấn luyện mô hình trước khi chạy dự đoán thực tế.")
