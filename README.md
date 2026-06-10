# 🧠 NeuroScan AI: Clinical Stroke Segmentation System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B)
![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-5C3EE8)

## 📌 Giới thiệu dự án
Đây là hệ thống hỗ trợ ra quyết định lâm sàng (Clinical Decision Support System) ứng dụng học sâu để chẩn đoán và phân vùng tự động tổn thương đột quỵ (nhồi máu và xuất huyết) từ ảnh chụp cắt lớp vi tính (CT) sọ não. 

Dự án sử dụng kiến trúc lai giữa **U-Net++** và bộ mã hóa **EfficientNet-B0**, được triển khai dưới dạng một ứng dụng Web tương tác theo thời gian thực dành cho bác sĩ X-quang.

## 🚀 Tính năng nổi bật
*   **Tiền xử lý tự động (Auto-Crop):** Sử dụng OpenCV để tự động nhận diện, cắt bỏ viền đen và chuẩn hóa khu vực sọ não từ nhiều nguồn ảnh khác nhau.
*   **Phân vùng chính xác (Segmentation):** Khoanh vùng ranh giới các ổ nhồi máu não và xuất huyết não ở cấp độ điểm ảnh.
*   **Bản đồ nhiệt (Explainable AI):** Cung cấp Heatmap xác suất giúp minh bạch hóa tư duy của AI, hiển thị rõ độ tự tin tại từng khu vực tổn thương.
*   **Tương tác lâm sàng động:** Bác sĩ có thể tự do điều chỉnh *Ngưỡng tự tin (Threshold)* với độ phân giải 0.0001 và *Lọc diện tích (Min Area)* để tùy biến hệ thống theo từng thiết bị chụp CT và bối cảnh y khoa.

## 🛠️ Công nghệ sử dụng
*   **Framework Học sâu:** PyTorch, Segmentation Models PyTorch (smp)
*   **Xử lý thị giác máy tính:** OpenCV, Albumentations
*   **Giao diện người dùng (Frontend):** Streamlit
*   **Môi trường huấn luyện:** Kaggle

## 📊 Hiệu suất mô hình (Trên External Test Set)
Hệ thống sử dụng hàm mất mát tổng hợp Hybrid Loss (Dice + BCE) để xử lý tình trạng mất cân bằng dữ liệu, đạt hiệu suất tối ưu trên tập kiểm thử độc lập:
*   **Dice Score:** 0.7705
*   **Precision:** 0.8641
*   **Recall:** 0.8313

## 💻 Hướng dẫn cài đặt và sử dụng
### 1. Clone repository
```bash
git clone https://github.com/quanglam19082004/stroke-segmentation-unet-streamlit.git
cd stroke-segmentation-unet-streamlit
```
### 2. Cài đặt thư viện
```bash
pip install -r requirements.txt
```
### 3. Chạy ứng dụng Web
```bash
streamlit run app.py
```
---
## 👨‍💻 Tác giả
- **Đặng Quang Lâm**
- **Sinh viên năm tư Kỹ thuật Máy tính - Đại học Bách khoa (Đại học Đà Nẵng)**
- **Định hướng:** Computer Networks, Embedded Systems & Applied AI
- **Mọi đóng góp và thắc mắc vui lòng liên hệ qua GitHub Issues**
