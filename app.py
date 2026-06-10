import streamlit as st
import torch
import cv2
import numpy as np
import pydicom
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage.measure import label, regionprops

# ==========================================
# CẤU HÌNH GIAO DIỆN STREAMLIT
# ==========================================
st.set_page_config(page_title="Hệ thống Chẩn đoán Đột quỵ", layout="wide")
st.title("Phần Mềm Hỗ Trợ Chẩn Đoán Phân Vùng Đột Quỵ Y Khoa")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 256

# ==========================================
# 1. KHỞI TẠO MÔ HÌNH (CACHE)
# ==========================================
# st.cache_resource giúp model chỉ load 1 lần duy nhất vào RAM
@st.cache_resource
def load_model():
    import segmentation_models_pytorch as smp
    
    # 1. Khởi tạo mô hình gốc
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b0", 
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    
    checkpoint_path = 'best_unetplusplus_stroke.pth' 
    
    try:
        # 2. Đọc file trọng số
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        
        # 3. Trích xuất đúng state_dict
        if 'model_state' in checkpoint:
            state_dict = checkpoint['model_state']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        # 4. CHỮA CHÁY: Tự động gọt bỏ tiền tố "unet." (nếu có) do custom class tạo ra
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('unet.'):
                cleaned_key = key[5:]  # Cắt bỏ 5 ký tự "unet."
            else:
                cleaned_key = key
            cleaned_state_dict[cleaned_key] = value
            
        # 5. Nạp trọng số đã làm sạch vào mô hình
        model.load_state_dict(cleaned_state_dict)
        print("✅ Load model thành công!")
        
    except Exception as e:
        st.error(f"Lỗi load model: {e}")
        
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# ==========================================
# 2. CÁC HÀM TIỀN XỬ LÝ & HẬU XỬ LÝ
# ==========================================
def process_dicom(file_upload):
    """Hàm đọc file DICOM cơ bản"""
    dicom = pydicom.dcmread(file_upload)
    img = dicom.pixel_array
    # Đưa về dải 0-255 uint8
    img = img - np.min(img)
    img = (img / np.max(img) * 255).astype(np.uint8)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img

def crop_brain_contour(image):
    """Hàm tự động cắt sọ não mà chúng ta đã tối ưu"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
        
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    cropped_img = image[y:y+h, x:x+w]
    return cv2.resize(cropped_img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)

def clean_mask(mask, min_area=100):
    """Hậu xử lý: Lọc nhiễu báo động giả"""
    mask = mask.astype(np.uint8)
    labeled_mask = label(mask)
    for region in regionprops(labeled_mask):
        if region.area < min_area:
            mask[labeled_mask == region.label] = 0
    return mask

# Transform Pipeline chuẩn ImageNet
val_transforms = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# ==========================================
# 3. GIAO DIỆN TẢI ẢNH & ĐIỀU HƯỚNG
# ==========================================
st.sidebar.header("Tải dữ liệu bệnh nhân")
uploaded_file = st.sidebar.file_uploader("Chọn ảnh CT (PNG, JPG, DCM)", type=['png', 'jpg', 'jpeg', 'dcm'])
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Thiết lập thông số lâm sàng")

# --- 1. THIẾT LẬP NGƯỠNG (THRESHOLD) ---
col_s1, col_n1 = st.sidebar.columns([0.65, 0.35])

# Thanh kéo trượt (bước nhảy siêu nhỏ 0.0001)
slider_thresh = col_s1.slider("Ngưỡng tự tin", 0.0000, 0.1000, 0.0010, 0.0001, format="%.4f")

# Ô nhập số có nút +/- (nhận giá trị mặc định từ thanh kéo)
THRESHOLD = col_n1.number_input("Nhập số", min_value=0.0000, max_value=0.1000, value=slider_thresh, step=0.0001, format="%.4f")


# --- 2. THIẾT LẬP LỌC NHIỄU (MIN AREA) ---
col_s2, col_n2 = st.sidebar.columns([0.65, 0.35])

# Thanh kéo trượt 
slider_area = col_s2.slider("Lọc diện tích", 0, 1000, 50, 10)

# Ô nhập số có nút +/-
MIN_AREA = col_n2.number_input("Nhập số", min_value=0, max_value=2000, value=slider_area, step=10)
if uploaded_file is not None:
    # --- ĐỌC ẢNH ---
    if uploaded_file.name.endswith('.dcm'):
        raw_img = process_dicom(uploaded_file)
    else:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        raw_img = cv2.imdecode(file_bytes, 1)
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    raw_img = cv2.resize(raw_img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    # --- TIỀN XỬ LÝ ---
    cropped_img = crop_brain_contour(raw_img)
    input_tensor = val_transforms(image=cropped_img)["image"].unsqueeze(0).to(DEVICE)

# # --- INFERENCE ---
#     with st.spinner('Mô hình đang phân tích...'):
#         with torch.no_grad():
#             seg_logits = model(input_tensor)
#             probs = torch.sigmoid(seg_logits)
            
#             # Lấy ma trận xác suất gốc (chưa qua ngưỡng)
#             probs_np = probs.cpu().numpy()[0, 0] 
# # --- CÔNG THỨC MỚI: Tự động chuẩn hóa ngưỡng ---
#             # Chỉ lấy những phần có xác suất cao hơn 50% so với giá trị cao nhất của ảnh đó
#             adaptive_threshold = np.max(probs_np) * THRESHOLD 
#             preds = (probs_np > adaptive_threshold).astype(np.float32)            
#             # Áp dụng ngưỡng từ thanh trượt
#             preds = (probs_np > THRESHOLD).astype(np.float32) 
# --- INFERENCE ---
    with st.spinner('Mô hình đang phân tích...'):
        with torch.no_grad():
            seg_logits = model(input_tensor)
            probs = torch.sigmoid(seg_logits)
            probs_np = probs.cpu().numpy()[0, 0] 
            
            # QUAY VỀ LOGIC CƠ BẢN: Ngưỡng tuyệt đối
            # Đơn giản là lấy những pixel có xác suất cao hơn ngưỡng trên thanh trượt
            preds = (probs_np > THRESHOLD).astype(np.float32)
            
    # --- HẬU XỬ LÝ (Tăng Min Area để chặn đỏ toàn ảnh) ---
    # Min Area càng cao, nó càng lọc bỏ những vùng đỏ lấm tấm
    final_mask = clean_mask(preds, min_area=int(MIN_AREA))  
    # --- HẬU XỬ LÝ (Lọc nhiễu) ---
    final_mask = clean_mask(preds, min_area=MIN_AREA)

    # --- TÍNH TOÁN LÂM SÀNG ---
    gray_cropped = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2GRAY)
    brain_pixels = np.sum(gray_cropped > 10) 
    lesion_pixels = np.sum(final_mask == 1)
    
    lesion_percentage = (lesion_pixels / brain_pixels) * 100 if brain_pixels > 0 else 0.0

# ==========================================
    # 4. HIỂN THỊ DASHBOARD
    # ==========================================
    st.subheader("Báo cáo Chẩn đoán Hình ảnh")
    
    max_prob = np.max(probs_np) # Tìm điểm tự tin nhất của mô hình
    
    # Chia 4 cột thông số
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Trạng thái", "Phát hiện Đột quỵ" if lesion_pixels > 0 else "Bình thường", 
                delta="Cảnh báo" if lesion_pixels > 0 else "An toàn", delta_color="inverse")
    col2.metric("Diện tích tổn thương", f"{lesion_pixels} pixels")
    col3.metric("Tỷ lệ tổn thương (Lesion %)", f"{lesion_percentage:.2f} %")
    
    # THÊM CỘT 4: Theo dõi nội bộ
    col4.metric("Độ tự tin cao nhất (Max Prob)", f"{max_prob*100:.2f} %")

    st.markdown("---")

    # Hiển thị 4 cột ảnh
    img_col1, img_col2, img_col3, img_col4 = st.columns(4)
    
    with img_col1:
        st.image(raw_img, caption="1. Ảnh CT Gốc", use_container_width=True)
        
    with img_col2:
        st.image(cropped_img, caption="2. Tiền xử lý (Auto-Crop)", use_container_width=True)
        
    with img_col3:
        final_mask_fixed = cv2.flip(final_mask, 1)        
        overlay_img = cropped_img.copy()
        overlay_img[final_mask == 1] = [255, 0, 0] 
        st.image(overlay_img, caption="3. Kết quả U-Net++ (Overlay)", use_container_width=True)

    with img_col4:
        # SUPER HEATMAP: Chuẩn hóa để luôn nhìn thấy điểm nóng nhất
        probs_norm = (probs_np - np.min(probs_np)) / (max_prob - np.min(probs_np) + 1e-8)
        heatmap = cv2.applyColorMap(np.uint8(255 * probs_norm), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        st.image(heatmap, caption="4. Bản đồ Nhiệt (Siêu nhạy)", use_container_width=True)