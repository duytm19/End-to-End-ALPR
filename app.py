# app.py

import streamlit as st
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import time
from pathlib import Path
import tempfile # Thêm thư viện để xử lý file tạm

# Import các module từ thư mục src
from src import config
from src.vehicle_detector import VehicleDetector
from src.lp_detector import LicensePlateDetector
from src import preprocessor
from src.ocr_reader import OcrReader

# --- Cấu hình trang Streamlit ---
st.set_page_config(
    page_title="Hệ thống Nhận dạng Biển số xe",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🚗 Giao diện Nhận dạng Biển số xe (ALPR)")

# --- Tải và Cache Model ---
# Sử dụng @st.cache_resource để đảm bảo model chỉ được tải một lần
@st.cache_resource
def load_models():
    """Tải tất cả các model và cache chúng."""
    with st.spinner('Đang tải các model AI, vui lòng chờ...'):
        vehicle_detector = VehicleDetector(config.VEHICLE_DETECTION_MODEL_PATH)
        lp_detector = LicensePlateDetector(config.LP_DETECTION_MODEL_PATH)
        denoising_model = tf.keras.models.load_model(config.DENOISING_MODEL_PATH, compile=False)
        ocr = OcrReader(config.PADDLEOCR_MODEL_DIR, config.CHAR_DICT_PATH)
    
    # Kiểm tra xem các model đã được tải thành công chưa
    if not all([vehicle_detector.model, lp_detector.model, denoising_model, ocr.ocr_reader]):
        st.error("Lỗi nghiêm trọng: Một hoặc nhiều model không thể tải. Vui lòng kiểm tra đường dẫn trong file config.py.")
        return None
    return vehicle_detector, lp_detector, denoising_model, ocr

# Tải model
models = load_models()
if models:
    vehicle_detector, lp_detector, denoising_model, ocr = models
    st.sidebar.success("Tải model thành công!")
else:
    st.stop()


# THAY THẾ TOÀN BỘ HÀM CŨ BẰNG HÀM MỚI NÀY
def process_image(frame, image_name, verbose=False):
    """
    Hàm xử lý ảnh/khung hình.
    - verbose=True: In ra đầy đủ log, dùng cho ảnh đơn.
    - verbose=False: Chạy trong im lặng, dùng cho video.
    """
    output_image = frame.copy()
    csv_results = []

    if verbose:
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(config.PROCESSED_LP_DIR, exist_ok=True)
        os.makedirs(config.SUCCESSFUL_VEHICLES_DIR, exist_ok=True)

    # Dùng st.status chỉ ở chế độ verbose
    status_context = st.status(f"🔍 Đang phát hiện xe trong ảnh {image_name}...", expanded=True) if verbose else None

    vehicle_boxes = vehicle_detector.detect(frame, config.VEHICLE_CONF_THRESHOLD)
    
    if not np.any(vehicle_boxes):
        if verbose and status_context:
            st.warning("Không phát hiện được xe nào.")
            status_context.update(label="Không tìm thấy xe.", state="complete")
        return output_image, []

    if verbose and status_context:
        st.write(f"✅ Tìm thấy **{len(vehicle_boxes)}** xe.")
        status_context.update(label="Phát hiện xe hoàn tất!", state="running")

    for i, (x1, y1, x2, y2) in enumerate(vehicle_boxes):
        vehicle_id = f"{Path(image_name).stem}_{i}"
        if verbose:
            st.write(f"--- Đang xử lý **Xe ID: {vehicle_id}** ---")
        
        vehicle_crop = frame[int(y1):int(y2), int(x1):int(x2)]
        lp_box = lp_detector.detect(vehicle_crop, config.LP_CONF_THRESHOLD)
        if lp_box is None:
            if verbose: st.write(f"  - ⚠️ Không tìm thấy biển số trên Xe ID: {vehicle_id}.")
            continue
        
        if verbose: st.write(f"  - ✅ Tìm thấy biển số. Bắt đầu tiền xử lý...")
        lp_x1, lp_y1, lp_x2, lp_y2 = lp_box
        lp_crop = vehicle_crop[int(lp_y1):int(lp_y2), int(lp_x1):int(lp_x2)]

        images_for_ocr = preprocessor.process_lp_for_ocr(lp_crop, denoising_model)
        if not images_for_ocr:
            if verbose: st.write("  - ⚠️ Tiền xử lý không tạo ra ảnh để nhận diện.")
            continue
        
        full_text, avg_confidence = ocr.recognize(images_for_ocr, vehicle_id)

        if full_text:
            if verbose: st.write(f"  - ✅ **Kết quả:** `{full_text}` (Độ tin cậy: {avg_confidence:.2f})")
            
            csv_results.append({
                'frame': image_name, 'id_vehicle': vehicle_id,
                'full_text': full_text, 'avg_confidence': f"{avg_confidence:.4f}"
            })
            
            label = f"ID:{i} LP:{full_text}"
            cv2.rectangle(output_image, (int(x1), int(y1)), (int(x2), int(y2)), (36, 255, 12), 2)
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(output_image, (int(x1), int(y1) - h - 10), (int(x1) + w, int(y1)), (36, 255, 12), -1)
            cv2.putText(output_image, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            if verbose:
                try:
                    safe_lp_text = "".join(c for c in full_text if c.isalnum())
                    vehicle_filename = f"{Path(image_name).stem}_id{i}_{safe_lp_text}.jpg"
                    save_path = os.path.join(config.SUCCESSFUL_VEHICLES_DIR, vehicle_filename)
                    cv2.imwrite(save_path, vehicle_crop)
                    st.write(f"  - 💾 Đã lưu ảnh xe thành công vào: `{save_path}`")
                except Exception as e:
                    st.warning(f"  - ⚠️ Đã có lỗi khi lưu ảnh xe: {e}")
        else:
            if verbose: st.write(f"  - ⚠️ OCR thất bại cho Xe ID: {vehicle_id}.")
    
    if verbose and status_context:
        status_context.update(label="Hoàn tất xử lý!", state="complete")
        
    return output_image, csv_results


# --- GIAO DIỆN CHÍNH ---

st.sidebar.header("Chọn chế độ")
app_mode = st.sidebar.selectbox("Vui lòng chọn chế độ xử lý:",
                                ["Chọn chế độ...", "ALPR 1 ảnh", "ALPR 1 thư mục",  "ALPR từ Video"])

# --- CHẾ ĐỘ 1: XỬ LÝ 1 ẢNH ---
if app_mode == "ALPR 1 ảnh":
    st.header("Chế độ 1: Xử lý một ảnh duy nhất")
    uploaded_file = st.file_uploader("Tải lên một ảnh (JPG, PNG)...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Đọc ảnh từ file tải lên
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)

        col1, col2 = st.columns(2)
        with col1:
            st.image(frame, channels="BGR", caption="Ảnh gốc")

        if st.button("🚀 Bắt đầu xử lý"):
            result_image, results_list = process_image(frame, uploaded_file.name)
            
            with col2:
                st.image(result_image, channels="BGR", caption="Ảnh kết quả")
            
            if results_list:
                st.subheader("Kết quả nhận diện:")
                df_results = pd.DataFrame(results_list)
                st.dataframe(df_results)

                # Tạo nút download CSV
                @st.cache_data
                def convert_df_to_csv(df):
                    return df.to_csv(index=False).encode('utf-8')
                
                csv_data = convert_df_to_csv(df_results)
                st.download_button(
                    label="📥 Tải kết quả (CSV)",
                    data=csv_data,
                    file_name=f"result_{Path(uploaded_file.name).stem}.csv",
                    mime='text/csv',
                )
            else:
                st.info("Không nhận diện được biển số nào trong ảnh.")


# --- CHẾ ĐỘ 2: XỬ LÝ THƯ MỤC ---
elif app_mode == "ALPR 1 thư mục":
    st.header("Chế độ 2: Xử lý toàn bộ ảnh trong một thư mục")
    folder_path = st.text_input("Nhập đường dẫn đến thư mục chứa ảnh:")

    if st.button("📁 Bắt đầu xử lý thư mục"):
        if folder_path and os.path.isdir(folder_path):
            image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if not image_files:
                st.error("Không tìm thấy file ảnh nào trong thư mục đã chọn.")
            else:
                st.info(f"Tìm thấy **{len(image_files)}** ảnh. Bắt đầu xử lý...")
                
                progress_bar = st.progress(0)
                all_results = []

                
                # Nơi hiển thị ảnh
                image_placeholder = st.empty()
                
                for i, image_name in enumerate(image_files):
                    image_path = os.path.join(folder_path, image_name)
                    frame = cv2.imread(image_path)
                    
                    st.subheader(f"Đang xử lý: {image_name}")
                    result_image, results_list = process_image(frame, image_name)
                    
                    # Hiển thị ảnh kết quả
                    image_placeholder.image(result_image, channels="BGR", caption=f"Kết quả cho: {image_name}", use_column_width=True)
                    
                    if results_list:
                        all_results.extend(results_list)

                    # Cập nhật progress bar
                    progress_bar.progress((i + 1) / len(image_files))
                    time.sleep(0.1) # Tạm dừng để UI mượt hơn
                
                st.success("🎉 Hoàn tất xử lý toàn bộ thư mục!")
                if all_results:
                    st.subheader("Tổng hợp kết quả:")
                    df_all_results = pd.DataFrame(all_results)
                    st.dataframe(df_all_results)

                    @st.cache_data
                    def convert_df_to_csv(df):
                        return df.to_csv(index=False).encode('utf-8')
                    
                    csv_data = convert_df_to_csv(df_all_results)
                    st.download_button(
                        label="📥 Tải tất cả kết quả (CSV)",
                        data=csv_data,
                        file_name="results_full_folder.csv",
                        mime='text/csv',
                    )
                else:
                    st.info("Không nhận diện được biển số nào trong toàn bộ thư mục.")

        else:
            st.error("Đường dẫn thư mục không hợp lệ. Vui lòng kiểm tra lại.")

elif app_mode == "ALPR từ Video":
    st.header("Chế độ 3: Xử lý từ một file Video")
    
    frame_skip = st.sidebar.slider("Xử lý mỗi N khung hình (frame skip)", min_value=1, max_value=30, value=10,
                                        help="Giá trị càng cao, xử lý càng nhanh nhưng có thể bỏ lỡ biển số. Giá trị 1 là xử lý mọi khung hình.")

    uploaded_video = st.file_uploader("Tải lên một video (MP4, MOV, AVI)...", type=["mp4", "mov", "avi"])

    if uploaded_video is not None:
        if st.button("🎬 Bắt đầu xử lý Video"):
            # Tạo file tạm để OpenCV có thể đọc
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())

            cap = cv2.VideoCapture(tfile.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Thiết lập VideoWriter để lưu video kết quả
            output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

            # Các biến để theo dõi tiến trình
            progress_bar = st.progress(0)
            status_text = st.empty()
            image_placeholder = st.empty()
            # all_results = []
            # seen_plates = set() # Dùng để tránh lưu trùng lặp biển số
            best_results = {}
            frame_count = 0
            with st.spinner(f"Đang xử lý video... có tổng cộng {total_frames} khung hình."):
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_count += 1
                    
                    # Chỉ xử lý khung hình nếu nó là frame đầu tiên hoặc đến lượt skip
                    if frame_count % frame_skip == 0 or frame_count == 1:
                        status_text.text(f"Đang xử lý khung hình {frame_count}/{total_frames}...")
                        result_image, results_list = process_image(frame, f"frame_{frame_count}")
                        
                        if results_list:
                            for result in results_list:
                                lp_text = result['full_text']
                                # Chuyển đổi confidence sang kiểu float để so sánh
                                new_confidence = float(result['avg_confidence'])

                                # Nếu biển số chưa có trong kết quả hoặc có confidence cao hơn,
                                # thì lưu hoặc cập nhật kết quả mới.
                                if lp_text not in best_results or new_confidence > float(best_results[lp_text]['avg_confidence']):
                                    best_results[lp_text] = result
                        
                        image_placeholder.image(result_image, channels="BGR", caption=f"Khung hình {frame_count}")
                        out_writer.write(result_image)
                    else:
                        # Với các frame bị bỏ qua, vẫn ghi vào video output để không bị giật
                        out_writer.write(frame)

                    # Cập nhật progress bar
                    progress_bar.progress(frame_count / total_frames)

            # Giải phóng tài nguyên
            cap.release()
            out_writer.release()

            st.success("🎉 Hoàn tất xử lý video!")
            
            # Hiển thị video kết quả
            st.video(output_video_path)

            if best_results:
                st.subheader("Tổng hợp các biển số đã nhận diện (kết quả tốt nhất cho mỗi xe):")
                # Chuyển các giá trị của dictionary thành list để tạo DataFrame
                final_results = list(best_results.values())
                df_all_results = pd.DataFrame(final_results)
                st.dataframe(df_all_results)

                @st.cache_data
                def convert_df_to_csv(df):
                    return df.to_csv(index=False).encode('utf-8')
                
                csv_data = convert_df_to_csv(df_all_results)
                st.download_button(
                    label="📥 Tải tất cả kết quả (CSV)",
                    data=csv_data,
                    file_name="video_results.csv",
                    mime='text/csv',
                )
            else:
                st.info("Không nhận diện được biển số nào trong video.")
# --- Màn hình chờ ---
else:
    st.info("Vui lòng chọn một chế độ xử lý từ thanh công cụ bên trái.")
    st.image("https://www.luxoft.com/upload/medialibrary/729/automated_license_plate_recognition.png")