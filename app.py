# app.py

import streamlit as st
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import time
from pathlib import Path
import tempfile # Thư viện để xử lý file tạm
import io
import zipfile

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
@st.cache_resource
def load_models():
    """Tải tất cả các model và cache chúng."""
    with st.spinner('Đang tải các model AI, vui lòng chờ...'):
        vehicle_detector = VehicleDetector(config.VEHICLE_DETECTION_MODEL_PATH)
        lp_detector = LicensePlateDetector(config.LP_DETECTION_MODEL_PATH)
        denoising_model = tf.keras.models.load_model(config.DENOISING_MODEL_PATH, compile=False)
        ocr = OcrReader(config.PADDLEOCR_MODEL_DIR, config.CHAR_DICT_PATH)

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


def process_single_vehicle(vehicle_crop, vehicle_id, lp_conf, two_line_ratio, verbose=False):
    processed_vehicle_img = vehicle_crop.copy()
    results_list = []

    lp_box = lp_detector.detect(vehicle_crop, lp_conf)
    if lp_box is None:
        return processed_vehicle_img, []

    lp_x1, lp_y1, lp_x2, lp_y2 = lp_box
    lp_crop = vehicle_crop[int(lp_y1):int(lp_y2), int(lp_x1):int(lp_x2)]

    images_for_ocr = preprocessor.process_lp_for_ocr(lp_crop, denoising_model, two_line_ratio=two_line_ratio)
    if not images_for_ocr:
        return processed_vehicle_img, []

    full_text, avg_confidence = ocr.recognize(images_for_ocr, vehicle_id, verbose=verbose)

    if full_text:
        results_list.append({
            'frame': vehicle_id.split('_')[0], 'id_vehicle': vehicle_id,
            'full_text': full_text, 'avg_confidence': f"{avg_confidence:.4f}"
        })

        label = f"LP:{full_text}"
        cv2.rectangle(processed_vehicle_img, (int(lp_x1), int(lp_y1)), (int(lp_x2), int(lp_y2)), (36, 255, 12), 2)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(processed_vehicle_img, (int(lp_x1), int(lp_y1) - h - 10), (int(lp_x1) + w, int(lp_y1)), (36, 255, 12), -1)
        cv2.putText(processed_vehicle_img, label, (int(lp_x1), int(lp_y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    return processed_vehicle_img, results_list


def run_alpr_on_frame(frame, image_name, vehicle_conf, lp_conf, two_line_ratio, verbose=False):
    output_image = frame.copy()
    all_results = []

    status_context = st.status(f"🔍 Đang xử lý {image_name}...", expanded=True) if verbose else None

    vehicle_boxes = vehicle_detector.detect(frame, vehicle_conf)

    if not np.any(vehicle_boxes):
        if verbose and status_context:
            status_context.update(label="Không tìm thấy xe.", state="warning", expanded=False)
        return output_image, []

    if verbose and status_context:
        status_context.update(label=f"Tìm thấy {len(vehicle_boxes)} xe. Đang nhận dạng...", state="running")

    for i, (x1, y1, x2, y2) in enumerate(vehicle_boxes):
        vehicle_id = f"{Path(image_name).stem}_{i}"
        vehicle_crop = frame[int(y1):int(y2), int(x1):int(x2)]

        processed_vehicle, results = process_single_vehicle(
            vehicle_crop, vehicle_id, lp_conf, two_line_ratio, verbose=verbose
        )

        output_image[int(y1):int(y2), int(x1):int(x2)] = processed_vehicle

        if results:
            all_results.extend(results)
            cv2.rectangle(output_image, (int(x1), int(y1)), (int(x2), int(y2)), (36, 255, 12), 2)

    if verbose and status_context:
        status_context.update(label="Hoàn tất!", state="complete", expanded=False)

    return output_image, all_results

# --- GIAO DIỆN CHÍNH ---

st.sidebar.header("Chọn chế độ")
app_mode = st.sidebar.selectbox("Vui lòng chọn chế độ xử lý:",
                                ["Chọn chế độ...", "ALPR 1 ảnh", "ALPR 1 thư mục",  "ALPR từ Video"])

st.sidebar.header("⚙️ Tinh chỉnh tham số")
st.sidebar.info("Các tham số này sẽ được áp dụng cho tất cả các chế độ.")
vehicle_conf_adj = st.sidebar.slider("Ngưỡng tin cậy phát hiện XE", 0.05, 0.95, config.VEHICLE_CONF_THRESHOLD, 0.05)
lp_conf_adj = st.sidebar.slider("Ngưỡng tin cậy phát hiện BIỂN SỐ", 0.05, 0.95, config.LP_CONF_THRESHOLD, 0.05)
ratio_adj = st.sidebar.slider("Tỷ lệ nhận diện biển 2 dòng", 0.2, 1.0, config.TWO_LINE_LP_ASPECT_RATIO_THRESHOLD, 0.05,
                            help="Giá trị càng thấp, càng dễ nhận diện biển số dọc (2 dòng).")

# --- CHẾ ĐỘ 1: XỬ LÝ 1 ẢNH ---
if app_mode == "ALPR 1 ảnh":
    st.header("Chế độ 1: Xử lý một ảnh duy nhất")
    uploaded_file = st.file_uploader("Tải lên một ảnh (JPG, PNG)...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)

        col1, col2 = st.columns(2)
        with col1:
            st.image(frame, channels="BGR", caption="Ảnh gốc")

        if st.button("🚀 Bắt đầu xử lý"):
            result_image, results_list = run_alpr_on_frame(
                frame, uploaded_file.name, vehicle_conf_adj, lp_conf_adj, ratio_adj, verbose=True
            )

            with col2:
                st.image(result_image, channels="BGR", caption="Ảnh kết quả")

            if results_list:
                st.subheader("Kết quả nhận diện:")
                df_results = pd.DataFrame(results_list)
                st.dataframe(df_results)

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

# --- CHẾ ĐỘ 2: XỬ LÝ THƯ MỤC (ĐÃ CẬP NHẬT) ---
elif app_mode == "ALPR 1 thư mục":
    st.header("Chế độ 2: Xử lý nhiều ảnh từ một thư mục")
    
    # Cho phép người dùng tải lên nhiều file
    uploaded_files = st.file_uploader(
        "Tải lên các ảnh (JPG, PNG) từ thư mục của bạn...", 
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.info(f"Đã chọn **{len(uploaded_files)}** ảnh. Nhấn nút bên dưới để bắt đầu.")
        
        if st.button("📁 Bắt đầu xử lý các ảnh đã chọn"):
            progress_bar = st.progress(0, text="Bắt đầu xử lý...")
            all_results = []
            
            # Dùng io.BytesIO để tạo file zip trong bộ nhớ
            zip_buffer = io.BytesIO()
            
            # Dictionary để lưu ảnh kết quả trong bộ nhớ
            processed_images_data = {}

            for i, uploaded_file in enumerate(uploaded_files):
                image_name = uploaded_file.name
                progress_text = f"Đang xử lý ảnh: {image_name} ({i + 1}/{len(uploaded_files)})"
                progress_bar.progress((i + 1) / len(uploaded_files), text=progress_text)
                
                # Đọc ảnh từ file tải lên
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                frame = cv2.imdecode(file_bytes, 1)

                # Chạy ALPR trên ảnh
                result_image, results_list = run_alpr_on_frame(
                    frame, image_name, vehicle_conf_adj, lp_conf_adj, ratio_adj, verbose=False
                )
                
                # Mã hóa ảnh kết quả sang định dạng PNG để lưu vào bộ nhớ
                is_success, buffer = cv2.imencode(".png", result_image)
                if is_success:
                    processed_images_data[image_name] = io.BytesIO(buffer)

                # Thêm tên file vào kết quả để xuất CSV
                if results_list:
                    for result in results_list:
                        result['image_name'] = image_name
                    all_results.extend(results_list)
                
                time.sleep(0.1) # Dừng một chút để UI mượt hơn

            progress_bar.empty()
            st.success("🎉 Hoàn tất xử lý tất cả ảnh!")

            # --- TẠO CÁC NÚT DOWNLOAD ---
            col1, col2 = st.columns(2)

            # 1. NÚT TẢI FILE ZIP
            with col1:
                if processed_images_data:
                    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                        for filename, data in processed_images_data.items():
                            zf.writestr(f"processed_{filename}", data.getvalue())
                    
                    st.download_button(
                        label="📥 Tải tất cả ảnh kết quả (.zip)",
                        data=zip_buffer.getvalue(),
                        file_name="processed_images.zip",
                        mime="application/zip",
                        use_container_width=True
                    )

            # 2. NÚT TẢI FILE CSV
            with col2:
                if all_results:
                    df = pd.DataFrame(all_results)
                    # Đổi tên và sắp xếp lại các cột theo yêu cầu
                    df.rename(columns={
                        'id_vehicle': 'vehicle_id',
                        'full_text': 'text',
                        'avg_confidence': 'confidence'
                    }, inplace=True)
                    df_final = df[['image_name', 'vehicle_id', 'text', 'confidence']]

                    csv_data = df_final.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Tải tất cả kết quả (.csv)",
                        data=csv_data,
                        file_name="folder_alpr_results.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            # Hiển thị bảng kết quả tổng hợp
            if all_results:
                st.subheader("Tổng hợp kết quả:")
                st.dataframe(df_final)
            else:
                st.info("Không nhận diện được biển số nào trong các ảnh đã chọn.")


# --- CHẾ ĐỘ 3: XỬ LÝ VIDEO ---
# --- CHẾ ĐỘ 3: XỬ LÝ VIDEO (ĐÃ CẬP NHẬT VỚI BỘ THEO DÕI XE) ---
# --- CHẾ ĐỘ 3: XỬ LÝ VIDEO (PHIÊN BẢN TỐI ƯU HÓA) ---
elif app_mode == "ALPR từ Video":
    st.header("Chế độ 3: Xử lý từ một file Video")
    st.sidebar.subheader("Tinh chỉnh Video")
    # Tăng giá trị mặc định của frame_skip vì tracker đã hiệu quả hơn
    frame_skip = st.sidebar.slider("Xử lý mỗi N khung hình (frame skip)", 1, 30, 5)
    # Thêm tùy chọn cho khoảng thời gian phát hiện lại xe
    detection_interval = st.sidebar.slider("Phát hiện lại xe sau mỗi X khung hình xử lý", 1, 50, 10, help="Giá trị càng cao, tốc độ càng nhanh nhưng có thể bỏ lỡ xe mới. Giá trị này được nhân với Frame Skip.")

    uploaded_video = st.file_uploader("Tải lên một video (MP4, MOV, AVI)...", type=["mp4", "mov", "avi"])

    roi_rect = None
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        tfile.close()

        video_filename = uploaded_video.name
        cap_preview = cv2.VideoCapture(tfile.name)
        ret, first_frame = cap_preview.read()
        cap_preview.release()

        if ret:
            st.subheader("Bước 1: (Bắt buộc) Xác định Vùng Quan Tâm (Region of Interest)")
            st.info("Bộ đếm và nhận dạng sẽ chỉ hoạt động với những xe đi vào vùng ROI bạn đã vẽ.")
            
            preview_frame = first_frame.copy()
            height, width, _ = preview_frame.shape

            col1, col2 = st.columns(2)
            with col1:
                x_min = st.slider("ROI X-min", 0, width, 0, 1)
                x_max = st.slider("ROI X-max", 0, width, width, 1)
            with col2:
                y_min = st.slider("ROI Y-min", 0, height, 0, 1)
                y_max = st.slider("ROI Y-max", 0, height, height, 1)

            if x_min >= x_max: x_max = x_min + 1
            if y_min >= y_max: y_max = y_min + 1

            roi_rect = (x_min, y_min, x_max, y_max)
            cv2.rectangle(preview_frame, (x_min, y_min), (x_max, y_max), (36, 255, 12), 3)
            st.image(preview_frame, channels="BGR", caption="Khung hình đầu tiên với ROI")

        st.subheader("Bước 2: Bắt đầu xử lý video")
        if st.button("🎬 Bắt đầu xử lý Video"):
            if roi_rect is None or (roi_rect[0] == 0 and roi_rect[2] == width and roi_rect[1] == 0 and roi_rect[3] == height):
                st.warning("Vui lòng xác định một vùng ROI cụ thể ở Bước 1 để bộ đếm hoạt động chính xác.")
                st.stop()

            cap = cv2.VideoCapture(tfile.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            out_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            out_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (out_width, out_height))

            # --- KHỞI TẠO CÁC BIẾN CHO TRACKER TỐI ƯU ---
            trackers = {} # {tracker_id: {'tracker': cv2.Tracker, 'box': (x,y,w,h), 'id_vehicle': None}}
            next_tracker_id = 0
            vehicle_pass_count = 0

            progress_bar = st.progress(0, text="Bắt đầu xử lý...")
            status_text = st.empty()
            image_placeholder = st.empty()
            all_video_results = []
            frame_count = 0
            processed_frame_count = 0

            with st.spinner("Đang xử lý video với bộ theo dõi tối ưu..."):
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    frame_count += 1
                    
                    # Luôn bắt đầu với một khung hình sạch để vẽ lên
                    display_frame = frame.copy()
                    
                    # KIỂM TRA XEM ĐÂY CÓ PHẢI LÀ KHUNG HÌNH CẦN XỬ LÝ KHÔNG
                    if frame_count % frame_skip == 0:
                        processed_frame_count += 1
                        status_text.text(f"Đang xử lý khung hình {frame_count}/{total_frames}...")

                        # 1. CHẠY PHÁT HIỆN XE ĐỊNH KỲ
                        if processed_frame_count % detection_interval == 0:
                            vehicle_boxes = vehicle_detector.detect(frame, vehicle_conf_adj)
                            for (x1, y1, x2, y2) in vehicle_boxes:
                                box_w, box_h = (x2 - x1), (y2 - y1)
                                center_x, center_y = x1 + box_w / 2, y1 + box_h / 2
                                
                                is_new = True
                                for tid, tdata in trackers.items():
                                    tx, ty, tw, th = tdata['box']
                                    if abs(center_x - (tx + tw/2)) < tw/2 and abs(center_y - (ty + th/2)) < th/2:
                                        is_new = False
                                        break
                                
                                if is_new:
                                    bbox_int = (int(x1), int(y1), int(box_w), int(box_h))
                                    new_tracker = cv2.TrackerCSRT_create()
                                    new_tracker.init(frame, bbox_int)
                                    trackers[next_tracker_id] = {
                                        'tracker': new_tracker,
                                        'box': bbox_int,
                                        'id_vehicle': None,
                                        'processed_image': None
                                    }
                                    next_tracker_id += 1

                        # 2. CẬP NHẬT VỊ TRÍ TỪ CÁC TRACKER
                        failed_trackers = []
                        for tracker_id, tdata in trackers.items():
                            success, box = tdata['tracker'].update(frame)
                            if success:
                                trackers[tracker_id]['box'] = box
                                x, y, w, h = [int(v) for v in box]
                                center_x, center_y = x + w / 2, y + h / 2
                                rx1, ry1, rx2, ry2 = roi_rect

                                if rx1 < center_x < rx2 and ry1 < center_y < ry2:
                                    if tdata['id_vehicle'] is None:
                                        vehicle_pass_count += 1
                                        assigned_id = vehicle_pass_count
                                        trackers[tracker_id]['id_vehicle'] = assigned_id

                                        vehicle_crop = frame[y:y+h, x:x+w]
                                        processed_vehicle, results = process_single_vehicle(
                                            vehicle_crop, f"v_{assigned_id}", lp_conf_adj, ratio_adj, verbose=False
                                        )
                                        trackers[tracker_id]['processed_image'] = processed_vehicle

                                        if results:
                                            for r in results:
                                                r['frame'] = frame_count
                                                r['id_vehicle'] = assigned_id
                                            all_video_results.extend(results)
                            else:
                                failed_trackers.append(tracker_id)
                        
                        # 3. XÓA CÁC TRACKER BỊ LỖI
                        for tracker_id in failed_trackers:
                            del trackers[tracker_id]

                    # ---- LOGIC VẼ (CHẠY TRÊN MỌI KHUNG HÌNH) ----
                    # Vẽ vùng ROI
                    rx1, ry1, rx2, ry2 = roi_rect
                    cv2.rectangle(display_frame, (rx1, ry1), (rx2, ry2), (36, 255, 12), 2)
                    
                    # Vẽ tất cả các xe đang được theo dõi
                    for tracker_id, tdata in trackers.items():
                        x, y, w, h = [int(v) for v in tdata['box']]
                        
                        # Vẽ lại ảnh xe đã xử lý nếu có
                        # Vẽ lại ảnh xe đã xử lý nếu có
                        if tdata.get('processed_image') is not None:
                            img_h, img_w, _ = display_frame.shape
                            crop_h, crop_w, _ = tdata['processed_image'].shape
                            
                            # ---- SỬA LỖI: THÊM KIỂM TRA TOÀN DIỆN HƠN ----
                            # Đảm bảo toàn bộ vùng ảnh nằm trọn trong khung hình trước khi vẽ
                            if y >= 0 and x >= 0 and (y + crop_h) <= img_h and (x + crop_w) <= img_w:
                                display_frame[y:y+crop_h, x:x+crop_w] = tdata['processed_image']
                        
                        # Vẽ hộp và ID
                        assigned_id = tdata.get('id_vehicle')
                        if assigned_id is not None:
                            id_label = f"ID: {assigned_id}"
                            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (36, 255, 12), 2)
                            (lw, lh), _ = cv2.getTextSize(id_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                            cv2.rectangle(display_frame, (x, y - lh - 10), (x + lw, y), (36, 255, 12), -1)
                            cv2.putText(display_frame, id_label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                    
                    # ---- GHI KHUNG HÌNH VÀ CẬP NHẬT GIAO DIỆN ----
                    # Ghi lại MỌI khung hình đã được vẽ vào video output
                    out_writer.write(display_frame)
                    
                    # Chỉ cập nhật giao diện trên các frame đã xử lý để tránh lag
                    if frame_count % frame_skip == 0:
                        image_placeholder.image(display_frame, channels="BGR")
                        progress_bar.progress(frame_count / total_frames)
            cap.release()
            out_writer.release()
            os.remove(tfile.name) 
            st.success("🎉 Hoàn tất xử lý video!")

            # --- HIỂN THỊ VÀ TẢI KẾT QUẢ VIDEO ---
            st.subheader("Kết quả Video đã xử lý")
            with open(output_video_path, 'rb') as f:
                video_bytes = f.read()
            st.video(video_bytes)
            st.download_button(label="📥 Tải video đã xử lý", data=video_bytes, file_name=f"processed_{video_filename}", mime="video/mp4")
            os.remove(output_video_path)

            if all_video_results:
                st.subheader("Tổng hợp các lượt nhận diện trong video:")
                df_all = pd.DataFrame(all_video_results)
                df_all.rename(columns={'full_text': 'text', 'avg_confidence': 'confidence'}, inplace=True)
                df_all['confidence'] = pd.to_numeric(df_all['confidence']) # Đảm bảo confidence là số
                
                # Giữ lại kết quả tốt nhất cho mỗi ID xe
                df_final = df_all.loc[df_all.groupby('id_vehicle')['confidence'].idxmax()]
                df_final = df_final[['frame', 'id_vehicle', 'text', 'confidence']].sort_values(by='id_vehicle')
                df_final['confidence'] = df_final['confidence'].apply(lambda x: f"{x:.4f}")

                st.dataframe(df_final)

                csv_data = df_final.to_csv(index=False).encode('utf-8')
                st.download_button(label="📥 Tải kết quả nhận diện (CSV)", data=csv_data, file_name=f"results_{Path(video_filename).stem}.csv", mime='text/csv')
            else:
                st.info("Không nhận diện được biển số nào hợp lệ trong video.")

# --- Màn hình chờ ---
else:
    st.info("Vui lòng chọn một chế độ xử lý từ thanh công cụ bên trái.")
    st.image("https://www.luxoft.com/upload/medialibrary/729/automated_license_plate_recognition.png")