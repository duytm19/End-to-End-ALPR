import streamlit as st
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import time
from pathlib import Path
import tempfile
import io
import zipfile

from src import config
from src.vehicle_detector import VehicleDetector
from src.lp_detector import LicensePlateDetector
from src import preprocessor
from src.ocr_reader import OcrReader


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
    with st.spinner('Đang tải các model AI, vui lòng chờ...'):
        vehicle_detector = VehicleDetector(config.VEHICLE_DETECTION_MODEL_PATH)
        lp_detector = LicensePlateDetector(config.LP_DETECTION_MODEL_PATH)
        denoising_model = tf.keras.models.load_model(config.DENOISING_MODEL_PATH, compile=False)
        ocr = OcrReader(config.PADDLEOCR_MODEL_DIR, config.CHAR_DICT_PATH)

    if not all([vehicle_detector.model, lp_detector.model, denoising_model, ocr.ocr_reader]):
        st.error("Lỗi nghiêm trọng: Một hoặc nhiều model không thể tải. Vui lòng kiểm tra đường dẫn trong file config.py.")
        return None
    return vehicle_detector, lp_detector, denoising_model, ocr

models = load_models()
if models:
    vehicle_detector, lp_detector, denoising_model, ocr = models
    st.sidebar.success("Tải model thành công!")
else:
    st.stop()


def process_single_vehicle(vehicle_crop, vehicle_id, lp_conf, two_line_ratio, hough_threshold, verbose=False):
    processed_vehicle_img = vehicle_crop.copy()
    results_list = []

    lp_box = lp_detector.detect(vehicle_crop, lp_conf)
    if lp_box is None:
        return processed_vehicle_img, []

    lp_x1, lp_y1, lp_x2, lp_y2 = lp_box
    lp_crop = vehicle_crop[int(lp_y1):int(lp_y2), int(lp_x1):int(lp_x2)]

    images_for_ocr = preprocessor.process_lp_for_ocr(
        lp_crop, denoising_model,
        two_line_ratio=two_line_ratio,
        hough_threshold=hough_threshold
    )
    if not images_for_ocr:
        return processed_vehicle_img, []

    full_text, avg_confidence = ocr.recognize(images_for_ocr, vehicle_id, verbose=verbose)

    if full_text:
        results_list.append({
            'full_text': full_text, 'avg_confidence': avg_confidence
        })

        label = f"LP:{full_text}"
        cv2.rectangle(processed_vehicle_img, (int(lp_x1), int(lp_y1)), (int(lp_x2), int(lp_y2)), (36, 255, 12), 2)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(processed_vehicle_img, (int(lp_x1), int(lp_y1) - h - 10), (int(lp_x1) + w, int(lp_y1)), (36, 255, 12), -1)
        cv2.putText(processed_vehicle_img, label, (int(lp_x1), int(lp_y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    return processed_vehicle_img, results_list


def run_alpr_on_frame(frame, image_name, vehicle_conf, lp_conf, two_line_ratio, hough_threshold, verbose=False):
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
            vehicle_crop, vehicle_id, lp_conf, two_line_ratio, hough_threshold, verbose=verbose
        )

        output_image[int(y1):int(y2), int(x1):int(x2)] = processed_vehicle

        if results:
            for res in results:
                res['id_vehicle'] = vehicle_id
                res['frame'] = Path(image_name).stem
                res['avg_confidence'] = f"{res['avg_confidence']:.4f}"
            all_results.extend(results)
            cv2.rectangle(output_image, (int(x1), int(y1)), (int(x2), int(y2)), (36, 255, 12), 2)

    if verbose and status_context:
        status_context.update(label="Hoàn tất!", state="complete", expanded=False)

    return output_image, all_results


st.sidebar.header("Chọn chế độ")
app_mode = st.sidebar.selectbox("Vui lòng chọn chế độ xử lý:",
                                ["Chọn chế độ...", "ALPR 1 ảnh", "ALPR 1 thư mục",  "ALPR từ Video"])

st.sidebar.header("⚙️ Tinh chỉnh tham số")
st.sidebar.info("Các tham số này sẽ được áp dụng cho tất cả các chế độ.")
vehicle_conf_adj = st.sidebar.slider("Ngưỡng tin cậy phát hiện XE", 0.05, 0.95, config.VEHICLE_CONF_THRESHOLD, 0.05)
lp_conf_adj = st.sidebar.slider("Ngưỡng tin cậy phát hiện BIỂN SỐ", 0.05, 0.95, config.LP_CONF_THRESHOLD, 0.05)
ratio_adj = st.sidebar.slider("Tỷ lệ nhận diện biển 2 dòng", 0.2, 1.0, config.TWO_LINE_LP_ASPECT_RATIO_THRESHOLD, 0.05,
                            help="Giá trị càng thấp, càng dễ nhận diện biển số dọc (2 dòng).")
# === THÊM SLIDER MỚI TẠI ĐÂY ===
hough_threshold_adj = st.sidebar.slider("Ngưỡng Hough (Chỉnh nghiêng)", 10, 150, 40, 5,
                                      help="Ngưỡng để phát hiện các đường thẳng trong ảnh biển số nhằm xoay nó về phương ngang. Tăng giá trị nếu biển số bị nhiễu và xoay sai.")


if app_mode == "ALPR 1 ảnh":
    st.header("Chế độ 1: Xử lý một ảnh duy nhất")
    uploaded_file = st.file_uploader("Tải lên một ảnh (JPG, PNG)...", type=["jpg", "jpeg", "png"])

    if 'processed_image' not in st.session_state:
        st.session_state.processed_image = None
        st.session_state.results_list = None
        st.session_state.image_name_with_ext = None
        st.session_state.image_name_stem = None


    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)

        col1, col2 = st.columns(2)
        with col1:
            st.image(frame, channels="BGR", caption="Ảnh gốc")

        if st.button("🚀 Bắt đầu xử lý"):
            image_name_with_ext = uploaded_file.name
            image_name_stem = Path(image_name_with_ext).stem

            result_image, results_list = run_alpr_on_frame(
                frame, image_name_with_ext, vehicle_conf_adj, lp_conf_adj, ratio_adj, hough_threshold_adj, verbose=True
            )

            st.session_state.processed_image = result_image
            st.session_state.results_list = results_list
            st.session_state.image_name_with_ext = image_name_with_ext
            st.session_state.image_name_stem = image_name_stem


    if st.session_state.processed_image is not None and st.session_state.results_list is not None:

        _, col2_display = st.columns(2)
        with col2_display:
             st.image(st.session_state.processed_image, channels="BGR", caption="Ảnh kết quả")

        if st.session_state.results_list:
            st.subheader("Kết quả nhận diện:")
            df_display = pd.DataFrame(st.session_state.results_list)
            st.dataframe(df_display)

            df_for_csv = df_display[['id_vehicle', 'full_text', 'avg_confidence']].copy()
            df_for_csv.rename(columns={'full_text': 'text', 'avg_confidence': 'confidence'}, inplace=True)

            @st.cache_data
            def convert_df_to_csv(df):
                return df.to_csv(index=False).encode('utf-8')

            csv_data = convert_df_to_csv(df_for_csv)

            is_success, buffer = cv2.imencode(".png", st.session_state.processed_image)
            if is_success:
                image_bytes = buffer.tobytes()
                dl_col1, dl_col2 = st.columns(2)
                with dl_col1:
                    st.download_button(
                        label="📥 Tải ảnh kết quả",
                        data=image_bytes,
                        file_name=f"processed_{st.session_state.image_name_with_ext}",
                        mime="image/png",
                        use_container_width=True
                    )
                with dl_col2:
                    st.download_button(
                        label="📥 Tải file CSV",
                        data=csv_data,
                        file_name=f"results_{st.session_state.image_name_stem}.csv",
                        mime='text/csv',
                        use_container_width=True
                    )
        else:
            st.info("Không nhận diện được biển số nào trong ảnh.")


elif app_mode == "ALPR 1 thư mục":
    st.header("Chế độ 2: Xử lý nhiều ảnh từ một thư mục")

    if 'folder_results' not in st.session_state:
        st.session_state.folder_results = None
        st.session_state.processed_images_data = None

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
            processed_images_data = {}

            for i, uploaded_file in enumerate(uploaded_files):
                image_name = uploaded_file.name
                progress_text = f"Đang xử lý ảnh: {image_name} ({i + 1}/{len(uploaded_files)})"
                progress_bar.progress((i + 1) / len(uploaded_files), text=progress_text)

                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                frame = cv2.imdecode(file_bytes, 1)

                result_image, results_list = run_alpr_on_frame(
                    frame, image_name, vehicle_conf_adj, lp_conf_adj, ratio_adj, hough_threshold_adj, verbose=False
                )

                is_success, buffer = cv2.imencode(".png", result_image)
                if is_success:
                    processed_images_data[f"processed_{image_name}"] = io.BytesIO(buffer)

                if results_list:
                    for result in results_list:
                        result['image_name'] = result.pop('frame')
                    all_results.extend(results_list)

                time.sleep(0.1)

            progress_bar.empty()
            st.success("🎉 Hoàn tất xử lý tất cả ảnh!")


            st.session_state.folder_results = all_results
            st.session_state.processed_images_data = processed_images_data

    if st.session_state.folder_results is not None and st.session_state.processed_images_data is not None:
        col1, col2 = st.columns(2)

        with col1:
            if st.session_state.processed_images_data:
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                    for filename, data in st.session_state.processed_images_data.items():
                        zf.writestr(filename, data.getvalue())

                st.download_button(
                    label="📥 Tải tất cả ảnh kết quả (.zip)",
                    data=zip_buffer.getvalue(),
                    file_name="processed_images.zip",
                    mime="application/zip",
                    use_container_width=True
                )

        with col2:
            if st.session_state.folder_results:
                df = pd.DataFrame(st.session_state.folder_results)
                df.rename(columns={
                    'id_vehicle': 'vehicle_id', 'full_text': 'text', 'avg_confidence': 'confidence'
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

        if st.session_state.folder_results:
            st.subheader("Tổng hợp kết quả:")
            df_final_display = pd.DataFrame(st.session_state.folder_results)
            df_final_display.rename(columns={
                'id_vehicle': 'vehicle_id', 'full_text': 'text', 'avg_confidence': 'confidence'
            }, inplace=True)
            st.dataframe(df_final_display[['image_name', 'vehicle_id', 'text', 'confidence']])
        else:
            st.info("Không nhận diện được biển số nào trong các ảnh đã chọn.")

elif app_mode == "ALPR từ Video":
    st.header("Chế độ 3: Xử lý từ một file Video")
    st.sidebar.subheader("Tinh chỉnh Video")
    frame_skip = st.sidebar.slider("Xử lý mỗi N khung hình (frame skip)", 1, 30, 5)

    if 'video_results' not in st.session_state:
        st.session_state.video_results = None
        st.session_state.processed_video_bytes = None
        st.session_state.original_video_filename = None

    uploaded_video = st.file_uploader("Tải lên một video (MP4, MOV, AVI)...", type=["mp4", "mov", "avi"])

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap_preview = cv2.VideoCapture(tfile.name)
        ret, first_frame = cap_preview.read()
        cap_preview.release()

        if ret:
            st.subheader("Bước 1: (Tùy chọn) Xác định Vùng Quan Tâm (ROI)")
            st.info("Nhận dạng sẽ chỉ được thực hiện với những xe có tâm nằm trong vùng ROI. Nếu không vẽ, toàn bộ khung hình sẽ được xử lý.")

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
            st.image(preview_frame, channels="BGR", caption="Khung hình đầu tiên với ROI đã chọn")
        else:
            st.error("Không thể đọc được video. Vui lòng thử lại với file khác.")
            roi_rect = None

        st.subheader("Bước 2: Bắt đầu xử lý video")
        if st.button("🎬 Bắt đầu xử lý Video"):
            if roi_rect:
                video_vehicle_detector = VehicleDetector(config.VEHICLE_DETECTION_MODEL_PATH)
                cap = cv2.VideoCapture(tfile.name)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                out_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                out_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_writer = cv2.VideoWriter(output_video_path, fourcc, fps / frame_skip, (out_width, out_height))

                track_history = {}
                progress_bar = st.progress(0, text="Bắt đầu xử lý...")
                image_placeholder = st.empty()
                frame_count = 0

                with st.spinner("Đang xử lý video với ByteTrack..."):
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret: break
                        frame_count += 1

                        if frame_count % frame_skip != 0: continue

                        display_frame = frame.copy()

                        results = video_vehicle_detector.model.track(frame, persist=True, conf=vehicle_conf_adj, tracker="bytetrack.yaml", verbose=False)

                        rx1, ry1, rx2, ry2 = roi_rect
                        cv2.rectangle(display_frame, (rx1, ry1), (rx2, ry2), (36, 255, 12), 2)

                        if results[0].boxes.id is not None:
                            boxes = results[0].boxes.xyxy.cpu().numpy()
                            track_ids = results[0].boxes.id.int().cpu().tolist()

                            for box, track_id in zip(boxes, track_ids):
                                x1, y1, x2, y2 = box
                                center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2


                                if rx1 < center_x < rx2 and ry1 < center_y < ry2:
                                    vehicle_crop = frame[int(y1):int(y2), int(x1):int(x2)]
                                    _, ocr_results = process_single_vehicle(
                                        vehicle_crop, str(track_id), lp_conf_adj, ratio_adj, hough_threshold_adj, verbose=False
                                    )

                                    if ocr_results:
                                        current_best = ocr_results[0]
                                        if track_id not in track_history or current_best['avg_confidence'] > track_history[track_id]['confidence']:
                                            track_history[track_id] = {
                                                'text': current_best['full_text'],
                                                'confidence': current_best['avg_confidence'],
                                                'frame': frame_count,
                                            }

                        if results[0].boxes.id is not None:
                            for box, track_id in zip(results[0].boxes.xyxy.cpu().numpy(), results[0].boxes.id.int().cpu().tolist()):
                                x1_v, y1_v, x2_v, y2_v = [int(v) for v in box]
                                cv2.rectangle(display_frame, (x1_v, y1_v), (x2_v, y2_v), (36, 255, 12), 2)
                                label = f"ID: {track_id}"
                                if track_id in track_history:
                                    label += f" LP: {track_history[track_id]['text']}"
                                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                                cv2.rectangle(display_frame, (x1_v, y1_v - h - 10), (x1_v + w, y1_v), (36, 255, 12), -1)
                                cv2.putText(display_frame, label, (x1_v, y1_v - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

                        out_writer.write(display_frame)
                        image_placeholder.image(display_frame, channels="BGR")
                        progress_text = f"Đang xử lý khung hình: {frame_count}/{total_frames}"
                        progress_bar.progress(frame_count / total_frames, text=progress_text)

                cap.release()
                out_writer.release()
                st.success("🎉 Hoàn tất xử lý video!")

                with open(output_video_path, 'rb') as f:
                    video_bytes = f.read()
                os.remove(output_video_path)

                st.session_state.processed_video_bytes = video_bytes
                st.session_state.video_results = track_history
                st.session_state.original_video_filename = uploaded_video.name


        try:
            os.remove(tfile.name)
        except:
            pass

    if st.session_state.processed_video_bytes:
        st.subheader("Kết quả Video đã xử lý")
        st.video(st.session_state.processed_video_bytes)
        st.download_button(
            label="📥 Tải video đã xử lý",
            data=st.session_state.processed_video_bytes,
            file_name=f"processed_{st.session_state.original_video_filename}",
            mime="video/mp4"
        )

        if st.session_state.video_results:
            st.subheader("Tổng hợp các lượt nhận diện trong video:")
            final_results = []
            for track_id, data in st.session_state.video_results.items():
                final_results.append({
                    'frame': data['frame'], 'id_vehicle': track_id,
                    'text': data['text'], 'confidence': f"{data['confidence']:.4f}"
                })
            df_final = pd.DataFrame(final_results).sort_values(by='id_vehicle')
            st.dataframe(df_final)

            csv_data = df_final.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Tải kết quả nhận diện (CSV)",
                data=csv_data,
                file_name=f"results_{Path(st.session_state.original_video_filename).stem}.csv",
                mime='text/csv'
            )
        else:
            st.info("Không nhận diện được biển số nào hợp lệ trong video.")

else:
    st.info("Vui lòng chọn một chế độ xử lý từ thanh công cụ bên trái.")
    st.image("https://www.luxoft.com/upload/medialibrary/729/automated_license_plate_recognition.png")