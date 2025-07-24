# src/main.py

import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import argparse

# Import các module và cấu hình từ các file khác trong package
from . import config
from .vehicle_detector import VehicleDetector
from .lp_detector import LicensePlateDetector
from . import preprocessor
from .ocr_reader import OcrReader

def main(args):
    # ---- 1. TẢI TẤT CẢ MODEL ----
    vehicle_detector = VehicleDetector(config.VEHICLE_DETECTION_MODEL_PATH)
    lp_detector = LicensePlateDetector(config.LP_DETECTION_MODEL_PATH)
    denoising_model = tf.keras.models.load_model(config.DENOISING_MODEL_PATH)
    ocr = OcrReader(config.PADDLEOCR_MODEL_DIR, config.CHAR_DICT_PATH)

    if not all([vehicle_detector.model, lp_detector.model, denoising_model, ocr.ocr_reader]):
        print("[ERROR] Một hoặc nhiều model không thể tải. Thoát chương trình.")
        return

    # ---- 2. CHUẨN BỊ ----
    os.makedirs(config.PROCESSED_LP_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    frame = cv2.imread(args.image_path)
    if frame is None:
        print(f"[ERROR] Không thể đọc ảnh từ: {args.image_path}")
        return

    output_image = frame.copy()
    csv_results = []
    print(f"\n--- BẮT ĐẦU XỬ LÝ ẢNH: {args.image_path} ---")

    # ---- 3. PHÁT HIỆN XE ----
    vehicle_boxes = vehicle_detector.detect(frame, config.VEHICLE_CONF_THRESHOLD)
    if len(vehicle_boxes) == 0:
        print("❌ LỖI: Không phát hiện được xe.")
    else:
        print(f"✅ Tìm thấy {len(vehicle_boxes)} xe.")

    # ---- 4. LẶP QUA TỪNG XE ----
    for i, (x1, y1, x2, y2) in enumerate(vehicle_boxes):
        vehicle_id = i
        print(f"\n--- Đang xử lý Xe ID: {vehicle_id} ---")
        vehicle_crop = frame[int(y1):int(y2), int(x1):int(x2)]

        # ---- 5. PHÁT HIỆN BIỂN SỐ ----
        lp_box = lp_detector.detect(vehicle_crop, config.LP_CONF_THRESHOLD)
        if lp_box is None:
            print(f"  - ⚠️ CẢNH BÁO: Không tìm thấy biển số trên Xe ID: {vehicle_id}.")
            continue
        
        print(f"  - ✅ Tìm thấy biển số. Bắt đầu tiền xử lý...")
        lp_x1, lp_y1, lp_x2, lp_y2 = lp_box
        lp_crop = vehicle_crop[int(lp_y1):int(lp_y2), int(lp_x1):int(lp_x2)]

        # ---- 6. TIỀN XỬ LÝ BIỂN SỐ ----
        images_for_ocr = preprocessor.process_lp_for_ocr(lp_crop, denoising_model)
        if not images_for_ocr:
            print("  - ⚠️ CẢNH BÁO: Tiền xử lý không tạo ra được ảnh để nhận diện.")
            continue
        
        # ---- 7. NHẬN DẠNG KÝ TỰ (OCR) ----
        print("  - ✅ Bắt đầu nhận diện ký tự (OCR)...")
        full_text, avg_confidence = ocr.recognize(images_for_ocr, vehicle_id)

        # ---- 8. XỬ LÝ VÀ LƯU KẾT QUẢ ----
        if full_text:
            print(f"  - ✅ TỔNG HỢP: '{full_text}' (Conf Trung Bình: {avg_confidence:.4f})")
            csv_results.append({
                'frame': os.path.basename(args.image_path), 'id_vehicle': vehicle_id,
                'full_text': full_text, 'avg_confidence': f"{avg_confidence:.4f}"
            })
            
            # Vẽ kết quả lên ảnh
            label = f"ID:{vehicle_id} LP:{full_text}"
            cv2.rectangle(output_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(output_image, (int(x1), int(y1) - h - 10), (int(x1) + w, int(y1)), (0, 255, 0), -1)
            cv2.putText(output_image, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        else:
            print(f"  - ⚠️ CẢNH BÁO: OCR thất bại hoàn toàn cho Xe ID: {vehicle_id}.")
    
    # ---- 9. LƯU FILE OUTPUT ----
    base_filename = os.path.splitext(os.path.basename(args.image_path))[0]
    output_image_path = os.path.join(config.OUTPUT_DIR, f"{base_filename}_result.jpg")
    output_csv_path = os.path.join(config.OUTPUT_DIR, 'results.csv')

    cv2.imwrite(output_image_path, output_image)
    if csv_results:
        df = pd.DataFrame(csv_results)
        df.to_csv(output_csv_path, index=False, mode='a', header=not os.path.exists(output_csv_path))
        print(f"\n✅ Xử lý hoàn tất! Kết quả được lưu tại:\n  - Ảnh: {output_image_path}\n  - CSV:  {output_csv_path}")
    else:
        print("\n⚠️ Không có biển số nào được nhận diện thành công.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ALPR Pipeline for a single image.')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image file.')
    args = parser.parse_args()
    main(args)