# src/lp_detector.py

from ultralytics import YOLO
import numpy as np

class LicensePlateDetector:
    def __init__(self, model_path):
        """
        Khởi tạo và tải model phát hiện biển số.
        """
        print("[INFO] Đang tải model phát hiện biển số...")
        try:
            self.model = YOLO(model_path)
            print("[INFO] Tải model phát hiện biển số thành công.")
        except Exception as e:
            print(f"[ERROR] Lỗi khi tải model phát hiện biển số: {e}")
            self.model = None

    def detect(self, vehicle_crop, conf_threshold=0.35):
        """
        Phát hiện biển số trong ảnh xe đã crop.
        Chỉ trả về bounding box có độ tin cậy cao nhất.
        Trả về:
            np.array: Bounding box của biển số [x1, y1, x2, y2] hoặc None.
        """
        if self.model is None or vehicle_crop.size == 0:
            return None
            
        results = self.model.predict(vehicle_crop, conf=conf_threshold, verbose=False)
        
        if results and len(results[0].boxes) > 0:
            # Trả về box có confidence cao nhất
            best_box = results[0].boxes.xyxy.cpu().numpy()[0]
            return best_box
        return None