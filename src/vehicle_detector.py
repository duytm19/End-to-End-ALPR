

from ultralytics import YOLO

class VehicleDetector:
    def __init__(self, model_path):
        """
        Khởi tạo và tải model phát hiện xe.
        """
        print("[INFO] Đang tải model phát hiện xe...")
        try:
            self.model = YOLO(model_path)
            print("[INFO] Tải model phát hiện xe thành công.")
        except Exception as e:
            print(f"[ERROR] Lỗi khi tải model phát hiện xe: {e}")
            self.model = None

    def detect(self, frame, conf_threshold=0.15):
        """
        Phát hiện xe trong một khung hình.
        Trả về:
            list: Danh sách các bounding box của xe [x1, y1, x2, y2].
        """
        if self.model is None:
            return []
        
        results = self.model.predict(frame, conf=conf_threshold, verbose=False)
        
        if results and len(results[0].boxes) > 0:
            return results[0].boxes.xyxy.cpu().numpy()
        return []