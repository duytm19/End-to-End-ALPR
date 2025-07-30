from ultralytics import YOLO

class VehicleDetector:
    def __init__(self, model_path):
        print("[INFO] Downloading vehicle detection model...")
        try:
            self.model = YOLO(model_path)
            print("[INFO] Model loaded successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            self.model = None

    def detect(self, frame, conf_threshold=0.15):
        if self.model is None:
            return []
        
        results = self.model.predict(frame, conf=conf_threshold, verbose=False)
        
        if results and len(results[0].boxes) > 0:
            return results[0].boxes.xyxy.cpu().numpy()
        return []