from ultralytics import YOLO
import numpy as np

class LicensePlateDetector:
    def __init__(self, model_path):
        print("[INFO] Downloading license plate detection model...")
        try:
            self.model = YOLO(model_path)
            print("[INFO] Model loaded successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            self.model = None

    def detect(self, vehicle_crop, conf_threshold=0.35):
       
        if self.model is None or vehicle_crop.size == 0:
            return None
        results = self.model.predict(vehicle_crop, conf=conf_threshold, verbose=False)
        
        if results and len(results[0].boxes) > 0:
            best_box = results[0].boxes.xyxy.cpu().numpy()[0]
            return best_box
        return None