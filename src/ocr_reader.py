from paddleocr import PaddleOCR
import numpy as np
import os
import cv2
from . import config

class OcrReader:
    def __init__(self, model_dir, char_dict_path):
        try:
            self.ocr_reader = PaddleOCR(
                use_angle_cls=True, use_gpu=True, lang='en', det=False,
                rec_model_dir=model_dir, rec_char_dict_path=char_dict_path, show_log=False
            )
        except Exception as e:
            print(f"[ERROR] Error downloading model: {e}")
            self.ocr_reader = None

    def recognize(self, image_parts, vehicle_id, verbose=True):
        if self.ocr_reader is None or not image_parts:
            return "", 0.0

        ocr_parts_results = []
        for idx, ocr_input_image in enumerate(image_parts):
            if verbose:
                filename = f"vehicle_{vehicle_id}_lp_part_{idx+1}.png"
                save_path = os.path.join(config.PROCESSED_LP_DIR, filename)
                cv2.imwrite(save_path, ocr_input_image)

            ocr_result = self.ocr_reader.ocr(ocr_input_image, cls=True, det=False)
            if ocr_result and ocr_result[0]:
                text, confidence = ocr_result[0][0]
                text = text.upper().replace(" ", "")
                ocr_parts_results.append({'text': text, 'confidence': confidence})
            else:
                ocr_parts_results.append({'text': '', 'confidence': 0.0})

        if any(part['text'] for part in ocr_parts_results):
            full_text = "".join([part['text'] for part in ocr_parts_results])
            valid_confidences = [part['confidence'] for part in ocr_parts_results if part['text']]
            avg_confidence = np.mean(valid_confidences) if valid_confidences else 0.0
            return full_text, avg_confidence
        
        return "", 0.0