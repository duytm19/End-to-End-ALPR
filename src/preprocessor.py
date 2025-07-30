import cv2
import numpy as np
from . import config

def correct_skew_hough(image, threshold=40):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 200, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=threshold, minLineLength=30, maxLineGap=15)

    if lines is None:
        return image

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 != 0:
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if -45 < angle < 45:
                angles.append(angle)

    if not angles:
        return image

    median_angle = np.median(angles)
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return rotated

def denoise_full_plate(image_bgr, denoising_model):
    original_h, original_w = image_bgr.shape[:2]
    gray_img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    model_input_shape = denoising_model.input_shape
    h_model, w_model = model_input_shape[1], model_input_shape[2]

    resized_img = cv2.resize(gray_img, (w_model, h_model))
    normalized_img = resized_img / 255.0
    input_tensor = np.expand_dims(normalized_img, axis=(0, -1))

    predicted_tensor = denoising_model.predict(input_tensor, verbose=0)
    denoised_normalized = np.squeeze(predicted_tensor, axis=(0, -1))
    denoised_img_model_size = (denoised_normalized * 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced_img = clahe.apply(denoised_img_model_size)
    final_denoised_img = cv2.resize(contrast_enhanced_img, (original_w, original_h))
    return final_denoised_img

def process_lp_for_ocr(lp_crop, denoising_model, two_line_ratio=config.TWO_LINE_LP_ASPECT_RATIO_THRESHOLD, hough_threshold=40):
    if lp_crop.size == 0:
        return []

    corrected_image = correct_skew_hough(lp_crop, threshold=hough_threshold)
    denoised_full_lp = denoise_full_plate(corrected_image, denoising_model)

    h, w = denoised_full_lp.shape
    image_parts = []

    if h > w * two_line_ratio:
        mid_point = h // 2
        top_line_img = denoised_full_lp[0:mid_point, :]
        bottom_line_img = denoised_full_lp[mid_point:, :]
        image_parts.extend([top_line_img, bottom_line_img])
    else:
        image_parts.append(denoised_full_lp)

    processed_images_for_ocr = []
    for part in image_parts:
        if part.size == 0: continue
        h_part, w_part = part.shape
        scale = config.TARGET_OCR_H / h_part
        new_w = min(int(w_part * scale), config.TARGET_OCR_W)

        resized_for_ocr = cv2.resize(part, (new_w, config.TARGET_OCR_H))
        padded_img = np.ones((config.TARGET_OCR_H, config.TARGET_OCR_W), dtype=np.uint8) * 255

        start_x = (config.TARGET_OCR_W - new_w) // 2
        padded_img[:, start_x:start_x + new_w] = resized_for_ocr
        processed_images_for_ocr.append(padded_img)

    return processed_images_for_ocr