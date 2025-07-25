# src/config.py

# --- Đường dẫn model ---
VEHICLE_DETECTION_MODEL_PATH = 'models/Vehicle.pt'
LP_DETECTION_MODEL_PATH = 'models/LicensePlate.pt'
DENOISING_MODEL_PATH = 'models/Denoise.h5'
PADDLEOCR_MODEL_DIR = 'models/InferenceModel/'
CHAR_DICT_PATH = 'models/dict.txt'

# --- Cấu hình cho việc xử lý ảnh ---
TARGET_OCR_H = 48
TARGET_OCR_W = 64
TWO_LINE_LP_ASPECT_RATIO_THRESHOLD = 0.4

# --- Cấu hình cho các model detector ---
VEHICLE_CONF_THRESHOLD = 0.05
LP_CONF_THRESHOLD = 0.35

# --- Đường dẫn Output ---
OUTPUT_DIR = 'output'
PROCESSED_LP_DIR = 'processed_lp'
SUCCESSFUL_VEHICLES_DIR = 'successful_vehicles'