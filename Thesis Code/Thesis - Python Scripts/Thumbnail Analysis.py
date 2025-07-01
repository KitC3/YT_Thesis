import os
import pandas as pd
import numpy as np
import cv2
from PIL import Image, ImageStat
from deepface import DeepFace
from colorthief import ColorThief
from ultralytics import YOLO
from easyocr import Reader
from skimage import filters
from skimage.morphology import disk
import extcolors
from Pylette import extract_colors
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging

# --- Configuration ---
THUMBNAILS_DIR = r"<PATH_TO_YOUR_THUMBNAILS_DIRECTORY>"
OUTPUT_DIR = r"<PATH_TO_YOUR_OUTPUT_DIRECTORY>"

# --- Logging Setup ---
logging.basicConfig(filename='thumbnail_analysis.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

def get_all_thumbnails():
    files = [f for f in os.listdir(THUMBNAILS_DIR) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    return [os.path.join(THUMBNAILS_DIR, f) for f in files]

# --- Model/Resource Initialization (Load Once) ---
yolo_model = YOLO('yolov8x.pt')
easyocr_reader = Reader(['en'], gpu=False)

def analyze_color_and_brightness(image_path):
    results = {}
    try:
        with Image.open(image_path) as img_pil:
            stat = ImageStat.Stat(img_pil)
            results['brightness_pil'] = sum(stat.mean) / len(stat.mean)
    except Exception as e:
        logging.warning(f"Color/Brightness PIL error: {e}")
        results['brightness_pil'] = None

    try:
        ct = ColorThief(image_path)
        results['dominant_color'] = str(ct.get_color(quality=1))
        results['color_palette'] = str(ct.get_palette(color_count=5))
    except Exception as e:
        logging.warning(f"ColorThief error: {e}")
        results['dominant_color'] = None
        results['color_palette'] = None

    try:
        colors, _ = extcolors.extract_from_path(image_path)
        results['dominant_colors_extcolors'] = str(colors[:5])
    except Exception as e:
        logging.warning(f"extcolors error: {e}")
        results['dominant_colors_extcolors'] = None

    try:
        palette = extract_colors(image=image_path, palette_size=5)
        results['colorfulness_pylette'] = np.std(np.array(palette[0].rgb)) if len(palette) > 0 else 0
    except Exception as e:
        logging.warning(f"Pylette error: {e}")
        results['colorfulness_pylette'] = None

    try:
        img_cv = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        (R, G, B) = cv2.split(img_rgb.astype("float"))
        rg = np.abs(R - G)
        yb = np.abs(0.5 * (R + G) - B)
        std_rg = np.std(rg)
        std_yb = np.std(yb)
        mean_rg = np.mean(rg)
        mean_yb = np.mean(yb)
        results['colorfulness_cv'] = np.sqrt(std_rg ** 2 + std_yb ** 2) + (0.3 * np.sqrt(mean_rg ** 2 + mean_yb ** 2))
        img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        results['brightness_cv'] = np.mean(img_gray)
    except Exception as e:
        logging.warning(f"OpenCV color/brightness error: {e}")
        results['colorfulness_cv'] = None
        results['brightness_cv'] = None

    results['image_path'] = image_path
    return results

def analyze_visual_complexity(image_path):
    try:
        img_cv = cv2.imread(image_path)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        entropy = filters.rank.entropy(gray, disk(9))
        complexity_score = np.mean(entropy)
        return {
            'image_path': image_path,
            'edge_count': int(np.sum(edges > 0)),
            'entropy_mean': float(complexity_score)
        }
    except Exception as e:
        logging.warning(f"Visual complexity error: {e}")
        return {
            'image_path': image_path,
            'edge_count': None,
            'entropy_mean': None
        }

def analyze_objects_and_concreteness(image_path):
    results = {'image_path': image_path}
    try:
        yolo_detections = yolo_model(image_path)[0]
        yolo_class_names = yolo_detections.names
        yolo_objects = [yolo_class_names[int(cls_id)] for cls_id in yolo_detections.boxes.cls.tolist()]
        results['yolo_objects'] = str(yolo_objects)
        results['object_count_yolo'] = len(yolo_objects)
    except Exception as e:
        logging.warning(f"YOLO error: {e}")
        results['yolo_objects'] = None
        results['object_count_yolo'] = None
    return results

def analyze_faces_and_emotions(image_path):
    results = {'image_path': image_path}
    try:
        deepface_analysis = DeepFace.analyze(img_path=image_path, actions=['age', 'gender', 'emotion'])
        face = deepface_analysis[0] if isinstance(deepface_analysis, list) else deepface_analysis
        results['deepface_emotion'] = face.get('emotion', {})
        results['age'] = face.get('age', None)
        results['gender'] = face.get('gender', None)
        results['dominant_emotion'] = face.get('dominant_emotion', None)
        results['face_count'] = len(deepface_analysis) if isinstance(deepface_analysis, list) else 1
    except Exception as e:
        logging.warning(f"DeepFace error: {e}")
        results['deepface_emotion'] = str(e)
        results['age'] = None
        results['gender'] = None
        results['dominant_emotion'] = None
        results['face_count'] = 0
    return results

def analyze_text_all(image_path):
    results = {'image_path': image_path}
    try:
        text_easyocr = easyocr_reader.readtext(image_path, detail=0)
        results['easyocr_text'] = str(text_easyocr)
    except Exception as e:
        logging.warning(f"EasyOCR error: {e}")
        results['easyocr_text'] = None
    return results

def analyze_element_complexity(objects_col, text_col, image_path):
    try:
        objects = eval(objects_col) if isinstance(objects_col, str) and objects_col else []
        unique_objects = set(objects)
    except Exception as e:
        logging.warning(f"Element complexity objects error: {e}")
        unique_objects = set()
    try:
        texts = eval(text_col) if isinstance(text_col, str) and text_col else []
        text_count = len(texts)
    except Exception as e:
        logging.warning(f"Element complexity text error: {e}")
        text_count = 0
    return {
        'image_path': image_path,
        'element_complexity': len(unique_objects) + text_count,
        'unique_object_count': len(unique_objects),
        'text_count': text_count
    }

def run_parallel_analysis(img_paths, func, desc):
    results = []
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(func, p): p for p in img_paths}
        for f in tqdm(as_completed(futures), total=len(futures), desc=desc):
            try:
                results.append(f.result())
            except Exception as e:
                logging.error(f"Parallel analysis error in {desc}: {e}")
    return results

if __name__ == "__main__":
    img_paths = get_all_thumbnails()
    df_color_brightness = pd.DataFrame(run_parallel_analysis(img_paths, analyze_color_and_brightness, "Color/Brightness"))
    df_color_brightness.to_csv(os.path.join(OUTPUT_DIR, 'color_brightness.csv'), index=False)

    df_visual_complexity = pd.DataFrame(run_parallel_analysis(img_paths, analyze_visual_complexity, "Visual Complexity"))
    df_visual_complexity.to_csv(os.path.join(OUTPUT_DIR, 'visual_complexity.csv'), index=False)

    df_object_concreteness = pd.DataFrame(run_parallel_analysis(img_paths, analyze_objects_and_concreteness, "Object/Concreteness"))
    df_object_concreteness.to_csv(os.path.join(OUTPUT_DIR, 'object_concreteness.csv'), index=False)

    df_face_emotion = pd.DataFrame(run_parallel_analysis(img_paths, analyze_faces_and_emotions, "Face/Emotion"))
    df_face_emotion.to_csv(os.path.join(OUTPUT_DIR, 'face_emotion.csv'), index=False)

    df_text = pd.DataFrame(run_parallel_analysis(img_paths, analyze_text_all, "Text"))
    df_text.to_csv(os.path.join(OUTPUT_DIR, 'text.csv'), index=False)

    # Element Complexity
    element_complexity_results = []
    for i, row in tqdm(df_object_concreteness.iterrows(), total=df_object_concreteness.shape[0], desc="Element Complexity"):
        image_path = row['image_path']
        yolo_objects = row.get('yolo_objects', None)
        text_row = df_text[df_text['image_path'] == image_path]
        easyocr_text = text_row['easyocr_text'].values[0] if not text_row.empty else ""
        element_complexity_results.append(
            analyze_element_complexity(yolo_objects, easyocr_text, image_path)
        )
    df_element_complexity = pd.DataFrame(element_complexity_results)
    df_element_complexity.to_csv(os.path.join(OUTPUT_DIR, 'element_complexity.csv'), index=False)

    print("All analyses saved separately!")
