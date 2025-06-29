# YouTube Video Metadata and Thumbnail Analysis Pipeline

This repository contains two main Python scripts for collecting YouTube video metadata and analyzing video thumbnails. The scripts are designed for research and data analysis workflows, especially for studies involving YouTube content and visual analytics.

---

## 1. Retrieve-Channel-Videos-2025.py

**Purpose**  
Retrieves metadata for videos from specified YouTube channels, including video titles, descriptions, statistics, and thumbnails. Designed to collect data for further analysis and supports filtering by video duration and publication date[1].

### Features

- **Channel Selection:** Select channels by row range or specific indices from a CSV file.
- **Video Filtering:** Retrieves only medium and long videos (skips shorts) published between 2016 and 2021.
- **Metadata Collection:** Gathers video titles, descriptions, tags, statistics (views, likes, comments), and thumbnails.
- **Channel Metadata:** Collects channel names, subscriber counts, and country information.
- **Thumbnail Download:** Downloads high-quality thumbnails for each video.
- **Output:** Saves all metadata and thumbnails to specified directories.

### Requirements

- **Python 3.x**
- **Libraries:**  
  - `googleapiclient` (YouTube Data API v3)
  - `pandas`
  - `requests`
  - `os`, `re`, `time`
- **YouTube Data API Key:** Required for API access.

### Usage

1. **Prepare CSV:**  
   - The script expects a CSV file with YouTube channel URLs and channel names.
2. **Configure:**  
   - Set the `API_KEY`, `CSV_PATH`, `OUTPUT_DIR`, and `THUMBNAIL_DIR` in the script.
3. **Run:**  
   - Execute the script.
   - Select channels by entering a row range (e.g., `0-9`) or specific indices (e.g., `0,5,10`).
   - Specify the maximum number of videos per channel (default: 500).
4. **Output:**  
   - The script saves metadata and thumbnails in the specified directories.

---

## 2. FINAL-THUMBNAIL-ANALYSIS-18062025.py

**Purpose**  
Performs comprehensive visual and textual analysis on YouTube video thumbnails. Extracts features such as color, brightness, saliency, object detection, face/emotion analysis, and text/sentiment analysis[2].

### Features

- **Color and Brightness Analysis:** Extracts dominant colors, color palettes, brightness, and colorfulness metrics using multiple methods.
- **Composition Analysis:** Computes saliency maps and checks for the rule of thirds.
- **Visual Complexity:** Calculates edge counts and entropy for complexity assessment.
- **Object Detection:** Uses YOLO to detect and count objects in thumbnails.
- **Face and Emotion Analysis:** Uses DeepFace to detect faces, estimate age, gender, and dominant emotion.
- **Text and Sentiment Analysis:** Extracts text using EasyOCR and analyzes sentiment with VADER.
- **Element Complexity:** Combines object and text counts for a complexity score.
- **Parallel Processing:** Analyzes thumbnails in parallel for efficiency.
- **Logging:** Logs errors and warnings for troubleshooting.

### Requirements

- **Python 3.x**
- **Libraries:**  
  - `pandas`, `numpy`
  - `cv2` (OpenCV)
  - `PIL` (Pillow)
  - `deepface`
  - `colorthief`
  - `ultralytics` (YOLO)
  - `easyocr`
  - `vaderSentiment`
  - `skimage` (scikit-image)
  - `extcolors`
  - `Pylette`
  - `concurrent.futures`
  - `tqdm`
  - `logging`
- **Model Files:**  
  - YOLOv8 model (`yolov8x.pt`)
  - EasyOCR language models

### Usage

1. **Prepare Thumbnails:**  
   - Place all thumbnails in the directory specified by `THUMBNAILS_DIR`.
2. **Configure:**  
   - Set the `THUMBNAILS_DIR` and `OUTPUT_DIR` in the script.
3. **Run:**  
   - Execute the script.
   - The script analyzes all thumbnails and saves results in separate CSV files for each analysis type.
4. **Output:**  
   - CSV files for color/brightness, composition, visual complexity, object/concreteness, face/emotion, text, and element complexity are saved in the output directory.

