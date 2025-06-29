# YouTube Video Metadata, Thumbnail Analysis, and Panel Modeling Pipeline

This repository provides a full workflow for collecting YouTube video metadata, analyzing video thumbnails, and performing advanced statistical modeling on the results. The pipeline is designed for research and data analysis projects involving YouTube content, visual analytics, and causal inference.

---

## 1. Python Scripts: Data Collection & Thumbnail Analysis

### Retrieve-Channel-Videos-2025.py

**Purpose:**  
Retrieves metadata for videos from specified YouTube channels, including video titles, descriptions, statistics, and thumbnails.

**Features:**
- **Channel Selection:** Select channels by row range or specific indices from a CSV file.
- **Video Filtering:** Retrieves only medium and long videos (skips shorts) published between 2016 and 2021.
- **Metadata Collection:** Gathers video titles, descriptions, tags, statistics (views, likes, comments), and thumbnails.
- **Channel Metadata:** Collects channel names, subscriber counts, and country information.
- **Thumbnail Download:** Downloads high-quality thumbnails for each video.
- **Output:** Saves all metadata and thumbnails to specified directories.

**Requirements:**
- Python 3.x
- Libraries: `googleapiclient`, `pandas`, `requests`, `os`, `re`, `time`
- YouTube Data API Key

**Usage:**
1. Prepare a CSV file with YouTube channel URLs and channel names.
2. Set the `API_KEY`, `CSV_PATH`, `OUTPUT_DIR`, and `THUMBNAIL_DIR` in the script.
3. Run the script and select channels by row range or index.
4. Output: Metadata and thumbnails saved to the specified directories.

---

### FINAL-THUMBNAIL-ANALYSIS-18062025.py

**Purpose:**  
Performs comprehensive visual and textual analysis on YouTube video thumbnails.

**Features:**
- **Color and Brightness Analysis:** Extracts dominant colors, color palettes, brightness, and colorfulness metrics.
- **Composition Analysis:** Computes saliency maps and checks for the rule of thirds.
- **Visual Complexity:** Calculates edge counts and entropy.
- **Object Detection:** Uses YOLO to detect and count objects.
- **Face and Emotion Analysis:** Uses DeepFace for face, age, gender, and emotion detection.
- **Text and Sentiment Analysis:** Extracts text using EasyOCR and analyzes sentiment with VADER.
- **Element Complexity:** Combines object and text counts for a complexity score.
- **Parallel Processing:** Efficient analysis of large thumbnail sets.
- **Logging:** Logs errors and warnings.

**Requirements:**
- Python 3.x
- Libraries: `pandas`, `numpy`, `cv2`, `PIL`, `deepface`, `colorthief`, `ultralytics`, `easyocr`, `vaderSentiment`, `skimage`, `extcolors`, `Pylette`, `concurrent.futures`, `tqdm`, `logging`
- Model Files: YOLOv8 model (`yolov8x.pt`), EasyOCR language models

**Usage:**
1. Place thumbnails in the directory specified by `THUMBNAILS_DIR`.
2. Set `THUMBNAILS_DIR` and `OUTPUT_DIR` in the script.
3. Run the script; results are saved as CSVs in the output directory.

---

## 2. R Scripts: Panel Analysis of Thumbnail Features

These R scripts analyze the effects of thumbnail features (face and text presence) on video performance (views, likes) using advanced count data models. The workflow includes propensity score matching, Poisson Pseudo-Maximum Likelihood (PPML), and Negative Binomial Regression (NBR), with and without channel and month fixed effects.

### File Overview

| Script Name                                    | Model(s)    | Fixed Effects        | Description                                       |
|------------------------------------------------|-------------|----------------------|---------------------------------------------------|
| Thesis-Overdispersion-PPML-NBR-Fixed-Effects.R | PPML, NBR   | Channel, Month       | Full workflow: matching, modeling, visualization  |
| Thesis-Overdispersion-NBR-Fixed-Effects.R      | NBR         | Channel, Month       | NBR with/without panel regression, time trends    |
| Thesis-Overdispersion-PPML-Fixed-Effects.R     | PPML        | Channel, Month       | PPML with matching, fixed effects, time trends    |
| Thesis-Overdispersion-NBR-No-Fixed-Effects.R   | NBR         | None                 | NBR without fixed effects, matching, visualization|
| Thesis-Overdispersion-PPML-No-Fixed-Effects.R  | PPML        | None                 | PPML without fixed effects, matching, visualization|

---

### Data Requirements

All scripts expect a CSV file with columns such as:
- `face_present`, `text_presence`, `channel_name`, `published_at`
- `view_count`, `like_count`, `title_sentiment_score`, `text_sentiment_score`
- `duration`, `title_len`, `tags_presence`, `colorfulness_pylette`, `brightness_pil`, `edge_count`, `entropy_mean`, `object_count_yolo`, `channel_subscribers`

**Update the `file_path` variable in each script to match your data location.**

---

### Workflow Summary

**1. Load Required Libraries**
- `tidyverse`, `MatchIt`, `fixest`, `cobalt`, `nnet`, `broom`, `ggeffects`, `ggplot2`, `patchwork` (for some scripts)

**2. Data Preparation**
- Read CSV, convert types, create `face_text_combo` (neither, face_only, text_only, both)

**3. Overdispersion Diagnostics**
- Print mean, variance, and variance-to-mean ratio for count outcomes

**4. Propensity Score Matching**
- Matching for `face_presence`, `text_presence`, and `face_text_combo` using covariates

**5. Model Estimation**
- PPML and NBR models, with and without fixed effects for channel and month

**6. Output and Visualization**
- Coefficient tables (CSV), forest plots, predicted value plots, residuals, temporal trends, fixed effects plots

---

### Script Selection Guide

| Analysis Goal                                  | Use Script(s)                                     |
|------------------------------------------------|---------------------------------------------------|
| Full workflow (matching, PPML & NBR, FE, plots)| Thesis-Overdispersion-PPML-NBR-Fixed-Effects.R    |
| Only Negative Binomial, with FE                | Thesis-Overdispersion-NBR-Fixed-Effects.R         |
| Only PPML, with FE                             | Thesis-Overdispersion-PPML-Fixed-Effects.R        |
| Only Negative Binomial, no FE                  | Thesis-Overdispersion-NBR-No-Fixed-Effects.R      |
| Only PPML, no FE                               | Thesis-Overdispersion-PPML-No-Fixed-Effects.R     |

---

### How to Run

1. **Edit the data path** in each script to point to your CSV file.
2. **Open the script in RStudio** or another R environment.
3. **Run the script** section by section, or source the entire script.
4. **Check output files** (plots, CSVs) in your working directory.

---

### Key Functions and Outputs

- `check_overdisp()`: Prints mean/variance diagnostics
- `matchit()`: Propensity score matching
- `fepois()` / `fenegbin()`: PPML/NBR models (with/without fixed effects)
- `ggpredict()`: Marginal predictions for plotting
- `ggplot2`: Visualizations

---

## Notes

- All scripts assume the presence of the same covariates and outcome variables.
- Fixed effects models are recommended for causal inference when unobserved heterogeneity is suspected.
- Scripts are modular and can be adapted for additional treatments or outcomes.
