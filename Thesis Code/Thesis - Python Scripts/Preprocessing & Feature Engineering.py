import os
import pandas as pd
import numpy as np
import ast
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ======================
# CONFIGURATION
# ======================
BASE_PATH = r'C:\Your\Project\Path'
INPUT_FOLDER = os.path.join(BASE_PATH)
OUTPUT_FOLDER = os.path.join(BASE_PATH)

# ======================
# CONCATENATE METADATA CSVs
# ======================
def concatenate_metadata():
    metadata_dir = os.path.join(BASE_PATH, 'METADATA')
    output_path = os.path.join(BASE_PATH, 'metadata_rows.csv')

    if not os.path.exists(metadata_dir):
        raise FileNotFoundError(f"Directory not found: {metadata_dir}")

    csv_files = [f for f in os.listdir(metadata_dir) if f.endswith('.csv')]
    if not csv_files:
        print("Warning: No CSV files found in METADATA directory")
        return

    df_list = []
    for f in csv_files:
        file_path = os.path.join(metadata_dir, f)
        try:
            df = pd.read_csv(file_path)
            df_list.append(df)
        except Exception as e:
            print(f"Could not read {file_path}: {e}")

    if df_list:
        concatenated_df = pd.concat(df_list, ignore_index=True)
        concatenated_df.to_csv(output_path, index=False)
        print(f"Saved concatenated metadata to: {output_path} ({len(concatenated_df)} rows)")
    else:
        print("No valid CSVs to concatenate.")

# Run the concatenation step before anything else
concatenate_metadata()

# ======================
# INPUT FILES
# ======================
input_files = {
    "visual_complexity": os.path.join(INPUT_FOLDER, 'visual_complexity.csv'),
    "object_concreteness": os.path.join(INPUT_FOLDER, 'object_concreteness.csv'),
    "face_emotion": os.path.join(INPUT_FOLDER, 'face_emotion.csv'),
    "color_brightness": os.path.join(INPUT_FOLDER, 'color_brightness.csv'),
    "text": os.path.join(INPUT_FOLDER, 'text.csv'),
    "merged": os.path.join(BASE_PATH, 'metadata_rows.csv')  # Use the new concatenated file
}

# Output configuration
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
INTERMEDIATE_OUTPUT = os.path.join(OUTPUT_FOLDER, 'YouTube_Productivity.csv')

# ======================
# HELPER FUNCTIONS
# ======================
def safe_join(text_list):
    """Safely join text lists handling various formats"""
    if pd.isna(text_list) or text_list == "":
        return ""
    if isinstance(text_list, str):
        try:
            text_list = ast.literal_eval(text_list)
        except (ValueError, SyntaxError):
            return text_list
    return ' '.join(text_list) if isinstance(text_list, list) else str(text_list)

def iso8601_duration_to_seconds(duration):
    """Convert ISO 8601 duration to seconds"""
    match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', duration)
    if not match:
        return 0
    hours = int(match.group(1)) if match.group(1) else 0
    minutes = int(match.group(2)) if match.group(2) else 0
    seconds = int(match.group(3)) if match.group(3) else 0
    return hours * 3600 + minutes * 60 + seconds

# ======================
# DATA PROCESSING
# ======================
df_text = pd.read_csv(input_files["text"])
df_vis = pd.read_csv(input_files["visual_complexity"])
df_object_concreteness = pd.read_csv(input_files["object_concreteness"])
df_face_emotion = pd.read_csv(input_files["face_emotion"])
df_color_brightness = pd.read_csv(input_files["color_brightness"])
df_text = pd.read_csv(input_files["text"])

# Handle object concreteness data
df_object_concreteness['yolo_objects'] = df_object_concreteness['yolo_objects'].apply(
    lambda x: [] if pd.isna(x) else ast.literal_eval(x) if isinstance(x, str) else x
)
df_object_concreteness['object_count_yolo'] = df_object_concreteness['object_count_yolo'].fillna(0)

# Text processing
analyzer = SentimentIntensityAnalyzer()
df_text['easyocr_text_list'] = df_text['easyocr_text'].apply(safe_join)
df_text['text_sentiment_score'] = df_text['easyocr_text_list'].apply(
    lambda x: analyzer.polarity_scores(x)['compound'] if x else 0.0
)
df_text['text_presence'] = df_text['easyocr_text_list'].apply(lambda x: 1 if x else 0)
df_text['num_text'] = df_text['easyocr_text_list'].apply(len)

# Face present
df_face_emotion['face_present'] = df_face_emotion['face_count'].apply(lambda x: 1 if x > 0 else 0)

# Process merged data
df_merged = pd.read_csv(input_files["merged"]).drop('description', axis=1)
df_merged['title_len'] = df_merged['title'].apply(len)
df_merged['title_sentiment'] = df_merged['title'].apply(
    lambda x: analyzer.polarity_scores(x)['compound']
)
df_merged['tags_list'] = df_merged['tags'].apply(
    lambda x: ast.literal_eval(x) if pd.notnull(x) and x.strip() else []
)
df_merged['tags_presence'] = df_merged['tags_list'].apply(bool)
df_merged['tags_count'] = df_merged['tags_list'].apply(len)
df_merged['published_at'] = pd.to_datetime(df_merged['published_at']).dt.strftime('%Y-%m-%d %H:%M:%S')
df_merged['duration'] = df_merged['duration'].apply(iso8601_duration_to_seconds)

# ======================
# FEATURE MERGING
# ======================
feature_dfs = [
    df_vis.set_index('image_path'),
    df_face_emotion.set_index('image_path'),
    df_color_brightness.set_index('image_path'),
    df_text.set_index('image_path'),
    df_object_concreteness.set_index('image_path')
]

final_features = pd.concat(feature_dfs, axis=1)
final_features = final_features.loc[:, ~final_features.columns.duplicated()].reset_index()
final_features['video_id'] = final_features['image_path'].apply(
    lambda x: x.split('\\')[-1].split('.')[0] if pd.notnull(x) else None
)

merged_df = pd.merge(
    df_merged,
    final_features,
    on='video_id',
    how='inner'
)

# ======================
# COLUMN FILTERING
# ======================
COLUMNS_TO_KEEP = [
    'video_id', 'title', 'tags', 'published_at', 'duration', 'view_count',
    'like_count', 'comment_count', 'thumbnail_url', 'channel_name',
    'channel_subscribers', 'channel_country', 'title_len', 'tags_list',
    'tags_presence', 'tags_count', 'image_path', 'edge_count',
    'entropy_mean', 'age', 'gender', 'dominant_emotion', 'face_count',
    'angry', 'fear', 'neutral', 'sad', 'disgust', 'happy', 'surprise',
    'face_present', 'brightness_pil', 'colorfulness_pylette',
    'colorfulness_cv', 'brightness_cv', 'dominant_color_r',
    'dominant_color_g', 'dominant_color_b', 'easyocr_text_list',
    'text_presence', 'num_text', 'sentiment_neg', 'sentiment_pos',
    'sentiment_neu', 'yolo_objects', 'object_count_yolo', 'sentiment_score',
    'title_sentiment_score', 'text_sentiment_score'
]

existing_columns = [col for col in COLUMNS_TO_KEEP if col in merged_df.columns]
filtered_df = merged_df[existing_columns]

# ======================
# FINAL OUTPUT
# ======================
merged_df.to_csv(INTERMEDIATE_OUTPUT, index=False)
print(f"Intermediate merged data saved to: {INTERMEDIATE_OUTPUT}")

print(f"Original dimensions: {merged_df.shape} → Filtered dimensions: {filtered_df.shape}")
print(f"Columns retained: {len(existing_columns)}/{len(COLUMNS_TO_KEEP)}")
if len(existing_columns) < len(COLUMNS_TO_KEEP):
    missing = set(COLUMNS_TO_KEEP) - set(existing_columns)
    print(f"Warning: Missing columns: {missing}")
