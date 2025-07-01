import os
import re
import requests
import pandas as pd
from googleapiclient.discovery import build
from time import sleep



# ========== USER CONFIGURATION ==========
API_KEY = "<YOUR_YOUTUBE_API_KEY_HERE>"
CSV_PATH = r"<PATH_TO_YOUR_CHANNELS_CSV>"
OUTPUT_DIR = r"<OUTPUT_DIRECTORY_PATH>"
THUMBNAIL_DIR = os.path.join(OUTPUT_DIR, "thumbnails")

os.makedirs(THUMBNAIL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== SETUP YOUTUBE API ==========
youtube = build("youtube", "v3", developerKey=API_KEY)

# ========== READ CHANNEL IDs FROM CSV ==========
df_channels = pd.read_csv(CSV_PATH)
df_channels['channel_id'] = df_channels['URL'].str.extract(r'/channel/([a-zA-Z0-9_-]+)')

print("Available channels in CSV:")
for idx, row in df_channels.iterrows():
    print(f"{idx}: {row['CHANNEL']} ({row['URL']})")

print("\nHow do you want to select channels?")
print(" - Enter a row range (e.g. 0-9 for first 10 rows)")
print(" - Or a comma-separated list of indices (e.g. 0,5,10)")
selection = input("Enter your selection: ").strip()

if '-' in selection:
    start, end = map(int, selection.split('-'))
    selected_df = df_channels.iloc[start:end+1]
elif ',' in selection:
    indices = [int(i) for i in selection.split(',')]
    selected_df = df_channels.iloc[indices]
else:
    try:
        idx = int(selection)
        selected_df = df_channels.iloc[[idx]]
    except Exception:
        print("Invalid selection. Exiting.")
        exit()

channel_ids = selected_df['channel_id'].dropna().unique().tolist()

max_videos_per_channel = int(input("Max videos per channel (default 500): ") or 500)

def batched(iterable, n):
    l = list(iterable)
    for i in range(0, len(l), n):
        yield l[i:i+n]

# ========== MAIN SCRIPT ==========
all_metadata = []
all_channel_ids = set()

for channel_id in channel_ids:
    print(f"\nProcessing channel ID: {channel_id}")
    video_ids = []
    # Only fetch medium and long videos (skip shorts) from 2022 to June 2025
    for duration in ["medium", "long"]:
        next_page_token = None
        while len(video_ids) < max_videos_per_channel:
            search_params = {
                "part": "id",
                "channelId": channel_id,
                "type": "video",
                "videoDuration": duration,
                "publishedAfter": "2022-01-01T00:00:00Z",  # Start of 2022
                "publishedBefore": "2025-07-01T00:00:00Z", # Start of July 2025 
                "maxResults": min(50, max_videos_per_channel - len(video_ids)),
                "order": "date"
            }
            if next_page_token:
                search_params["pageToken"] = next_page_token

            search_response = youtube.search().list(**search_params).execute()
            video_ids.extend(
                item['id']['videoId']
                for item in search_response['items']
                if item['id']['kind'] == 'youtube#video'
            )
            next_page_token = search_response.get('nextPageToken')
            if not next_page_token or len(video_ids) >= max_videos_per_channel:
                break
    # Remove duplicates and limit to max_videos_per_channel
    video_ids = list(dict.fromkeys(video_ids))[:max_videos_per_channel]
    print(f"  Found {len(video_ids)} medium/long videos (2022-June 2025).")

    # Batch get video metadata
    metadata = []
    for batch in batched(video_ids, 50):
        batch = [vid for vid in batch if vid]
        if not batch:
            continue
        video_response = youtube.videos().list(
            part="snippet,contentDetails,statistics,status",
            id=",".join(batch)
        ).execute()
        for item in video_response['items']:
            metadata.append({
                "video_id": item['id'],
                "channel_id": item['snippet']['channelId'],
                "title": item['snippet']['title'],
                "description": item['snippet']['description'],
                "tags": item['snippet'].get('tags', []),
                "published_at": item['snippet']['publishedAt'],
                "duration": item['contentDetails']['duration'],
                "view_count": int(item['statistics'].get('viewCount', 0)),
                "like_count": int(item['statistics'].get('likeCount', 0)),
                "comment_count": int(item['statistics'].get('commentCount', 0)),
                "thumbnail_url": item['snippet']['thumbnails']['high']['url']
            })
            all_channel_ids.add(item['snippet']['channelId'])
        sleep(0.1)  # API rate limit protection

    all_metadata.extend(metadata)

# ========== BATCH GET CHANNEL METADATA ==========
channel_cache = {}
for batch in batched(all_channel_ids, 50):
    batch = [cid for cid in batch if cid]
    if not batch:
        continue
    channel_response = youtube.channels().list(
        part="snippet,statistics",
        id=",".join(batch)
    ).execute()
    for channel in channel_response['items']:
        channel_cache[channel['id']] = {
            "name": channel['snippet']['title'],
            "subscribers": int(channel['statistics'].get('subscriberCount', 0)),
            "country": channel['snippet'].get('country', None)
        }
    sleep(0.1)

# ========== MERGE & DOWNLOAD THUMBNAILS ==========
final_data = []
for video in all_metadata:
    channel_id = video.pop('channel_id')
    final_data.append({
        **video,
        "channel_name": channel_cache.get(channel_id, {}).get('name', 'Unknown'),
        "channel_subscribers": channel_cache.get(channel_id, {}).get('subscribers', 0),
        "channel_country": channel_cache.get(channel_id, {}).get('country', None)
    })
    # Download thumbnail
    vid = video['video_id']
    thumb_path = os.path.join(THUMBNAIL_DIR, f"{vid}.jpg")
    if not os.path.exists(thumb_path):
        try:
            response = requests.get(video['thumbnail_url'])
            if response.status_code == 200:
                with open(thumb_path, "wb") as f:
                    f.write(response.content)
        except Exception as e:
            print(f"Thumbnail error for {vid}: {str(e)}")

# ========== SAVE RESULTS ==========
selection_str = selection.replace(',', '_').replace('-', '_')
selection_str = re.sub(r'\W+', '_', selection_str)

output_filename = f"metadata_rows_{selection_str}_2022-2025_June.csv"
df = pd.DataFrame(final_data)
df.to_csv(os.path.join(OUTPUT_DIR, output_filename), index=False)
print(f"\nSaved metadata and thumbnails in '{OUTPUT_DIR}' as '{output_filename}'")
print("Processing complete!")
