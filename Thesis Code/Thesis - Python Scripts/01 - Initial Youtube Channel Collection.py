import re
import pandas as pd
from googleapiclient.discovery import build

YOUTUBE_API_KEY = "<YOUR_YOUTUBE_API_KEY_HERE>"

# Official handles and user channels from HireHarbour

CHANNEL_HANDLES = [
    '@mattdavella',
    '@JeffSu',
    '@gillianzperkins',
    '@nathanieldrew',
    '@aliabdaal',
    'muchelleb',
    '@rowena',
    '@Thomasfrank',
    '@mariana-vieira',
    'Lavendaire',
]

OUTPUT_PATH = r'<YOUR_OUTPUT_PATH_HERE>/channels_output.csv'

def resolve_handle_or_user(youtube, identifier):
    try:
        if identifier.startswith('@'):
            req = youtube.channels().list(forHandle=identifier, part='id,snippet,contentDetails')
        else:
            req = youtube.channels().list(forUsername=identifier, part='id,snippet,contentDetails')
        res = req.execute()
        if 'items' in res and res['items']:
            item = res['items'][0]
            return (
                item['id'],
                item['snippet']['title'],
                item['contentDetails']['relatedPlaylists']['uploads']
            )
    except Exception as e:
        print(f"Error resolving {identifier}: {e}")
    return None, None, None

def get_all_video_ids_from_playlist(youtube, playlist_id):
    video_ids = []
    next_page_token = None
    while True:
        try:
            req = youtube.playlistItems().list(
                playlistId=playlist_id,
                part='snippet',
                maxResults=50,
                pageToken=next_page_token
            )
            res = req.execute()
            for item in res.get('items', []):
                vid = item['snippet'].get('resourceId', {}).get('videoId')
                if vid:
                    video_ids.append(vid)
            next_page_token = res.get('nextPageToken')
            if not next_page_token:
                break
        except Exception as e:
            print(f"Error paginating playlist {playlist_id}: {e}")
            break
    return video_ids

def get_video_details(youtube, video_ids):
    details = []
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i+50]
        try:
            req = youtube.videos().list(
                id=','.join(batch),
                part='snippet'
            )
            res = req.execute()
            details.extend(res.get('items', []))
        except Exception as e:
            print(f"Error fetching video details: {e}")
    return details

def extract_channel_mentions(description):
    handles = re.findall(r'@([A-Za-z0-9_\-]+)', description)
    channel_links = re.findall(r'youtube\.com/channel/([A-Za-z0-9_\-]+)', description)
    return set(handles + channel_links)

def resolve_mention(youtube, mention):
    try:
        if len(mention) == 24 and mention.startswith('UC'):
            req = youtube.channels().list(id=mention, part='id,snippet')
            res = req.execute()
            if 'items' in res and res['items']:
                item = res['items'][0]
                return item['id'], item['snippet']['title']
        else:
            handle = '@' + mention if not mention.startswith('@') else mention
            req = youtube.channels().list(forHandle=handle, part='id,snippet')
            res = req.execute()
            if 'items' in res and res['items']:
                item = res['items'][0]
                return item['id'], item['snippet']['title']
    except Exception as e:
        print(f"Error resolving mention '{mention}': {e}")
    return None, None

def main():
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    channel_data = {}
    uploads_playlists = {}

    print("Resolving initial channels and their uploads playlists...")
    for identifier in CHANNEL_HANDLES:
        cid, name, uploads_pid = resolve_handle_or_user(youtube, identifier)
        if cid and name and uploads_pid:
            channel_data[cid] = {
                'URL': f'https://www.youtube.com/channel/{cid}',
                'Channel Name': name,
                'Channel ID': cid
            }
            uploads_playlists[cid] = uploads_pid
        else:
            print(f"Could not resolve: {identifier}")

    print("Collecting all video IDs from each channel's uploads playlist (this may take a while)...")
    all_video_ids = set()
    for cid, uploads_pid in uploads_playlists.items():
        vids = get_all_video_ids_from_playlist(youtube, uploads_pid)
        all_video_ids.update(vids)

    print(f"Total unique videos collected: {len(all_video_ids)}")
    print("Fetching video details and extracting mentioned channels...")
    all_video_ids = list(all_video_ids)
    for i in range(0, len(all_video_ids), 50):
        batch = all_video_ids[i:i+50]
        videos = get_video_details(youtube, batch)
        for video in videos:
            desc = video['snippet'].get('description', '')
            mentions = extract_channel_mentions(desc)
            for mention in mentions:
                resolved_cid, resolved_name = resolve_mention(youtube, mention)
                if resolved_cid and resolved_cid not in channel_data:
                    channel_data[resolved_cid] = {
                        'URL': f'https://www.youtube.com/channel/{resolved_cid}',
                        'Channel Name': resolved_name,
                        'Channel ID': resolved_cid
                    }

    df = pd.DataFrame(channel_data.values())
    df = df[['URL', 'Channel Name', 'Channel ID']]
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"CSV saved as {OUTPUT_PATH}")

if __name__ == "__main__":
    main()