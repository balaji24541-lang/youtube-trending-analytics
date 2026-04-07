import os
import sys
import pandas as pd
from googleapiclient.discovery import build

# Add parent directory to path so we can import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def get_trending_videos(api_key, region_code='US', max_results=200):
    print(f"Fetching trending videos for region: {region_code}")
    youtube = build('youtube', 'v3', developerKey=api_key)
    
    videos = []
    next_page_token = None
    
    while len(videos) < max_results:
        request = youtube.videos().list(
            part="snippet,contentDetails,statistics",
            chart="mostPopular",
            regionCode=region_code,
            maxResults=50,
            pageToken=next_page_token
        )
        response = request.execute()
        
        for item in response.get('items', []):
            video_data = {
                'video_id': item['id'],
                'title': item['snippet']['title'],
                'published_at': item['snippet']['publishedAt'],
                'channel_id': item['snippet']['channelId'],
                'channel_title': item['snippet']['channelTitle'],
                'category_id': item['snippet']['categoryId'],
                'tags': '|'.join(item['snippet'].get('tags', [])),
                'description': item['snippet']['description'],
                'duration': item['contentDetails']['duration'],
                'view_count': item['statistics'].get('viewCount', 0),
                'like_count': item['statistics'].get('likeCount', 0),
                'comment_count': item['statistics'].get('commentCount', 0)
            }
            videos.append(video_data)
        
        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break
            
    print(f"Total videos fetched: {len(videos)}")
    return pd.DataFrame(videos)

if __name__ == "__main__":
    API_KEY = config.API_KEY
    if not API_KEY or API_KEY == "YOUR_API_KEY_HERE":
        print("Please set your YouTube API Key in config.py")
        sys.exit(1)
        
    df = get_trending_videos(API_KEY, region_code='US', max_results=200)
    
    # Save the data
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'trending_videos_us.csv')
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")
