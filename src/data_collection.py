import os
import sys
import pandas as pd
from typing import Dict, List, Any
from googleapiclient.discovery import build
import datetime

# Add parent directory to path so we can import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import config
except ImportError:
    print("Warning: config.py not found. Please create one with your API_KEY.")
    sys.exit(1)

def get_youtube_service(api_key: str) -> Any:
    """
    Initializes and returns the YouTube API service object.
    
    Args:
        api_key (str): YouTube Data API v3 Key.
        
    Returns:
        googleapiclient.discovery.Resource: The YouTube API service instance.
    """
    return build('youtube', 'v3', developerKey=api_key)

def get_category_mapping(youtube: Any, region_code: str = 'US') -> Dict[str, str]:
    """
    Fetches the mapping of YouTube Category IDs to their human-readable string names.
    
    Args:
        youtube: YouTube API service instance.
        region_code (str): The country code for categories (default 'US').
        
    Returns:
        Dict[str, str]: A dictionary mapping category ID strings to category names.
    """
    print("Fetching YouTube Category mapping...")
    request = youtube.videoCategories().list(
        part="snippet",
        regionCode=region_code
    )
    response = request.execute()
    
    category_map = {}
    for item in response.get('items', []):
        category_map[item['id']] = item['snippet']['title']
        
    return category_map

def get_trending_videos(youtube: Any, category_map: Dict[str, str], region_code: str = 'US', max_results: int = 200) -> pd.DataFrame:
    """
    Fetches the current trending videos from YouTube and structures them into a DataFrame.
    
    Args:
        youtube: YouTube API service instance.
        category_map (Dict[str, str]): Mapping from category_id to category_name.
        region_code (str): Region code to get trends for.
        max_results (int): Maximum number of videos to fetch.
        
    Returns:
        pd.DataFrame: A DataFrame containing the trending videos data.
    """
    print(f"Fetching trending videos for region: {region_code}")
    
    videos: List[Dict[str, Any]] = []
    next_page_token = None
    
    while len(videos) < max_results:
        try:
            request = youtube.videos().list(
                part="snippet,contentDetails,statistics",
                chart="mostPopular",
                regionCode=region_code,
                maxResults=50,
                pageToken=next_page_token
            )
            response = request.execute()
            
            for item in response.get('items', []):
                snippet = item.get('snippet', {})
                statistics = item.get('statistics', {})
                content_details = item.get('contentDetails', {})
                
                cat_id = snippet.get('categoryId', '')
                
                video_data = {
                    'video_id': item.get('id'),
                    'title': snippet.get('title'),
                    'published_at': snippet.get('publishedAt'),
                    'channel_id': snippet.get('channelId'),
                    'channel_title': snippet.get('channelTitle'),
                    'category_id': cat_id,
                    'category_name': category_map.get(cat_id, 'Unknown'),
                    'tags': '|'.join(snippet.get('tags', [])),
                    'description': snippet.get('description'),
                    'duration': content_details.get('duration'),
                    'view_count': int(statistics.get('viewCount', 0)),
                    'like_count': int(statistics.get('likeCount', 0)),
                    'comment_count': int(statistics.get('commentCount', 0)),
                    'trending_date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                videos.append(video_data)
            
            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break
                
        except Exception as e:
            print(f"An error occurred during API fetch: {e}")
            break
            
    # Ensure we don't return more than requested
    videos = videos[:max_results]
    print(f"Total videos fetched: {len(videos)}")
    
    return pd.DataFrame(videos)

def main():
    """Main execution function for data collection."""
    API_KEY = getattr(config, 'API_KEY', None)
    if not API_KEY or API_KEY == "YOUR_API_KEY_HERE":
        print("Please set your YouTube API Key in config.py")
        sys.exit(1)
        
    youtube_service = get_youtube_service(API_KEY)
    
    # Get mapping and then the data
    category_map = get_category_mapping(youtube_service, region_code='US')
    df = get_trending_videos(youtube_service, category_map, region_code='US', max_results=200)
    
    # Save the data
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, 'data')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'trending_videos_us.csv')
    
    df.to_csv(output_path, index=False)
    print(f"Data successfully saved to {output_path}")

if __name__ == "__main__":
    main()
