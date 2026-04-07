import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create output dir
os.makedirs('outputs', exist_ok=True)

# Set styling
sns.set_theme(style="whitegrid", context="talk")

def perform_eda():
    print("Loading data...")
    df = pd.read_csv('data/trending_videos_us.csv')
    
    # Cleaning and Feature Extraction
    print("Cleaning data and extracting features...")
    df['published_at'] = pd.to_datetime(df['published_at'])
    df['title_length'] = df['title'].apply(lambda x: len(str(x)))
    # Count tags (tags are separated by |)
    df['tags_count'] = df['tags'].apply(lambda x: len(str(x).split('|')) if pd.notna(x) else 0)
    
    # Parse duration to seconds roughly
    def parse_duration(duration_str):
        import isodate
        try:
            return isodate.parse_duration(duration_str).total_seconds()
        except:
            return 0
    # we don't have isodate module installed right now, so we will skip parsing ISO 8601 dur
            
    # Visualizations
    
    # 1. Views Distribution
    print("Generating Views Distribution...")
    plt.figure(figsize=(10, 6))
    sns.histplot(df['view_count'] / 1e6, bins=30, kde=True, color='skyblue')
    plt.title('Distribution of View Counts for Trending Videos')
    plt.xlabel('Views (Millions)')
    plt.ylabel('Number of Videos')
    plt.tight_layout()
    plt.savefig('outputs/views_distribution.png', dpi=300)
    plt.close()
    
    # 2. Correlation between metrics
    print("Generating Correlation Heatmap...")
    plt.figure(figsize=(8, 6))
    corr_df = df[['view_count', 'like_count', 'comment_count', 'title_length', 'tags_count']].corr()
    sns.heatmap(corr_df, annot=True, cmap='coolwarm', fmt=".2f", vmin=0, vmax=1)
    plt.title('Correlation Matrix of Video Metrics')
    plt.tight_layout()
    plt.savefig('outputs/correlation_matrix.png', dpi=300)
    plt.close()
    
    # 3. Top Channels by Views
    print("Generating Top Channels by Average Views...")
    top_channels = df.groupby('channel_title')['view_count'].mean().sort_values(ascending=False).head(10)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_channels.values / 1e6, y=top_channels.index, hue=top_channels.index, palette='viridis', legend=False)
    plt.title('Top 10 Channels by Average Views in Trending')
    plt.xlabel('Average Views (Millions)')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig('outputs/top_channels_views.png', dpi=300)
    plt.close()

    print("EDA completed. Charts saved in outputs/")

if __name__ == "__main__":
    perform_eda()
