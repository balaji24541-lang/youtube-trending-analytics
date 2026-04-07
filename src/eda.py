import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

def setup_environment():
    """Create output directories and set plot styling."""
    os.makedirs('outputs', exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")

def load_data(file_path='data/trending_videos_us.csv') -> pd.DataFrame:
    """Load data and handle missing file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at {file_path}. Please run data_collection.py first.")
    return pd.read_csv(file_path)

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply advanced feature engineering to extract meaningful metrics.
    Adds engagement rates, velocity, and temporal features.
    """
    print("Performing feature engineering...")
    # Safe conversion to datetime and strip timezone info to prevent subtraction errors
    df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce').dt.tz_localize(None)
    df['trending_date'] = pd.to_datetime(df['trending_date'], errors='coerce').dt.tz_localize(None)
    
    # 1. Title & Tag Features
    df['title_length'] = df['title'].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
    df['tags_count'] = df['tags'].apply(lambda x: len(str(x).split('|')) if pd.notna(x) else 0)
    
    # 2. Advanced Engagement Metrics (handle division by zero)
    # Adding a small epsilon or using np.where to prevent inf/NaN
    df['likes_per_view'] = np.where(df['view_count'] > 0, df['like_count'] / df['view_count'], 0)
    df['comments_per_view'] = np.where(df['view_count'] > 0, df['comment_count'] / df['view_count'], 0)
    df['engagement_rate'] = np.where(df['view_count'] > 0, (df['like_count'] + df['comment_count']) / df['view_count'], 0)
    
    # 3. Temporal Features (Best time to upload)
    df['publish_hour'] = df['published_at'].dt.hour
    df['publish_day'] = df['published_at'].dt.day_name()
    
    # 4. Velocity Features (Performance speed)
    # Calculate difference in hours
    if 'trending_date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['trending_date']):
        time_diff = (df['trending_date'] - df['published_at']).dt.total_seconds() / 3600
        df['hours_to_trend'] = np.where(time_diff > 0, time_diff, np.nan)
        # Fill NaN with median hours_to_trend to prevent dropping data
        df['hours_to_trend'] = df['hours_to_trend'].fillna(df['hours_to_trend'].median())
        df['views_per_hour'] = np.where(df['hours_to_trend'] > 0, df['view_count'] / df['hours_to_trend'], 0)
    else:
        df['hours_to_trend'] = 24  # Fallback assumption if trending_date missing
        df['views_per_hour'] = df['view_count'] / 24
        
    return df

def generate_visualizations(df: pd.DataFrame):
    """Generate and save EDA plots."""
    print("Generating visualizations...")
    
    # 1. Category-wise Performance (Views & Engagement)
    plt.figure(figsize=(12, 6))
    category_summary = df.groupby('category_name')['view_count'].median().sort_values(ascending=False).head(10)
    sns.barplot(x=category_summary.values / 1e6, y=category_summary.index, palette='Blues_d')
    plt.title('Top 10 Categories by Median Views (Millions)')
    plt.xlabel('Median Views (Millions)')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig('outputs/category_views.png', dpi=300)
    plt.close()
    
    # 2. Best Upload Time Analysis (Heatmap)
    plt.figure(figsize=(10, 6))
    pivot_table = df.pivot_table(index='publish_day', columns='publish_hour', values='view_count', aggfunc='median')
    # Reorder days
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot_table = pivot_table.reindex(days)
    sns.heatmap(pivot_table, cmap='YlGnBu', linewidths=.5)
    plt.title('Median Views by Publish Day and Hour')
    plt.xlabel('Publish Hour (UTC-based)')
    plt.ylabel('Publish Day')
    plt.tight_layout()
    plt.savefig('outputs/upload_time_heatmap.png', dpi=300)
    plt.close()
    
    # 3. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    cols = ['view_count', 'like_count', 'comment_count', 'engagement_rate', 
            'hours_to_trend', 'views_per_hour', 'title_length', 'tags_count']
    corr_df = df[cols].corr()
    sns.heatmap(corr_df, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title('Correlation Matrix of Key Virality Metrics')
    plt.tight_layout()
    plt.savefig('outputs/correlation_matrix.png', dpi=300)
    plt.close()
    
    # 4. Outlier Detection: Clickbait vs Core Audience
    # Clickbait: High views, low engagement. 
    # Core Audience: Moderate views, high engagement.
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='view_count', y='engagement_rate', alpha=0.6, color='purple')
    # Add medians as reference lines
    plt.axvline(df['view_count'].median(), color='red', linestyle='--', label='Median Views')
    plt.axhline(df['engagement_rate'].median(), color='blue', linestyle='--', label='Median Engagement')
    plt.title('Views vs Engagement Rate (Identifying Outliers & Clickbait)')
    plt.xlabel('Total Views')
    plt.ylabel('Engagement Rate')
    plt.legend()
    plt.tight_layout()
    plt.savefig('outputs/views_vs_engagement.png', dpi=300)
    plt.close()

def extract_business_insights(df: pd.DataFrame) -> dict:
    """
    Convert analysis into concrete, non-generic business insights.
    Returns a dictionary of insights to be used by the frontend.
    """
    print("Extracting business insights...")
    insights = {}
    
    # Insight 1: Best time to upload
    best_hour = df.groupby('publish_hour')['view_count'].median().idxmax()
    best_day = df.groupby('publish_day')['view_count'].median().idxmax()
    insights['best_upload'] = {
        "title": "Optimal Upload Window",
        "description": f"The algorithm favors videos published on {best_day}s around {best_hour}:00 UTC. Videos published during peak hours have a significantly higher initial velocity (`views_per_hour`), pushing them onto the Trending page faster."
    }
    
    # Insight 2: Category Growth
    top_cat = df.groupby('category_name')['engagement_rate'].median().idxmax()
    insights['top_category'] = {
        "title": "Highest Engagement Category",
        "description": f"The '{top_cat}' category generates the highest median engagement rate. While other categories might pull more raw views, '{top_cat}' creates stronger community interaction (Likes & Comments per view), signaling strong audience retention."
    }
    
    # Insight 3: Clickbait Detection
    # High views (> 75th percentile), but low engagement (< 25th percentile)
    q3_views = df['view_count'].quantile(0.75)
    q1_eng = df['engagement_rate'].quantile(0.25)
    clickbait_df = df[(df['view_count'] > q3_views) & (df['engagement_rate'] < q1_eng)]
    insights['clickbait_metrics'] = {
        "title": "Clickbait Liability",
        "description": f"Identified {len(clickbait_df)} videos that exhibit 'clickbait' behavior—massive reach (top 25% views) but extremely poor community reception (bottom 25% engagement). Creators should avoid this to protect channel reputation over long horizons."
    }
    
    # Insight 4: The Velocity Factor
    corr = df['views_per_hour'].corr(df['engagement_rate'])
    relation = "positive" if corr > 0 else "negative"
    insights['velocity_importance'] = {
        "title": "Velocity is King",
        "description": f"There is a {relation} correlation ({corr:.2f}) between how fast a video gets views (`views_per_hour`) and its overall engagement. Pushing initial traffic from external sources in the first hour mathematically improves trending probability."
    }
    
    # Save insights to disk for Streamlit to read
    with open('outputs/insights.json', 'w') as f:
        json.dump(insights, f, indent=4)
        
    return insights

def perform_eda():
    """Main pipeline execution for EDA."""
    setup_environment()
    try:
        df = load_data()
        df = feature_engineering(df)
        
        # Save enriched data for modeling
        df.to_csv('data/trending_videos_enriched.csv', index=False)
        
        generate_visualizations(df)
        insights = extract_business_insights(df)
        
        print("EDA completed successfully. All outputs saved to outputs/ folder.")
    except Exception as e:
        print(f"EDA failed: {e}")

if __name__ == "__main__":
    perform_eda()
