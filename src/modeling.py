import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import os

os.makedirs('outputs', exist_ok=True)
sns.set_theme(style="whitegrid", context="talk")

def get_sentiment(text):
    if pd.isna(text):
        return 0.0
    return TextBlob(str(text)).sentiment.polarity

def run_modeling():
    print("Loading data for modeling...")
    df = pd.read_csv('data/trending_videos_us.csv')
    
    print("Extracting NLP and advanced features...")
    # Clean data first
    df = df[df['view_count'] > 0]
    
    # 1. Title sentiment
    df['title_sentiment'] = df['title'].apply(get_sentiment)
    
    # 2. Tag sentiment
    df['tags_sentiment'] = df['tags'].apply(get_sentiment)
    
    # Target variable: Engagement Ratio
    df['engagement_ratio'] = (df['like_count'] + df['comment_count']) / df['view_count']
    
    # Other features
    df['title_length'] = df['title'].apply(lambda x: len(str(x)))
    df['tags_count'] = df['tags'].apply(lambda x: len(str(x).split('|')) if pd.notna(x) else 0)
    
    # Prep for modeling
    features = ['title_length', 'tags_count', 'title_sentiment', 'tags_sentiment']
    X = df[features].fillna(0)
    y = df['engagement_ratio'].fillna(0)
    
    # Train Random Forest
    print("Training Random Forest to predict engagement ratio...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Feature Importance
    importances = rf.feature_importances_
    feature_names = features
    
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', hue='Feature', data=importance_df, palette='magma')
    plt.title('Feature Importance for Predicting Engagement Ratio')
    plt.tight_layout()
    plt.savefig('outputs/feature_importance.png', dpi=300)
    plt.close()

    print("Modeling phase completed. Insights saved in outputs/feature_importance.png.")

if __name__ == "__main__":
    run_modeling()
