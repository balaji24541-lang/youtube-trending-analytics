import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os
import json

def setup_environment():
    os.makedirs('outputs', exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")

def get_sentiment(text: str) -> float:
    """Calculate sentiment polarity from text (-1.0 to 1.0)."""
    if pd.isna(text):
        return 0.0
    return TextBlob(str(text)).sentiment.polarity

def prepare_data(df: pd.DataFrame) -> tuple:
    """Preprocess data, extract NLP features, and define the binary target."""
    print("Preparing data and extracting NLP features...")
    
    # NLP Features
    df['title_sentiment'] = df['title'].apply(get_sentiment)
    df['tags_sentiment'] = df['tags'].apply(get_sentiment)
    
    # Define Binary Target: Top 25% Engagement Rate
    threshold = df['engagement_rate'].quantile(0.75)
    df['is_highly_engaging'] = (df['engagement_rate'] >= threshold).astype(int)
    
    # Feature selection
    feature_cols = [
        'title_length', 
        'tags_count', 
        'title_sentiment', 
        'tags_sentiment',
        'hours_to_trend',
        'views_per_hour'
    ]
    
    # Handle NaNs and inf before training
    X = df[feature_cols].copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)
    
    y = df['is_highly_engaging']
    
    return X, y, feature_cols

def train_and_evaluate(X, y, feature_cols):
    """Train classification models and log evaluation metrics."""
    print("Training models...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale Data for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 1. Logistic Regression
    lr = LogisticRegression(random_state=42, class_weight='balanced')
    lr.fit(X_train_scaled, y_train)
    lr_preds = lr.predict(X_test_scaled)
    lr_probs = lr.predict_proba(X_test_scaled)[:, 1]
    
    # 2. Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_probs = rf.predict_proba(X_test)[:, 1]
    
    # Evaluation Metrics
    metrics = {
        "Random_Forest": {
            "Accuracy": accuracy_score(y_test, rf_preds),
            "Precision": precision_score(y_test, rf_preds),
            "Recall": recall_score(y_test, rf_preds),
            "F1_Score": f1_score(y_test, rf_preds),
            "ROC_AUC": roc_auc_score(y_test, rf_probs)
        },
        "Logistic_Regression": {
            "Accuracy": accuracy_score(y_test, lr_preds),
            "Precision": precision_score(y_test, lr_preds),
            "Recall": recall_score(y_test, lr_preds),
            "F1_Score": f1_score(y_test, lr_preds),
            "ROC_AUC": roc_auc_score(y_test, lr_probs)
        }
    }
    
    print("\nModel Evaluation Results:")
    for model_name, model_metrics in metrics.items():
        print(f"\n{model_name}:")
        for k, v in model_metrics.items():
            print(f"  {k}: {v:.4f}")
            
    # Save the metrics to insights for the UI
    try:
        with open('outputs/insights.json', 'r') as f:
            insights = json.load(f)
    except FileNotFoundError:
        insights = {}
        
    insights['ml_metrics'] = metrics
    
    # Feature Importance (Random Forest)
    importances = rf.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_cols, 
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    top_3 = importance_df['Feature'].head(3).tolist()
    insights['model_interpretation'] = {
        "title": "What Drives Viral Engagement?",
        "description": f"Our Random Forest model isolated the top predictive signals for High Engagement videos. The top 3 predictors are: 1) `{top_3[0]}` 2) `{top_3[1]}` 3) `{top_3[2]}`. This means that video velocity and metadata length strongly predict if an audience will interact heavily.",
        "top_features": top_3
    }
    
    with open('outputs/insights.json', 'w') as f:
        json.dump(insights, f, indent=4)
        
    # Plot Feature Importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', hue='Feature', data=importance_df, palette='magma')
    plt.title('Feature Importance for Predicting High Engagement (Random Forest)')
    plt.tight_layout()
    plt.savefig('outputs/feature_importance.png', dpi=300)
    plt.close()
    
    print("Modeling complete. Interpretations and plots saved.")

def run_modeling():
    setup_environment()
    data_path = 'data/trending_videos_enriched.csv'
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Please run eda.py first.")
        return
        
    df = pd.read_csv(data_path)
    X, y, feature_cols = prepare_data(df)
    train_and_evaluate(X, y, feature_cols)

if __name__ == "__main__":
    run_modeling()
