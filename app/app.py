import streamlit as st
import pandas as pd
import os
import json

st.set_page_config(page_title="YouTube Virality Analytics", layout="wide", page_icon="📈")

# --- Anti-Gravity CSS Animation ---
# Injects custom CSS to make main containers "fall" from the top of the screeen 
# and settle physically into place on load.
st.markdown("""
    <style>
    @keyframes dropIn {
        0% { transform: translateY(-100vh); opacity: 0; }
        60% { transform: translateY(30px); opacity: 1; }
        80% { transform: translateY(-10px); }
        100% { transform: translateY(0); }
    }
    
    .stApp > header {
        display: none;
    }
    
    /* Apply animation to all main block containers */
    .block-container {
        animation: dropIn 1.2s cubic-bezier(0.25, 0.46, 0.45, 0.94) forwards !important;
    }
    
    /* Sleek card styling */
    .metric-card {
        background-color: #1e1e2e;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ff4b4b;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 20px;
    }
    .metric-title { font-size: 1.1rem; color: #a6accd; font-weight: 600; margin-bottom: 5px;}
    .metric-value { font-size: 1.8rem; color: #ffffff; font-weight: 700; }
    
    </style>
""", unsafe_allow_html=True)

# --- Define Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'trending_videos_enriched.csv')
INSIGHTS_PATH = os.path.join(BASE_DIR, 'outputs', 'insights.json')

def load_data():
    if not os.path.exists(DATA_PATH):
        return None
    return pd.read_csv(DATA_PATH)

def load_insights():
    if not os.path.exists(INSIGHTS_PATH):
        return None
    with open(INSIGHTS_PATH, 'r') as f:
        return json.load(f)

# --- App Structure ---
st.title("🚀 What Makes a YouTube Video Go Viral?")
st.markdown("An end-to-end Machine Learning pipeline reverse engineering the YouTube algorithm to identify high-engagement indicators, optimal upload windows, and predict viral probability.")

df = load_data()
insights = load_insights()

if df is None:
    st.warning("Data not found. Please run the `data_collection.py` and `eda.py` scripts first.")
    st.stop()

# Main Tabs
tab1, tab2, tab3 = st.tabs(["💡 Business Insights", "📊 Exploratory Data Analysis", "🤖 Predict & Interpret (ML)"])

# --- TAB 1: Business Insights ---
with tab1:
    st.header("Key Virality Insights")
    
    if insights:
        col1, col2 = st.columns(2)
        
        # Optimal Upload Time
        with col1:
            if 'best_upload' in insights:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">{insights['best_upload']['title']}</div>
                    <div style="color:#d9d9d9;">{insights['best_upload']['description']}</div>
                </div>
                """, unsafe_allow_html=True)
                
        # Top Category
        with col2:
            if 'top_category' in insights:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">{insights['top_category']['title']}</div>
                    <div style="color:#d9d9d9;">{insights['top_category']['description']}</div>
                </div>
                """, unsafe_allow_html=True)
                
        col3, col4 = st.columns(2)
        # Clickbait Detection
        with col3:
            if 'clickbait_metrics' in insights:
                st.markdown(f"""
                <div class="metric-card" style="border-left-color: #f7a000;">
                    <div class="metric-title">{insights['clickbait_metrics']['title']}</div>
                    <div style="color:#d9d9d9;">{insights['clickbait_metrics']['description']}</div>
                </div>
                """, unsafe_allow_html=True)
                
        # Velocity
        with col4:
            if 'velocity_importance' in insights:
                st.markdown(f"""
                <div class="metric-card" style="border-left-color: #4cd137;">
                    <div class="metric-title">{insights['velocity_importance']['title']}</div>
                    <div style="color:#d9d9d9;">{insights['velocity_importance']['description']}</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("Insights file not found. Run the EDA and ML scripts.")

# --- TAB 2: Exploratory Data Analysis ---
with tab2:
    st.header("Visualizing the Algorithm")
    
    # 1. Best Time to Upload Heatmap
    st.subheader("1. Optimal Upload Windows (Heatmap)")
    hm_path = os.path.join(BASE_DIR, 'outputs', 'upload_time_heatmap.png')
    if os.path.exists(hm_path):
        st.image(hm_path, use_column_width=True)
        
    st.markdown("---")
    
    # 2. Outliers (Clickbait vs Core Audience)
    st.subheader("2. Interpreting Outliers: Clickbait vs Core Communities")
    st.markdown("**Clickbait:** High Views, Low Engagement Rate. **Strong Community:** Moderate Views, High Engagement Rate.")
    oc_path = os.path.join(BASE_DIR, 'outputs', 'views_vs_engagement.png')
    if os.path.exists(oc_path):
        st.image(oc_path, use_column_width=True)

    st.markdown("---")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Category Performance")
        cat_path = os.path.join(BASE_DIR, 'outputs', 'category_views.png')
        if os.path.exists(cat_path):
            st.image(cat_path, use_column_width=True)
            
    with col_b:
        st.subheader("Correlation Matrix")
        corr_path = os.path.join(BASE_DIR, 'outputs', 'correlation_matrix.png')
        if os.path.exists(corr_path):
            st.image(corr_path, use_column_width=True)

# --- TAB 3: Machine Learning ---
with tab3:
    st.header("Predicting Highly Engaging Videos")
    
    if insights and 'ml_metrics' in insights:
        metrics = insights['ml_metrics']
        rf_metrics = metrics.get('Random_Forest', {})
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Model Accuracy", f"{rf_metrics.get('Accuracy', 0)*100:.1f}%")
        col_m2.metric("Precision", f"{rf_metrics.get('Precision', 0)*100:.1f}%")
        col_m3.metric("Recall", f"{rf_metrics.get('Recall', 0)*100:.1f}%")
        col_m4.metric("ROC-AUC", f"{rf_metrics.get('ROC_AUC', 0):.2f}")
        
    st.markdown("---")
    
    st.subheader("Model Interpretability: Feature Importance")
    if insights and 'model_interpretation' in insights:
        mi = insights['model_interpretation']
        st.markdown(f"**{mi['title']}**")
        st.info(mi['description'])
        
    fi_path = os.path.join(BASE_DIR, 'outputs', 'feature_importance.png')
    if os.path.exists(fi_path):
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.image(fi_path, use_column_width=True)
