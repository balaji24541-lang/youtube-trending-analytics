import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="YouTube Trending Analytics", layout="wide")

st.title("YouTube Trending Data Science Portfolio Project")
st.markdown("""
This project deeply explores YouTube's trending videos using the YouTube Data API v3. 
We performed data collection, exploratory data analysis, and predictive modeling (using Random Forest) to figure out which features contribute most to high engagement rates.
""")

@st.cache_data
def load_data():
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'trending_videos_us.csv')
    if not os.path.exists(data_path):
        return None
    return pd.read_csv(data_path)

df = load_data()

if df is None:
    st.warning("Data not found. Please run the data collection script first.")
else:
    st.header("1. High-Level Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Videos Analyzed", f"{len(df)}")
    with col2:
        st.metric("Total Views", f"{df['view_count'].sum():,.0f}")
    with col3:
        st.metric("Total Likes", f"{df['like_count'].sum():,.0f}")
    with col4:
        st.metric("Avg Views per Video", f"{df['view_count'].mean():,.0f}")
        
    st.header("2. Exploratory Data Analysis")
    st.markdown("We generated automated charts highlighting the distributions and relationships of the video metrics.")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Distribution of View Counts")
        img_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs', 'views_distribution.png')
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)
            
    with col_b:
        st.subheader("Correlation between Metrics")
        img_path2 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs', 'correlation_matrix.png')
        if os.path.exists(img_path2):
            st.image(img_path2, use_container_width=True)
            
    st.subheader("Top Channels by Trending Views")
    img_path3 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs', 'top_channels_views.png')
    if os.path.exists(img_path3):
        st.image(img_path3, use_container_width=True)

    st.header("3. Advanced Modeling (Random Forest)")
    st.markdown("We trained a Random Forest regressor on NLP sentiment and metadata to predict the video's engagement ratio (Likes + Comments / Views). Below is the Feature Importance graph.")
    
    img_path4 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs', 'feature_importance.png')
    if os.path.exists(img_path4):
        st.image(img_path4, use_container_width=True)
        
    st.header("Raw Data Sample")
    st.dataframe(df.head(10), use_container_width=True)
