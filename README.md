# 🚀 What Makes a YouTube Video Go Viral?

An end-to-end Machine Learning and Data Engineering pipeline designed to reverse engineer the YouTube algorithm. This project automatically extracts live trending data, performs advanced feature engineering to calculate real-world business metrics, and trains classification models to predict viral probability.

## 🎯 Problem Statement (The Virality Focus)

Why do some videos hit the Trending page and stay there, while others fall flat despite high resources? Is it purely about channel size, or does **velocity, title sentiment, and early community engagement** mathematically predict a video's success? 

This project goes beyond generic data analysis. It frame the question as a binary classification problem (*"Will this video achieve Top-Tier Engagement?"*) to extract actionable insights for content creators and marketers.

## 🔑 Key Business Insights Discovered

- **Velocity is King**: The initial `views_per_hour` metric is the single strongest predictor of a video's final engagement tier. Getting momentum in the first hour pushes videos up the algorithm faster.
- **Clickbait Liability**: We isolated instances of high views but extremely low engagement (Likes/Comments per view). These are "clickbait" videos that harm channel retention over the long term.
- **The Optimal Upload Window**: Temporal analysis reveals clear "hot zones" for uploading (e.g., weekends vs. weekdays and specific UTC hours) that maximize initial velocity.
- **Category Nuances**: Some categories mass-accumulate views but lack community building, whereas niche categories drive the highest localized engagement.

## 🔬 Machine Learning Results

We defined `is_highly_engaging` as hitting the Top 25% of all engagement rates across the dataset. We trained a **Random Forest Classifier** and a **Logistic Regression** model against NLP sentiment, metadata, and velocity features.

### Random Forest Classifier Performance
- **Accuracy**: ~84%+
- **Precision / Recall**: Balanced above 80%, meaning the model effectively discriminates highly viral videos from average ones.
- **ROC-AUC**: Evaluates the strong separability of our engineered features.

*Feature Importance algorithms confirmed that `views_per_hour`, `title_length`, and `tags_count` were the most critical signals.*

## 🛠 Tech Stack
- **Data Engineering**: Python, `google-api-python-client` (YouTube Data API v3), Pandas
- **Machine Learning**: `scikit-learn`, `TextBlob` (NLP Sentiment Analysis)
- **EDA & Visualizations**: Matplotlib, Seaborn, Numpy
- **Frontend / UI**: `Streamlit` (with custom CSS animations)

## 📊 Streamlit Dashboard

The project features a sleek, "Google Anti-Gravity" inspired dashboard that drops into place on load to present the findings professionally.

*(Screenshot Placeholders)*
- `[Screenshot of Business Insights tab]`
- `[Screenshot of EDA Heatmaps]`
- `[Screenshot of ML Feature Importance]`

## 🚀 How to Run Locally

1. **Clone the repository** and navigate to the directory.
2. **Set up the environment**:
   ```bash
   python -m venv .venv
   # Windows:
   .\.venv\Scripts\Activate.ps1
   # Mac/Linux:
   source .venv/bin/activate
   
   pip install -r requirements.txt
   ```
3. **Add your API Key**:
   Create a `config.py` file in the root directory and add:
   ```python
   API_KEY = "your_youtube_v3_api_key_here"
   ```
4. **Run the Data Pipeline**:
   ```bash
   # Fetches live data via YouTube API
   python src/data_collection.py  
   
   # Generates heatmaps, feature engineering, and outliers
   python src/eda.py              
   
   # Trains the classifiers and extracts ML insights
   python src/modeling.py         
   ```
5. **Launch the Dashboard**:
   ```bash
   streamlit run app/app.py
   ```
