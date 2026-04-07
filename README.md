# YouTube Trending Analytics

A complete end-to-end data science portfolio project analyzing what makes a video trend on YouTube in the USA.

## Project Overview
This project uses the official YouTube Data API v3 to collect, clean, analyze, and model data on daily trending videos. It includes a complete Machine Learning pipeline and an interactive exploratory dashboard built with Streamlit.

### Key Highlights:
- **Data Engineering**: Automated pipeline pulling paginated results from YouTube Data API v3.
- **Exploratory Data Analysis (EDA)**: Extracted meaningful features (e.g. tag count, title length) and generated insightful correlation and distribution graphs.
- **Natural Language Processing (NLP)**: Used `TextBlob` to calculate the sentiment polarity of video titles and tags.
- **Machine Learning**: Trained a `RandomForestRegressor` to predict "Engagement Ratio" (Likes + Comments / Views) and visualize Feature Importance.
- **Data Presentation**: Developed a clean interactive Web App using `Streamlit`.

## Technology Stack
- **Languages**: Python 3
- **Libraries**: `pandas`, `sklearn`, `matplotlib`, `seaborn`, `textblob`, `streamlit`, `google-api-python-client`
- **APIs**: YouTube Data API v3

## How to Run Local Environment

1. Clone this repository.
2. Create your virtual environment and install requirements:
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

3. Obtain a YouTube Data API Key from Google Cloud Platform. 
4. Place your API key inside `config.py`:
```python
API_KEY="YOUR_API_KEY_HERE"
```

### Running the Data Pipeline
Execute the scripts in the following order to reproduce the findings:
```bash
python src/data_collection.py  # Fetches trending data to data/
python src/eda.py              # Generates EDA charts in outputs/
python src/modeling.py         # Trains the Model & saves insights in outputs/
```

### Starting the Dashboard
Run the following to fire up the interactive dashboard in your local browser:
```bash
streamlit run src/app.py
```

## Insights Discovered
- Top trending channels often maintain a specific tag-sentiment and optimal title length.
- The `RandomForestRegressor` identified that Title Sentiment and Tag Counts act as moderate predictors of a video's fundamental Engagement Rate relative to pure view-span.
