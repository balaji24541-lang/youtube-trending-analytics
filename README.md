# 📈 What Makes a YouTube Video Trend?

I've always been super curious about the YouTube algorithm and what sets trending videos apart from the rest. Instead of just guessing, I decided to pull actual data from YouTube and do a deep dive. 

This is an end-to-end data pipeline I built to fetch, analyze, and visualize daily trending videos in the US. I also threw in a Random Forest model to see if I could predict engagement rates based on things like clickbait-y titles and tag sentiment.

## What's exactly in here?

Instead of using a stale CSV downloader from Kaggle, my `data_collection.py` script hits the official **YouTube Data API v3** to pull fresh lists of trending videos dynamically.

Here is how the pipeline flows natively:
1. **Data Engineering (`data_collection.py`)**: Connects to Google's API, paginates through the "Most Popular" chart, and dumps the raw JSON into structured Pandas DataFrames.
2. **Exploratory Data Analysis (`eda.py`)**: Cleans up the datasets and generates automated distribution metrics (like the views vs. correlation heatmaps found in the `outputs/` folder).
3. **Machine Learning (`modeling.py`)**: 
   - I used `TextBlob` to extract NLP sentiment polarity from the video titles and tags (e.g. are positive titles more likely to trend?).
   - Trained a `RandomForestRegressor` to map those features to the video's "Engagement Ratio" (Likes + Comments / Views).
4. **Interactive Dashboard (`app.py`)**: I wrapped everything into a clean Streamlit web app so you can interactively explore the data and see the Random Forest feature importances firsthand.

## How to run it yourself

If you want to pull today's trending data, follow these steps:

1. Clone this repo down to your machine.
2. Create a virtual environment and grab the requirements:
   ```bash
   python -m venv .venv
   # Windows:
   .\.venv\Scripts\Activate.ps1
   # Mac/Linux:
   source .venv/bin/activate
   
   pip install -r requirements.txt
   ```
3. You'll need your own YouTube Data API key. Grab one from the Google Cloud Console, and create a file named `config.py` in the root directory. Add this line to it:
   ```python
   API_KEY="your_api_key_here"
   ```
   *(Note: `config.py` is ignored by Git, so your key stays safe!).*

4. Run the pipeline scripts sequentially:
   ```bash
   python src/data_collection.py  # Hits the API and saves the new CSV
   python src/eda.py              # Generates fresh charts
   python src/modeling.py         # Runs the NLP/RF modeling
   ```

5. Finally, spin up the Streamlit dashboard:
   ```bash
   streamlit run src/app.py
   ```

## What I Learned
Interestingly, the model showed that raw tag volume and title sentiment definitely play a moderate, quantifiable role in driving the base engagement rates up, independent of just the channel size. 

Feel free to break it, fork it, or reach out if you have any cool ideas on what features I should add to the model next!
