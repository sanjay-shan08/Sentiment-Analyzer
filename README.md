# Reddit Sentiment Analyzer 📊

A sentiment analysis dashboard that pulls live Reddit posts for any topic and visualizes public opinion using machine learning.

## Tech Stack
- Python
- Streamlit
- scikit-learn
- PRAW 
- Plotly

![Python](https://img.shields.io/badge/Python-3.8+-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-orange)

---

## Features

- Search any topic and fetch live Reddit posts
- Sentiment classification using TF-IDF + Logistic Regression
- Sentiment trend over time (line chart)
- Sentiment distribution (pie chart)
- Confidence score distribution and upvotes vs sentiment scatter plot
- Top positive/negative post breakdown

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/sanjay-shan08/Sentiment-Analyzer.git
cd Sentiment-Analyser
```

### 2. Install dependencies
```bash
pip install streamlit>=1.28.0
pip install praw>=7.7.0
pip install pandas>=1.5.0
pip install plotly>=5.15.0
pip install scikit-learn>=1.2.0
pip install nltk>=3.8.0
pip install python-dotenv>=1.0.0
pip install numpy>=1.24.0
pip install requests>=2.31.0
```

### 3. Add Reddit API credentials

Copy the env file and fill in your credentials:
```bash
cp .env
```

Then open `.env` and add your Reddit app credentials:
```
REDDIT_CLIENT_ID=your_client_id_here
REDDIT_CLIENT_SECRET=your_client_secret_here
```

To get credentials, first you'll need to Submit an API access request to Reddit on [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps). This usually takes a few days, once your request is approved then you can use the same link to create a **script** type app and set the redirect URL to `http://localhost:8080`, You'll get your API credentials as the app is created.

### 4. Run the app
```bash
streamlit run app.py
```

The first run will download the NLTK training data and train the sentiment model — this takes about 10 seconds. After that it loads from cache instantly.

---

## How it works

**Data** — Reddit posts are fetched using the PRAW library via Reddit's official search API.

**Model** — A scikit-learn pipeline of TF-IDF vectorizer (15k features, bigrams) + Logistic Regression, trained on NLTK's Twitter Samples corpus (5,000 positive + 5,000 negative tweets). The model is saved to disk after the first training run.

**Sentiment score** — Each post gets a label (Positive/Negative), a confidence percentage, and a score from -1 (very negative) to +1 (very positive).

---

## Project structure

```bash
sentiment-dashboard/
=> app.py         # Streamlit UI and dashboard logic
=> model.py          # Model training and sentiment prediction
=> fetch_data.py     # Reddit API integration
=> .env      # Template for credentials
=> .gitignore
```

---

## Requirements

- Python 3.8+
- Reddit API credentials (see setup above)


NOTE:
This is still a work in progress as my API access request hasn't been confirmed yet.