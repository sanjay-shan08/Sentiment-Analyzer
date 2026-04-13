# fetch_data.py
# handles pulling posts from Reddit using PRAW

import praw
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

def get_reddit_client():
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")

    if not client_id or not client_secret:
        raise ConnectionError("Reddit credentials not found. Please add REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET to your .env file.")

    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent="SentimentAnalysisDashboard/1.0",
    )
    return reddit

def fetch_posts(topic, limit=100, time_filter="week"):
    # Fetches reddit posts about a topic using the Reddit search API.
    # Raises an error if credentials are missing or the request fails.
    
    reddit = get_reddit_client()

    posts = []

    results = reddit.subreddit("all").search(
        topic, limit=limit, sort="new", time_filter=time_filter
    )

    for submission in results:
        text = submission.title
        if submission.selftext and len(submission.selftext) > 20:
            text += " " + submission.selftext[:300]

        posts.append(
            {
                "text": text,
                "title": submission.title,
                "upvotes": submission.score,
                "created": datetime.utcfromtimestamp(submission.created_utc),
                "subreddit": submission.subreddit.display_name,
                "url": "https://reddit.com" + submission.permalink,
                "num_comments": submission.num_comments,
            }
        )

    if len(posts) == 0:
        return None

    return pd.DataFrame(posts)