# Run with: streamlit run app.py
# NOTE:
# Download dependencies
# Make sure you've set up your .env file with Reddit API credentials

import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from dotenv import load_dotenv

from fetch_data import fetch_posts
from model import load_model, predict

load_dotenv()

# PAGE CONFIG
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    unsafe_allow_html=True,
)

# LOAD MODEL 
@st.cache_resource(show_spinner="Loading sentiment model... (first run takes ~10 seconds to train)")
def get_model():
    return load_model()

# SIDEBAR
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    num_posts = st.slider(
        "Posts to fetch",
        min_value=25,
        max_value=200,
        value=100,
        step=25,
        help="More posts = more accurate sentiment but slower to load",
    )

    time_filter = st.radio(
        "Time range",
        options=["day", "week", "month"],
        index=1,
        format_func=lambda x: {"day": "⏰ Last 24 hours", "week": "📅 Last 7 days", "month": "🗓️ Last 30 days"}[x],
    )

    st.markdown("---")

    # actually test the connection rather than just checking if vars exist
    @st.cache_resource(show_spinner=False)
    def check_reddit_connection():
        import praw
        client_id = os.getenv("REDDIT_CLIENT_ID")
        client_secret = os.getenv("REDDIT_CLIENT_SECRET")
        placeholders = ["your_client_id_here", "your_secret_here", "", None]
        if client_id in placeholders or client_secret in placeholders:
            return False
        try:
            reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent="SentimentAnalysisDashboard/1.0",
            )
            # this actually hits the API to verify creds work
            reddit.subreddit("all").search("test", limit=1)
            return True
        except Exception:
            return False

    connected = check_reddit_connection()

    if connected:
        st.success("✅ Reddit API connected\nUsing live data")
    else:
        st.error("❌ Reddit not connected")
        st.markdown(
            "Add valid credentials to your `.env` file to use the app."
        )

    st.markdown("---")
    st.markdown("**About this project**")
    st.caption(
        "Built with Streamlit, PRAW, scikit-learn & Plotly. "
        "Uses a TF-IDF + Logistic Regression model trained on ~10k labeled tweets."
    )


# HEADER
st.title("📊 Reddit Sentiment Analyzer")
st.markdown("*Analyze public sentiment for any topic using Reddit data + machine learning*")
st.markdown("---")

# SEARCH BAR
search_col, btn_col = st.columns([5, 1])

with search_col:
    topic = st.text_input(
        "topic",
        placeholder="Enter a topic... e.g. Tesla, Bitcoin, ChatGPT, climate change",
        label_visibility="collapsed",
    )

with btn_col:
    run = st.button("🔍 Analyze", type="primary", use_container_width=True)

# MAIN DASHBOARD 
if run and topic.strip():

    model = get_model()

    # fetch data
    with st.spinner(f"Fetching Reddit posts about **'{topic}'**..."):
        try:
            df = fetch_posts(topic.strip(), limit=num_posts, time_filter=time_filter)
        except ConnectionError as e:
            st.error(f"❌ {e}")
            st.stop()
        except Exception as e:
            st.error(f"❌ Failed to fetch from Reddit: {e}")
            st.stop()

    if df is None or len(df) == 0:
        st.error("No posts found for that topic. Try a different keyword or time range.")
        st.stop()

    # run sentiment analysis
    with st.spinner("Running sentiment analysis..."):
        sentiment_results = predict(df["text"].tolist(), model)
        sent_df = pd.DataFrame(sentiment_results)
        df = pd.concat([df.reset_index(drop=True), sent_df], axis=1)

    # parse dates properly
    df["created"] = pd.to_datetime(df["created"])
    df["date"] = df["created"].dt.date

    st.success(f"✅ Analyzed **{len(df)} posts** about **'{topic}'**")
    st.markdown("---")

    # METRICS R
    st.markdown("### 📈 Summary")

    total = len(df)
    pos = (df["sentiment"] == "Positive").sum()
    neg = (df["sentiment"] == "Negative").sum()
    avg_conf = df["confidence"].mean()
    avg_score = df["sent_score"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Posts Analyzed", total)
    c2.metric("Positive 😊", f"{pos} ({pos/total*100:.0f}%)")
    c3.metric("Negative 😞", f"{neg} ({neg/total*100:.0f}%)")
    c4.metric("Avg Confidence", f"{avg_conf:.0%}")
    c5.metric(
        "Overall Sentiment",
        "Positive 📈" if avg_score > 0.05 else ("Negative 📉" if avg_score < -0.05 else "Neutral ➡️"),
        delta=f"{avg_score:+.2f}",
    )

    st.markdown("---")

    # CHARTS ROW 1: pie + time series
    st.markdown("### 📊 Sentiment Breakdown")

    col_left, col_right = st.columns([1, 2])

    with col_left:
        # donut chart
        counts = df["sentiment"].value_counts().reset_index()
        counts.columns = ["Sentiment", "Count"]

        fig_pie = px.pie(
            counts,
            values="Count",
            names="Sentiment",
            hole=0.45,
            color="Sentiment",
            color_discrete_map={"Positive": "#27ae60", "Negative": "#e74c3c", "Neutral": "#95a5a6"},
            title="Sentiment Distribution",
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        fig_pie.update_layout(
            height=360,
            showlegend=False,
            margin=dict(t=50, b=10, l=10, r=10),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        # sentiment over time — count per day per sentiment
        time_df = (
            df.groupby(["date", "sentiment"])
            .size()
            .reset_index(name="count")
        )

        fig_line = px.line(
            time_df,
            x="date",
            y="count",
            color="sentiment",
            markers=True,
            color_discrete_map={"Positive": "#27ae60", "Negative": "#e74c3c", "Neutral": "#95a5a6"},
            title="Sentiment Over Time",
            labels={"count": "Number of Posts", "date": "Date", "sentiment": "Sentiment"},
        )
        fig_line.update_layout(
            height=360,
            margin=dict(t=50, b=10, l=10, r=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig_line.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
        fig_line.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
        st.plotly_chart(fig_line, use_container_width=True)

    # CHARTS ROW 2: histogram + scatter
    st.markdown("### 🎯 Deeper Analysis")

    col_a, col_b = st.columns(2)

    with col_a:
        # sentiment score distribution
        fig_hist = px.histogram(
            df,
            x="sent_score",
            color="sentiment",
            nbins=30,
            opacity=0.75,
            barmode="overlay",
            color_discrete_map={"Positive": "#27ae60", "Negative": "#e74c3c", "Neutral": "#95a5a6"},
            title="Sentiment Score Distribution (−1 to +1)",
            labels={"sent_score": "Sentiment Score", "count": "Frequency"},
        )
        fig_hist.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig_hist.update_layout(height=320, margin=dict(t=50, b=10, l=10, r=10))
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_b:
        # upvotes vs sentiment
        fig_scatter = px.scatter(
            df,
            x="upvotes",
            y="sent_score",
            color="sentiment",
            opacity=0.6,
            color_discrete_map={"Positive": "#27ae60", "Negative": "#e74c3c", "Neutral": "#95a5a6"},
            title="Upvotes vs Sentiment Score",
            labels={"upvotes": "Reddit Upvotes", "sent_score": "Sentiment Score"},
            hover_data=["subreddit"],
        )
        fig_scatter.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig_scatter.update_layout(height=320, margin=dict(t=50, b=10, l=10, r=10))
        st.plotly_chart(fig_scatter, use_container_width=True)

    # TOP POSTS + FULL TABLE
    st.markdown("---")
    st.markdown("### 💬 Posts")

    tab_pos, tab_neg, tab_all = st.tabs(["😊 Most Positive", "😞 Most Negative", "📋 All Posts"])

    def render_post_card(row):
        emoji = "😊" if row["sentiment"] == "Positive" else "😞"
        color = "#27ae60" if row["sentiment"] == "Positive" else "#e74c3c"

        title_display = str(row["title"])[:160] + ("..." if len(str(row["title"])) > 160 else "")

        with st.container():
            st.markdown(f"**{title_display}**")
            meta_cols = st.columns([2, 2, 2, 2])
            meta_cols[0].markdown(
                f"<span style='color:{color}'>{emoji} {row['sentiment']} ({row['confidence']:.0%})</span>",
                unsafe_allow_html=True,
            )
            meta_cols[1].markdown(f"⬆️ **{int(row['upvotes'])}** upvotes")
            meta_cols[2].markdown(f"💬 **{int(row['num_comments'])}** comments")
            meta_cols[3].markdown(f"r/**{row['subreddit']}**")
            if row["url"] != "#":
                st.markdown(f"[→ View on Reddit]({row['url']})")
            st.markdown("---")

    with tab_pos:
        top_pos = df[df["sentiment"] == "Positive"].sort_values("confidence", ascending=False).head(10)
        if len(top_pos) == 0:
            st.info("No positive posts found for this topic.")
        for _, row in top_pos.iterrows():
            render_post_card(row)

    with tab_neg:
        top_neg = df[df["sentiment"] == "Negative"].sort_values("confidence", ascending=False).head(10)
        if len(top_neg) == 0:
            st.info("No negative posts found for this topic.")
        for _, row in top_neg.iterrows():
            render_post_card(row)

    with tab_all:
        display = df[["title", "sentiment", "confidence", "sent_score", "upvotes", "subreddit", "created"]].copy()
        display["title"] = display["title"].str[:100]
        display["confidence"] = display["confidence"].apply(lambda x: f"{x:.0%}")
        display["sent_score"] = display["sent_score"].round(2)
        display["created"] = display["created"].dt.strftime("%Y-%m-%d %H:%M")
        display = display.rename(columns={
            "title": "Post",
            "sentiment": "Sentiment",
            "confidence": "Confidence",
            "sent_score": "Score (-1→+1)",
            "upvotes": "Upvotes",
            "subreddit": "Subreddit",
            "created": "Posted At",
        })
        st.dataframe(display, use_container_width=True, height=420)

    # DOWNLOAD BUTTON
    st.markdown("---")
    export_df = df[["title", "sentiment", "confidence", "sent_score", "upvotes", "subreddit", "created"]].copy()
    export_df["created"] = export_df["created"].dt.strftime("%Y-%m-%d %H:%M")

    csv_data = export_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv_data,
        file_name=f"sentiment_{topic.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        help="Download all analyzed posts with their sentiment scores",
    )

elif run and not topic.strip():
    st.warning("⚠️ Please enter a topic before clicking Analyze.")

else:
    # placeholder / landing state
    st.markdown("### 👆 Enter a topic above and hit Analyze to get started")
    st.markdown("")

    info1, info2, info3 = st.columns(3)
    with info1:
        st.info(
            "**🌐 Live Reddit Data**\n\n"
            "Pulls the latest posts from across Reddit for any keyword or phrase you search."
        )
    with info2:
        st.info(
            "**🤖 ML Sentiment Analysis**\n\n"
            "TF-IDF vectorizer + Logistic Regression model, trained on 10,000 labeled tweets."
        )
    with info3:
        st.info(
            "**📊 Interactive Charts**\n\n"
            "Sentiment trends, score distributions, and a full post breakdown."
        )

    st.markdown("")
    st.markdown("**Try searching for:** `Bitcoin` · `ChatGPT` · `climate change` · `Netflix` · `Taylor Swift`")