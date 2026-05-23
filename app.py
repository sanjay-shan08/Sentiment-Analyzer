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

# ──────────────────────────────────────────────
# CSS 
# ──────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&display=swap');

    /* ── CSS Variables ── */
    :root {
        --bg-base: #f4f6fb;
        --bg-white: #ffffff;
        --bg-sidebar: #f9fafb;
        --accent: #5b5fc7;
        --accent-light: #7b7fdb;
        --accent-subtle: rgba(91, 95, 199, 0.08);
        --accent-border: rgba(91, 95, 199, 0.15);
        --text-heading: #1e293b;
        --text-body: #475569;
        --text-muted: #94a3b8;
        --card-bg: #ffffff;
        --card-border: #e8ecf2;
        --card-shadow: 0 2px 12px rgba(30, 41, 59, 0.05);
        --card-shadow-hover: 0 8px 30px rgba(30, 41, 59, 0.1);
        --positive: #16a34a;
        --positive-bg: rgba(22, 163, 74, 0.07);
        --positive-border: rgba(22, 163, 74, 0.18);
        --negative: #dc2626;
        --negative-bg: rgba(220, 38, 38, 0.06);
        --negative-border: rgba(220, 38, 38, 0.16);
        --neutral: #64748b;
        --radius: 14px;
    }

    /* ── Base ── */
    .stApp, [data-testid="stAppViewContainer"] {
        background: var(--bg-base) !important;
        color: var(--text-body) !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    .stApp > header, [data-testid="stHeader"] {
        background: rgba(244, 246, 251, 0.9) !important;
        backdrop-filter: blur(12px) !important;
    }

    /* ── Background Decoration ── */
    .bg-decoration {
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        pointer-events: none;
        z-index: 0;
        overflow: hidden;
    }
    .bg-blob {
        position: absolute;
        border-radius: 50%;
        filter: blur(120px);
    }
    .bg-blob-1 {
        width: 500px; height: 500px;
        background: rgba(91, 95, 199, 0.06);
        top: -150px; right: -100px;
        animation: blobFloat 20s ease-in-out infinite;
    }
    .bg-blob-2 {
        width: 400px; height: 400px;
        background: rgba(22, 163, 74, 0.04);
        bottom: -100px; left: -80px;
        animation: blobFloat 25s ease-in-out infinite reverse;
    }
    .bg-blob-3 {
        width: 300px; height: 300px;
        background: rgba(123, 127, 219, 0.05);
        top: 40%; left: 60%;
        animation: blobFloat 22s ease-in-out infinite;
        animation-delay: -8s;
    }
    @keyframes blobFloat {
        0%, 100% { transform: translate(0, 0) scale(1); }
        33% { transform: translate(25px, -20px) scale(1.05); }
        66% { transform: translate(-15px, 15px) scale(0.95); }
    }

    /* ── Typography ── */
    h1, h2, h3, h4, h5, h6,
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: var(--text-heading) !important;
        font-weight: 700 !important;
    }
    p, span, label, .stMarkdown p, .stMarkdown span {
        font-family: 'DM Sans', sans-serif !important;
        color: var(--text-body) !important;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb {
        background: rgba(91, 95, 199, 0.18);
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover { background: rgba(91, 95, 199, 0.3); }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: var(--bg-sidebar) !important;
        border-right: 1px solid var(--card-border) !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span,
    [data-testid="stSidebar"] .stRadio label span,
    [data-testid="stSidebar"] .stSlider label {
        color: var(--text-body) !important;
    }
    [data-testid="stSidebar"] .stSlider [data-testid="stThumbValue"],
    [data-testid="stSidebar"] .stSlider [data-testid="stTickBarMin"],
    [data-testid="stSidebar"] .stSlider [data-testid="stTickBarMax"] {
        color: var(--text-muted) !important;
    }
    [data-testid="stSidebar"] [role="slider"] {
        background-color: var(--accent) !important;
    }

    /* ── Input Fields ── */
    .stTextInput input {
        background: var(--bg-white) !important;
        border: 1px solid var(--card-border) !important;
        border-radius: 10px !important;
        color: var(--text-heading) !important;
        font-family: 'DM Sans', sans-serif !important;
        padding: 12px 16px !important;
        transition: all 0.3s ease !important;
        box-shadow: var(--card-shadow) !important;
    }
    .stTextInput input:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px var(--accent-subtle), var(--card-shadow) !important;
    }
    .stTextInput input::placeholder {
        color: var(--text-muted) !important;
        opacity: 0.8 !important;
    }

    /* ── Primary Button ── */
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="stBaseButton-primary"] {
        background: #1e293b !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 700 !important;
        padding: 12px 24px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 10px rgba(30, 41, 59, 0.2) !important;
        letter-spacing: 0.3px !important;
    }
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="stBaseButton-primary"]:hover {
        background: #334155 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(30, 41, 59, 0.3) !important;
    }

    /* ── Download Button ── */
    .stDownloadButton > button {
        background: var(--bg-white) !important;
        border: 1px solid var(--card-border) !important;
        border-radius: 10px !important;
        color: var(--text-heading) !important;
        transition: all 0.3s ease !important;
        font-family: 'DM Sans', sans-serif !important;
        box-shadow: var(--card-shadow) !important;
    }
    .stDownloadButton > button:hover {
        border-color: var(--accent-border) !important;
        transform: translateY(-2px) !important;
        box-shadow: var(--card-shadow-hover) !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--bg-white) !important;
        border-radius: 12px !important;
        padding: 4px !important;
        gap: 4px !important;
        border: 1px solid var(--card-border) !important;
        box-shadow: var(--card-shadow) !important;
    }
    .stTabs [data-baseweb="tab"] {
        color: var(--text-muted) !important;
        border-radius: 9px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 500 !important;
        padding: 8px 20px !important;
        transition: all 0.3s ease !important;
    }
    .stTabs [aria-selected="true"] {
        background: var(--accent) !important;
        color: #ffffff !important;
    }
    .stTabs [data-baseweb="tab-highlight"],
    .stTabs [data-baseweb="tab-border"] {
        display: none !important;
    }

    /* ── Alerts ── */
    [data-testid="stAlert"] {
        background: var(--bg-white) !important;
        border: 1px solid var(--card-border) !important;
        border-radius: 10px !important;
        color: var(--text-body) !important;
        box-shadow: var(--card-shadow) !important;
    }
    .stSuccess { border-left: 3px solid var(--positive) !important; }
    .stError, [data-testid="stAlert"][data-type="error"] { border-left: 3px solid var(--negative) !important; }
    .stWarning { border-left: 3px solid #d97706 !important; }
    .stInfo { border-left: 3px solid var(--accent) !important; }

    /* ── Metrics (Streamlit defaults) ── */
    [data-testid="stMetric"],
    [data-testid="stMetricValue"],
    [data-testid="stMetricDelta"],
    [data-testid="stMetricLabel"] {
        color: var(--text-heading) !important;
    }
    [data-testid="stMetricLabel"] { color: var(--text-muted) !important; }

    /* ── Dataframe ── */
    .stDataFrame { border-radius: var(--radius) !important; overflow: hidden !important; }
    [data-testid="stDataFrame"] > div {
        background: var(--bg-white) !important;
        border: 1px solid var(--card-border) !important;
        border-radius: var(--radius) !important;
        box-shadow: var(--card-shadow) !important;
    }

    /* ── Horizontal Rule ── */
    hr { border-color: var(--card-border) !important; opacity: 0.6 !important; }

    /* ── Caption ── */
    .stCaption, [data-testid="stCaptionContainer"] { color: var(--text-muted) !important; }

    /* ── Soft Divider ── */
    .soft-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--card-border), rgba(91,95,199,0.12), var(--card-border), transparent);
        border: none;
        margin: 24px 0;
    }

    /* ── Metric Card ── */
    .metric-card {
        background: var(--bg-white);
        border: 1px solid var(--card-border);
        border-radius: 14px;
        padding: 20px 16px;
        text-align: center;
        transition: all 0.35s ease;
        box-shadow: var(--card-shadow);
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: var(--card-shadow-hover);
        border-color: var(--accent-border);
    }
    .metric-icon { font-size: 26px; margin-bottom: 8px; }
    .metric-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 26px;
        font-weight: 700;
        color: var(--text-heading);
        margin: 4px 0;
    }
    .metric-value.accent-color {
        color: var(--accent);
    }
    .metric-label {
        font-family: 'DM Sans', sans-serif;
        font-size: 12px;
        color: var(--text-muted);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    /* ── Section Header ── */
    .section-header {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 21px;
        color: var(--text-heading);
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* ── Post Card ── */
    .post-card {
        background: var(--bg-white);
        border: 1px solid var(--card-border);
        border-radius: 12px;
        padding: 18px 18px 18px 22px;
        margin-bottom: 10px;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
        box-shadow: var(--card-shadow);
    }
    .post-card::before {
        content: '';
        position: absolute;
        left: 0; top: 0; bottom: 0;
        width: 3px;
        border-radius: 3px 0 0 3px;
    }
    .post-card.positive::before { background: var(--positive); }
    .post-card.negative::before { background: var(--negative); }
    .post-card:hover {
        transform: translateX(3px);
        box-shadow: var(--card-shadow-hover);
        border-color: var(--accent-border);
    }
    .post-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 14px;
        font-weight: 600;
        color: var(--text-heading);
        margin-bottom: 10px;
        line-height: 1.55;
    }
    .post-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        align-items: center;
    }
    .post-pill {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 3px 10px;
        border-radius: 16px;
        font-size: 11px;
        font-family: 'DM Sans', sans-serif;
        font-weight: 500;
        background: var(--bg-base);
        border: 1px solid var(--card-border);
        color: var(--text-body);
    }
    .post-pill.sentiment-pos {
        color: var(--positive);
        border-color: var(--positive-border);
        background: var(--positive-bg);
    }
    .post-pill.sentiment-neg {
        color: var(--negative);
        border-color: var(--negative-border);
        background: var(--negative-bg);
    }
    .post-pill.subreddit {
        color: var(--accent);
        border-color: var(--accent-border);
        background: var(--accent-subtle);
    }
    .post-pill.stat { color: var(--text-muted); }
    .post-link {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        color: var(--accent);
        text-decoration: none;
        font-size: 11px;
        font-family: 'DM Sans', sans-serif;
        font-weight: 500;
        transition: all 0.2s ease;
        padding: 3px 10px;
        border-radius: 16px;
        border: 1px solid var(--accent-border);
    }
    .post-link:hover {
        background: var(--accent-subtle);
        color: var(--accent-light);
    }

    /* ── Feature Card (landing) ── */
    .feature-card {
        background: var(--bg-white);
        border: 1px solid var(--card-border);
        border-radius: 14px;
        padding: 30px 22px;
        text-align: center;
        transition: all 0.4s ease;
        animation: riseIn 0.7s ease forwards;
        opacity: 0;
        box-shadow: var(--card-shadow);
    }
    .feature-card:hover {
        border-color: var(--accent-border);
        transform: translateY(-6px);
        box-shadow: var(--card-shadow-hover);
    }
    .feature-card.delay-1 { animation-delay: 0.15s; }
    .feature-card.delay-2 { animation-delay: 0.35s; }
    .feature-card.delay-3 { animation-delay: 0.55s; }
    .feature-icon {
        font-size: 38px;
        margin-bottom: 16px;
        display: block;
    }
    .feature-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 16px;
        font-weight: 700;
        color: var(--text-heading);
        margin-bottom: 10px;
    }
    .feature-desc {
        font-family: 'DM Sans', sans-serif;
        font-size: 13px;
        color: var(--text-muted);
        line-height: 1.65;
    }

    @keyframes riseIn {
        from { opacity: 0; transform: translateY(24px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* ── Topic Chips ── */
    .topic-chips {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        justify-content: center;
        margin-top: 14px;
    }
    .topic-chip {
        display: inline-block;
        padding: 7px 16px;
        border-radius: 20px;
        font-family: 'DM Sans', sans-serif;
        font-size: 13px;
        font-weight: 500;
        color: var(--accent);
        background: var(--accent-subtle);
        border: 1px solid var(--accent-border);
        transition: all 0.3s ease;
        cursor: default;
    }
    .topic-chip:hover {
        background: rgba(91, 95, 199, 0.13);
        border-color: rgba(91, 95, 199, 0.3);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(91, 95, 199, 0.1);
    }

    /* ── Success Banner ── */
    .success-banner {
        background: var(--positive-bg);
        border: 1px solid var(--positive-border);
        border-radius: 10px;
        padding: 12px 18px;
        color: var(--positive);
        font-family: 'DM Sans', sans-serif;
        font-weight: 500;
        font-size: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* ── Connection Status (sidebar) ── */
    .conn-status {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 9px 12px;
        border-radius: 9px;
        margin: 8px 0;
        font-family: 'DM Sans', sans-serif;
        font-size: 12px;
        font-weight: 600;
    }
    .conn-status.online {
        background: var(--positive-bg);
        border: 1px solid var(--positive-border);
        color: var(--positive);
    }
    .conn-status.offline {
        background: var(--negative-bg);
        border: 1px solid var(--negative-border);
        color: var(--negative);
    }
    .conn-dot {
        width: 7px; height: 7px;
        border-radius: 50%;
        flex-shrink: 0;
    }
    .conn-dot.online {
        background: var(--positive);
        box-shadow: 0 0 6px rgba(22, 163, 74, 0.4);
        animation: softPulse 2.5s ease-in-out infinite;
    }
    .conn-dot.offline { background: var(--negative); }

    @keyframes softPulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(0.85); }
    }

    /* ── About Card (sidebar) ── */
    .about-card {
        background: var(--bg-white);
        border: 1px solid var(--card-border);
        border-radius: 10px;
        padding: 14px;
        margin-top: 8px;
        box-shadow: var(--card-shadow);
    }
    </style>

    <!-- Subtle background blobs -->
    <div class="bg-decoration">
        <div class="bg-blob bg-blob-1"></div>
        <div class="bg-blob bg-blob-2"></div>
        <div class="bg-blob bg-blob-3"></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# LOAD MODEL 
@st.cache_resource(show_spinner="Loading sentiment model... (first run takes ~10 seconds to train)")
def get_model():
    return load_model()

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 8px 0 16px 0;">
            <div style="font-size: 32px; margin-bottom: 6px;">📊</div>
            <div style="font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 18px;
                        color: #5b5fc7;">
                Sentiment Analyzer
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)

    st.markdown(
        "<p style='font-family: Space Grotesk, sans-serif; font-weight: 600; font-size: 11px; "
        "color: #94a3b8; text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 4px;'>"
        "⚙️ &nbsp;SETTINGS</p>",
        unsafe_allow_html=True,
    )

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
        format_func=lambda x: {"day": "Last 24 hours", "week": "Last 7 days", "month": "Last 30 days"}[x],
    )

    st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)

    @st.cache_resource(show_spinner=False)
    def check_reddit_connection():
        import praw
        client_id = os.getenv("REDDIT_CLIENT_ID")
        client_secret = os.getenv("REDDIT_CLIENT_SECRET")
        placeholders = ["your_client_id_here", "your_secret_here", "your_client_secret_here", "xyz123", "321zyx", "", None]
        if client_id in placeholders or client_secret in placeholders:
            return False
        try:
            reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent="SentimentAnalysisDashboard/1.0",
            )
            # force the lazy iterator to actually hit the API
            list(reddit.subreddit("all").search("test", limit=1))
            return True
        except Exception:
            return False

    connected = check_reddit_connection()

    if connected:
        st.markdown(
            '<div class="conn-status online">'
            '<div class="conn-dot online"></div>'
            'Reddit API Connected'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="conn-status offline">'
            '<div class="conn-dot offline"></div>'
            'Reddit Not Connected'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='font-size:11px;color:#94a3b8;margin-top:2px;'>"
            "Add valid credentials to your <code>.env</code> file.</p>",
            unsafe_allow_html=True,
        )

    st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="about-card">
            <p style="font-family:'Space Grotesk',sans-serif; font-weight:600; font-size:12px;
                      color:#475569; margin-bottom:6px;">About this project</p>
            <p style="font-size:11px; color:#94a3b8; line-height:1.65; margin:0;">
                Built with Streamlit, PRAW, scikit-learn & Plotly.
                Uses a TF-IDF + Logistic Regression model trained on ~10k labeled tweets.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────
st.markdown(
    """
    <div style="margin-bottom: 6px;">
        <h1 style="font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 38px;
                   margin: 0; padding: 0; line-height: 1.2;
                   color: #1e293b;">
            📊 Reddit Sentiment Analyzer
        </h1>
        <p style="font-size: 15px; color: #94a3b8; margin-top: 8px; font-weight: 400;
                  font-family: 'DM Sans', sans-serif;">
            Analyze public sentiment for any topic using Reddit data + machine learning
        </p>
    </div>
    <div class="soft-divider"></div>
    """,
    unsafe_allow_html=True,
)

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

# ──────────────────────────────────────────────
# MAIN DASHBOARD 
# ──────────────────────────────────────────────
if run and topic.strip():

    model = get_model()

    # fetch data
    with st.spinner(f"Fetching Reddit posts about **'{topic}'**..."):
        try:
            if connected:
                df = fetch_posts(topic.strip(), limit=num_posts, time_filter=time_filter)
            else:
                # fallback to sample data when API isn't connected
                sample_path = os.path.join(os.path.dirname(__file__), "sample_data.csv")
                if os.path.exists(sample_path):
                    df = pd.read_csv(sample_path)
                    # filter by topic (case-insensitive match in text)
                    mask = df["text"].str.contains(topic.strip(), case=False, na=False)
                    df = df[mask].head(num_posts).reset_index(drop=True)
                    if len(df) == 0:
                        df = pd.read_csv(sample_path).head(num_posts)
                    st.warning("⚠️ Using sample data — Reddit API is not connected. Results are simulated.")
                else:
                    st.error("❌ Reddit API not connected and no sample data found.")
                    st.stop()
        except ConnectionError as e:
            st.error(f"❌ {e}")
            st.stop()
        except Exception as e:
            st.error(f"❌ Failed to fetch data: {e}")
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

    # Success banner
    st.markdown(
        f'<div class="success-banner">✅ Analyzed <strong>{len(df)} posts</strong> about <strong>\'{topic}\'</strong></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)

    # ── METRICS ROW ──
    st.markdown('<div class="section-header">📈 Summary</div>', unsafe_allow_html=True)
    st.markdown("")

    total = len(df)
    pos = (df["sentiment"] == "Positive").sum()
    neg = (df["sentiment"] == "Negative").sum()
    avg_conf = df["confidence"].mean()
    avg_score = df["sent_score"].mean()

    overall_label = "Positive 📈" if avg_score > 0.05 else ("Negative 📉" if avg_score < -0.05 else "Neutral ➡️")

    c1, c2, c3, c4, c5 = st.columns(5)

    def metric_card(icon, label, value, accent=False):
        accent_class = " accent-color" if accent else ""
        return f"""
        <div class="metric-card">
            <div class="metric-icon">{icon}</div>
            <div class="metric-value{accent_class}">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """

    with c1:
        st.markdown(metric_card("📋", "Posts Analyzed", total, accent=True), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card("😊", "Positive", f"{pos} ({pos/total*100:.0f}%)"), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card("😞", "Negative", f"{neg} ({neg/total*100:.0f}%)"), unsafe_allow_html=True)
    with c4:
        st.markdown(metric_card("🎯", "Avg Confidence", f"{avg_conf:.0%}"), unsafe_allow_html=True)
    with c5:
        st.markdown(metric_card("🧭", "Overall", overall_label, accent=True), unsafe_allow_html=True)

    st.markdown('<div class="soft-divider" style="margin: 28px 0;"></div>', unsafe_allow_html=True)

    # ── CHARTS ROW 1: pie + time series ──
    st.markdown('<div class="section-header">📊 Sentiment Breakdown</div>', unsafe_allow_html=True)

    CHART_COLORS = {"Positive": "#16a34a", "Negative": "#dc2626", "Neutral": "#64748b"}
    CHART_LAYOUT = dict(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#475569", family="DM Sans, Space Grotesk, sans-serif", size=12),
        margin=dict(t=50, b=10, l=10, r=10),
    )

    col_left, col_right = st.columns([1, 2])

    with col_left:
        counts = df["sentiment"].value_counts().reset_index()
        counts.columns = ["Sentiment", "Count"]

        fig_pie = px.pie(
            counts,
            values="Count",
            names="Sentiment",
            hole=0.55,
            color="Sentiment",
            color_discrete_map=CHART_COLORS,
            title="Sentiment Distribution",
        )
        fig_pie.update_traces(
            textposition="inside", textinfo="percent+label",
            textfont=dict(color="white", size=11, family="DM Sans, sans-serif"),
        )
        fig_pie.update_layout(
            **CHART_LAYOUT, height=380, showlegend=False,
            title_font=dict(color="#1e293b", size=14, family="Space Grotesk, sans-serif"),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
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
            color_discrete_map=CHART_COLORS,
            title="Sentiment Over Time",
            labels={"count": "Number of Posts", "date": "Date", "sentiment": "Sentiment"},
        )
        fig_line.update_layout(
            **CHART_LAYOUT, height=380,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                        font=dict(color="#94a3b8", size=11)),
            title_font=dict(color="#1e293b", size=14, family="Space Grotesk, sans-serif"),
        )
        fig_line.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.04)", color="#94a3b8")
        fig_line.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.04)", color="#94a3b8")
        st.plotly_chart(fig_line, use_container_width=True)

    # ── CHARTS ROW 2: histogram + scatter ──
    st.markdown('<div class="section-header">🎯 Deeper Analysis</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        fig_hist = px.histogram(
            df,
            x="sent_score",
            color="sentiment",
            nbins=30,
            opacity=0.75,
            barmode="overlay",
            color_discrete_map=CHART_COLORS,
            title="Sentiment Score Distribution (−1 to +1)",
            labels={"sent_score": "Sentiment Score", "count": "Frequency"},
        )
        fig_hist.add_vline(x=0, line_dash="dash", line_color="rgba(100,116,139,0.25)", opacity=0.5)
        fig_hist.update_layout(
            **CHART_LAYOUT, height=340,
            title_font=dict(color="#1e293b", size=14, family="Space Grotesk, sans-serif"),
        )
        fig_hist.update_xaxes(color="#94a3b8", gridcolor="rgba(0,0,0,0.04)")
        fig_hist.update_yaxes(color="#94a3b8", gridcolor="rgba(0,0,0,0.04)")
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_b:
        fig_scatter = px.scatter(
            df,
            x="upvotes",
            y="sent_score",
            color="sentiment",
            opacity=0.7,
            color_discrete_map=CHART_COLORS,
            title="Upvotes vs Sentiment Score",
            labels={"upvotes": "Reddit Upvotes", "sent_score": "Sentiment Score"},
            hover_data=["subreddit"],
        )
        fig_scatter.add_hline(y=0, line_dash="dash", line_color="rgba(100,116,139,0.25)", opacity=0.5)
        fig_scatter.update_layout(
            **CHART_LAYOUT, height=340,
            title_font=dict(color="#1e293b", size=14, family="Space Grotesk, sans-serif"),
        )
        fig_scatter.update_xaxes(color="#94a3b8", gridcolor="rgba(0,0,0,0.04)")
        fig_scatter.update_yaxes(color="#94a3b8", gridcolor="rgba(0,0,0,0.04)")
        st.plotly_chart(fig_scatter, use_container_width=True)

    # ── POSTS ──
    st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">💬 Posts</div>', unsafe_allow_html=True)

    tab_pos, tab_neg, tab_all = st.tabs(["😊 Most Positive", "😞 Most Negative", "📋 All Posts"])

    def render_post_card(row):
        sentiment = row["sentiment"]
        css_class = "positive" if sentiment == "Positive" else "negative"
        emoji = "😊" if sentiment == "Positive" else "😞"
        sent_pill_class = "sentiment-pos" if sentiment == "Positive" else "sentiment-neg"

        title_display = str(row["title"])[:160] + ("..." if len(str(row["title"])) > 160 else "")

        link_html = ""
        if row["url"] != "#":
            link_html = f'<a href="{row["url"]}" target="_blank" class="post-link">View on Reddit →</a>'

        card_html = f"""
        <div class="post-card {css_class}">
            <div class="post-title">{title_display}</div>
            <div class="post-meta">
                <span class="post-pill {sent_pill_class}">{emoji} {sentiment} ({row['confidence']:.0%})</span>
                <span class="post-pill stat">⬆️ {int(row['upvotes'])}</span>
                <span class="post-pill stat">💬 {int(row['num_comments'])}</span>
                <span class="post-pill subreddit">r/{row['subreddit']}</span>
                {link_html}
            </div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)

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
    st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)
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
    # ── LANDING / EMPTY STATE ──
    st.markdown("")

    st.markdown(
        """
        <div style="text-align:center; margin: 16px 0 28px 0;">
            <p style="font-family:'Space Grotesk',sans-serif; font-size:19px; font-weight:600; color:#1e293b; display:flex; align-items:center; justify-content:center; gap:8px;">
                👆 Enter a topic above and hit
                <span style="background-color: #1e293b; font-color: #ffffff; padding: 4px 12px; border-radius: 8px; font-weight:700; font-size: 16px;">Analyze</span>
                to get started
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    info1, info2, info3 = st.columns(3)
    with info1:
        st.markdown(
            """
            <div class="feature-card delay-1">
                <span class="feature-icon">🌐</span>
                <div class="feature-title">Live Reddit Data</div>
                <div class="feature-desc">
                    Pulls the latest posts from across Reddit for any keyword or phrase you search.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with info2:
        st.markdown(
            """
            <div class="feature-card delay-2">
                <span class="feature-icon">🤖</span>
                <div class="feature-title">ML Sentiment Analysis</div>
                <div class="feature-desc">
                    TF-IDF vectorizer + Logistic Regression model, trained on 10,000 labeled tweets.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with info3:
        st.markdown(
            """
            <div class="feature-card delay-3">
                <span class="feature-icon">📊</span>
                <div class="feature-title">Interactive Charts</div>
                <div class="feature-desc">
                    Sentiment trends, score distributions, and a full post breakdown.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.markdown(
        """
        <div style="text-align:center; margin-top:8px;">
            <p style="font-family:'DM Sans',sans-serif; font-size:13px; color:#94a3b8; font-weight:500; margin-bottom:10px;">
                Try searching for
            </p>
            <div class="topic-chips">
                <span class="topic-chip">Bitcoin</span>
                <span class="topic-chip">ChatGPT</span>
                <span class="topic-chip">climate change</span>
                <span class="topic-chip">Netflix</span>
                <span class="topic-chip">Taylor Swift</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )