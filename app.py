import html
from datetime import datetime
from pathlib import Path
import pandas as pd
import streamlit as st
import yfinance as yf
import altair as alt
from data_sources import (
    CATEGORIES,
    get_search_query_for_category,
    fetch_news,
    filter_market_moving,
    summarize_with_ollama,
)

try:
    from domain_updated import DomainSentimentPipeline
    DOMAIN_SENTIMENT_AVAILABLE = True
    DOMAIN_SENTIMENT_IMPORT_ERROR = None
except Exception as e:
    DOMAIN_SENTIMENT_AVAILABLE = False
    DOMAIN_SENTIMENT_IMPORT_ERROR = str(e)

# Map UI category to domain_sentiment_model domain
CATEGORY_TO_DOMAIN = {
    "Business and Finance": "finance",
    "Law and Government": "law",
    "Politics": "politics",
    "Technology": "tech",
}

# --- Setup ---
st.set_page_config(page_title="LexiSnap", layout="wide", initial_sidebar_state="collapsed")

# --- Custom CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Manrope:wght@400;500;600;700&display=swap');

    :root {
        --text-main: #16212f;
        --text-soft: #496079;
        --card-bg: rgba(255, 255, 255, 0.78);
        --card-border: rgba(255, 255, 255, 0.78);
        --brand: #0a7ea4;
        --brand-strong: #076386;
        --success: #0f9d78;
        --danger: #cc4b4b;
        --neutral: #5c7089;
    }

    .stApp {
        background:
            radial-gradient(circle at 0% 0%, rgba(255, 183, 77, 0.22) 0%, rgba(255, 183, 77, 0) 36%),
            radial-gradient(circle at 100% 0%, rgba(76, 182, 209, 0.28) 0%, rgba(76, 182, 209, 0) 35%),
            radial-gradient(circle at 50% 100%, rgba(95, 204, 186, 0.24) 0%, rgba(95, 204, 186, 0) 40%),
            linear-gradient(160deg, #f6f9fc 0%, #eef5fb 42%, #f8fafc 100%);
        font-family: 'Manrope', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--text-main);
    }

    .block-container {
        padding-top: 1.1rem;
    }

    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif;
        letter-spacing: -0.02em;
        color: var(--text-main);
    }

    .hero-wrap {
        border-radius: 26px;
        padding: 2.75rem 2.4rem 2.25rem 2.4rem;
        border: 1px solid var(--card-border);
        background: linear-gradient(140deg, rgba(255, 255, 255, 0.92), rgba(255, 255, 255, 0.7));
        box-shadow: 0 24px 60px rgba(20, 64, 91, 0.12);
        position: relative;
        overflow: hidden;
    }
    .hero-wrap::before {
        content: "";
        position: absolute;
        right: -90px;
        top: -130px;
        width: 300px;
        height: 300px;
        border-radius: 999px;
        background: radial-gradient(circle, rgba(10, 126, 164, 0.19), rgba(10, 126, 164, 0));
        pointer-events: none;
    }
    .hero {
        position: relative;
        z-index: 1;
    }
    .hero h1 {
        font-size: clamp(2.2rem, 5vw, 3.4rem);
        font-weight: 700;
        background: linear-gradient(92deg, #0f2438 0%, #0a7ea4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.3rem;
    }
    .hero p {
        color: var(--text-soft);
        font-size: 1.06rem;
        line-height: 1.6;
        margin-bottom: 1.4rem;
        max-width: 760px;
    }

    .chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin-top: 0.5rem;
    }
    .chip {
        display: inline-flex;
        align-items: center;
        border-radius: 999px;
        padding: 0.35rem 0.8rem;
        background: rgba(10, 126, 164, 0.09);
        border: 1px solid rgba(10, 126, 164, 0.18);
        color: #0e4d66;
        font-size: 0.8rem;
        font-weight: 600;
    }

    .panel {
        border-radius: 18px;
        padding: 1rem 1.1rem;
        border: 1px solid var(--card-border);
        background: var(--card-bg);
        box-shadow: 0 16px 40px rgba(13, 56, 80, 0.08);
        margin-bottom: 1rem;
        animation: riseIn 0.45s ease both;
    }
    .panel .kicker {
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #4d6f89;
        font-weight: 700;
        margin-bottom: 0.32rem;
    }
    .panel .value {
        color: var(--text-main);
        font-size: 1.18rem;
        font-weight: 700;
    }

    .dashboard-header {
        border-radius: 18px;
        border: 1px solid var(--card-border);
        background: rgba(255, 255, 255, 0.82);
        box-shadow: 0 14px 36px rgba(16, 67, 95, 0.08);
        margin-bottom: 1rem;
        padding: 1rem 1.15rem;
    }
    .dashboard-header h1 {
        margin: 0;
        font-size: 1.5rem;
        color: var(--text-main);
    }

    div[data-testid="stExpander"] {
        border-radius: 16px;
        border: 1px solid var(--card-border);
        background: var(--card-bg);
        box-shadow: 0 14px 34px rgba(20, 64, 91, 0.08);
        margin-bottom: 0.95rem;
        overflow: hidden;
        animation: riseIn 0.4s ease both;
    }
    .streamlit-expanderHeader {
        font-size: 1.04rem;
        font-weight: 700;
        color: #133146;
        background: rgba(255, 255, 255, 0.62) !important;
        border-radius: 16px !important;
        min-height: 54px !important;
    }

    .news-card {
        border-radius: 14px;
        border: 1px solid rgba(10, 126, 164, 0.15);
        background: linear-gradient(155deg, rgba(255, 255, 255, 0.95), rgba(245, 251, 255, 0.8));
        box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.65);
        padding: 0.9rem 0.95rem 0.7rem 0.95rem;
        margin-bottom: 0.9rem;
        animation: cardPop 0.45s ease both;
    }
    .news-meta {
        color: #55718b;
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 0.35rem;
    }
    .news-summary {
        color: #334f68;
        font-size: 0.9rem;
        line-height: 1.55;
        margin-top: 0.35rem;
    }

    .sentiment-badge {
        display: inline-block;
        padding: 0.33rem 0.66rem;
        border-radius: 10px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .sentiment-POSITIVE, .sentiment-positive { background: rgba(15, 157, 120, 0.16); color: var(--success); }
    .sentiment-NEGATIVE, .sentiment-negative { background: rgba(204, 75, 75, 0.16); color: var(--danger); }
    .sentiment-NEUTRAL, .sentiment-neutral { background: rgba(92, 112, 137, 0.16); color: var(--neutral); }

    .stButton > button {
        border-radius: 11px !important;
        font-weight: 700 !important;
        border: 1px solid rgba(7, 99, 134, 0.15) !important;
        color: white !important;
        background: linear-gradient(135deg, var(--brand), var(--brand-strong)) !important;
        transition: transform 0.16s ease, box-shadow 0.16s ease !important;
        box-shadow: 0 8px 20px rgba(7, 99, 134, 0.28);
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 10px 24px rgba(7, 99, 134, 0.35);
    }
    .stLinkButton > a {
        border-radius: 10px !important;
        font-weight: 600 !important;
    }

    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.85);
        border-radius: 14px;
        border: 1px solid var(--card-border);
        padding: 0.65rem;
    }
    div[data-testid="stMetric"] label, div[data-testid="stMetric"] div {
        color: var(--text-main) !important;
    }

    .loader-shell {
        position: relative;
        border-radius: 16px;
        border: 1px solid rgba(10, 126, 164, 0.18);
        background:
            linear-gradient(130deg, rgba(255,255,255,0.96), rgba(244,252,255,0.9)),
            linear-gradient(90deg, rgba(10,126,164,0.05), rgba(255,183,77,0.05));
        box-shadow: 0 14px 34px rgba(18, 72, 102, 0.12);
        padding: 1rem 1rem 1rem 4.8rem;
        margin-bottom: 0.95rem;
        overflow: hidden;
        min-height: 76px;
    }
    .loader-shell::after {
        content: "";
        position: absolute;
        inset: 0;
        background: linear-gradient(110deg, transparent 28%, rgba(255,255,255,0.45) 48%, transparent 68%);
        transform: translateX(-100%);
        animation: shimmerSweep 1.35s linear infinite;
        pointer-events: none;
    }
    .loader-core {
        position: absolute;
        left: 1.35rem;
        top: 50%;
        width: 1.15rem;
        height: 1.15rem;
        margin-top: -0.575rem;
        border-radius: 50%;
        background: linear-gradient(135deg, #0a7ea4, #39aac7);
        box-shadow: 0 0 0 7px rgba(10, 126, 164, 0.12);
        animation: pulseDot 1s ease-in-out infinite;
        z-index: 2;
    }
    .loader-orbit {
        position: absolute;
        left: 1.02rem;
        top: 50%;
        width: 1.8rem;
        height: 1.8rem;
        margin-top: -0.9rem;
        border-radius: 50%;
        border: 2px solid rgba(10, 126, 164, 0.45);
        border-right-color: transparent;
        animation: spinOrbit 1.1s linear infinite;
        z-index: 1;
    }
    .loader-orbit-two {
        width: 2.35rem;
        height: 2.35rem;
        left: 0.74rem;
        margin-top: -1.175rem;
        border-color: rgba(255, 151, 64, 0.55);
        border-top-color: transparent;
        animation-duration: 1.45s;
        animation-direction: reverse;
    }
    .loader-text {
        font-family: 'Space Grotesk', sans-serif;
        color: #103045;
        font-size: 1rem;
        font-weight: 700;
    }
    .loader-subtext {
        color: #4c6b83;
        font-size: 0.84rem;
        margin-top: 0.15rem;
        font-weight: 500;
    }


    @keyframes riseIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes cardPop {
        from { opacity: 0; transform: translateY(7px) scale(0.995); }
        to { opacity: 1; transform: translateY(0) scale(1); }
    }
    @keyframes spinOrbit {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    @keyframes pulseDot {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.18); }
    }
    @keyframes shimmerSweep {
        from { transform: translateX(-100%); }
        to { transform: translateX(100%); }
    }
</style>
""", unsafe_allow_html=True)

# --- Logic ---
@st.cache_resource
def load_domain_sentiment_model():
    """Load domain_sentiment_model.pt once."""
    if not DOMAIN_SENTIMENT_AVAILABLE:
        err = DOMAIN_SENTIMENT_IMPORT_ERROR or "domain_updated import failed"
        return None, f"domain_updated not loaded: {err}"
    model_path = Path(__file__).parent / "domain_sentiment_model.pt"
    if not model_path.exists():
        return None, f"Model file not found: {model_path}"
    try:
        return DomainSentimentPipeline.load(str(model_path)), None
    except Exception as e:
        return None, str(e)


@st.cache_data(ttl=300)
def get_category_data(category: str):
    """Fetch News for a category, filter to market-moving, add Ollama summaries."""
    query = get_search_query_for_category(category)
    news_items = fetch_news(query, limit=5, category=category)
    news_items = filter_market_moving(news_items)
    for item in news_items:
        text = f"{item['title']}. {item.get('snippet', '')}"
        item["summary"] = summarize_with_ollama(text)
    return {"news": news_items, "query": query}


def sentiment_badge(label, score):
    cls = label.upper() if label.upper() in ("POSITIVE", "NEGATIVE", "NEUTRAL") else "NEUTRAL"
    return f'<span class="sentiment-badge sentiment-{cls}">{label} ({score:.0%})</span>'


@st.cache_data(ttl=180)
def fetch_etf_chart_data(symbol: str, days: int = 5):
    """Fetch multi-day intraday ETF prices for chart."""
    try:
        history = yf.Ticker(symbol).history(period=f"{days}d", interval="30m")
        if history is None or history.empty:
            return None, "No market data returned"

        time_col = "Datetime" if "Datetime" in history.reset_index().columns else "Date"
        df = history.reset_index()[[time_col, "Close"]].rename(columns={time_col: "time", "Close": "close"})
        df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
        df = df.dropna(subset=["close"]).copy()
        if df.empty or len(df) < 2:
            return None, "Not enough data points to plot"
        df = df.sort_values("time").drop_duplicates(subset=["time"])
        return df[["time", "close"]], None
    except Exception as e:
        return None, str(e)


def show_loader(label: str, detail: str = "Crunching signals and live feeds..."):
    safe_label = html.escape(label)
    safe_detail = html.escape(detail)
    placeholder = st.empty()
    placeholder.markdown(
        f"""
        <div class="loader-shell">
            <div class="loader-orbit"></div>
            <div class="loader-orbit loader-orbit-two"></div>
            <div class="loader-core"></div>
            <div class="loader-text">{safe_label}</div>
            <div class="loader-subtext">{safe_detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return placeholder

# --- App Flow ---
if 'page' not in st.session_state:
    st.session_state.page = 'home'

if st.session_state.page == 'home':
    st.markdown(
        '<div class="hero-wrap"><div class="hero">'
        '<h1>LexiSnap</h1>'
        '<p>Real-time market intelligence from domain-specific news with fast sentiment signals built for action.</p>'
        '<div class="chip-row">'
        '<span class="chip">Live Sources</span>'
        '<span class="chip">Domain Sentiment</span>'
        '<span class="chip">Market-Moving Filter</span>'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )
    a, b, c = st.columns(3)
    with a:
        st.markdown('<div class="panel"><div class="kicker">Coverage</div><div class="value">4 Critical Domains</div></div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="panel"><div class="kicker">Pipeline</div><div class="value">Fetch → Process → Insight</div></div>', unsafe_allow_html=True)
    with c:
        st.markdown('<div class="panel"><div class="kicker">Model</div><div class="value">Domain-Aware Sentiment</div></div>', unsafe_allow_html=True)

    _, col_btn, _ = st.columns([1, 2, 1])
    with col_btn:
        if st.button("Enter Dashboard →", key="enter_btn", use_container_width=True):
            st.session_state.page = 'dashboard'
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.page == 'dashboard':
    st.markdown('<div class="dashboard-header"><h1>Market Intelligence Dashboard</h1></div>', unsafe_allow_html=True)
    
    if st.button("← Back to Home"):
        st.session_state.page = 'home'
        st.rerun()
    
    st.divider()
    
    domain_icons = {
        "Business and Finance": "📊",
        "Law and Government": "⚖️",
        "Politics": "🏛️",
        "Technology": "💻",
    }

    for cat in CATEGORIES:
        icon = domain_icons.get(cat, "📰")
        with st.expander(f"{icon}  {cat}", expanded=True):
            loader = show_loader(f"Loading {cat}", "Scanning and summarizing market-moving headlines...")
            data = get_category_data(cat)
            loader.empty()
            st.caption(f"Topic: **{data['query']}** | Articles: **{len(data['news'])}**")

            if not data["news"]:
                st.info("No news articles found for this category.")

            for i, item in enumerate(data["news"]):
                title = item.get("title", "No title")
                url = item.get("url", "#")
                raw_summary = item.get("summary") or item.get("snippet", "")
                summary = raw_summary[:200] + ("..." if len(raw_summary) > 200 else "")
                pub = item.get("publishedAt", "")
                if pub:
                    try:
                        dt = datetime.fromisoformat(pub.replace("Z", "+00:00"))
                        pub_str = dt.strftime("%b %d, %Y %H:%M")
                    except Exception:
                        pub_str = pub[:16] if len(pub) > 16 else pub
                else:
                    pub_str = ""

                safe_title = html.escape(title)
                safe_summary = html.escape(summary)
                meta = html.escape(pub_str) if pub_str else "Time unavailable"
                st.markdown(
                    f'<div class="news-card"><div class="news-meta">🕐 {meta}</div>'
                    f'<div><strong>{safe_title}</strong></div>'
                    f'<div class="news-summary">{safe_summary}</div></div>',
                    unsafe_allow_html=True,
                )
                c1, c2 = st.columns([2.2, 1.1])
                with c1:
                    if st.button(f"Analyze: {title[:70]}", key=f"pred_{cat}_{i}", use_container_width=True):
                        st.session_state.selected_article = {
                            "title": title,
                            "url": url,
                            "summary": raw_summary,
                            "category": cat,
                        }
                        st.session_state.page = "prediction"
                        st.rerun()
                with c2:
                    st.link_button("Open article →", url, type="secondary", use_container_width=True)


elif st.session_state.page == "prediction":
    article = st.session_state.get("selected_article")

    if not article:
        st.warning("No article selected. Going back to dashboard.")
        st.session_state.page = "dashboard"
        st.rerun()

    st.markdown('<div class="dashboard-header"><h1>📊 Sentiment Prediction</h1></div>', unsafe_allow_html=True)

    if st.button("← Back to Dashboard"):
        st.session_state.page = "dashboard"
        st.rerun()

    st.markdown(
        f'<div class="panel"><div class="kicker">Selected Article</div>'
        f'<div class="value">{html.escape(article["title"])}</div>'
        f'<div class="news-summary">{html.escape(article.get("summary", "")[:300])}</div></div>',
        unsafe_allow_html=True,
    )
    st.link_button("Open full article", article["url"], type="secondary", use_container_width=False)

    domain = CATEGORY_TO_DOMAIN.get(article["category"], "finance")

    pipeline, load_error = load_domain_sentiment_model()

    if pipeline:
        loader = show_loader("Running domain sentiment model", "Evaluating article context and confidence...")
        text = article["title"] + ". " + article["summary"]
        result = pipeline.predict(text, domain=domain)
        loader.empty()

        score = result["score"]
        label = result["label"]

        c1, c2 = st.columns([1, 1])
        with c1:
            st.metric("Sentiment Score", f"{score:.2f}")
        with c2:
            st.markdown(sentiment_badge(label, score), unsafe_allow_html=True)
            st.caption(f"Domain: **{domain}**")
        st.progress(float(max(0.0, min(1.0, score))))

        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        st.markdown(f"`[{bar}]` 0.0 ← → 1.0")

    else:
        st.error(f"Domain sentiment model not available: {load_error}")

    st.markdown('<div class="dashboard-header"><h1>📈 ETF Market View (Past 5 Days)</h1></div>', unsafe_allow_html=True)
    st.caption("30-minute intraday prices from Yahoo Finance.")
    symbols = ["SPY", "QQQ", "TLT", "GLD"]
    rows = [symbols[:2], symbols[2:]]
    for pair in rows:
        cols = st.columns(2)
        for col, sym in zip(cols, pair):
            with col:
                df, err = fetch_etf_chart_data(sym, days=5)
                if err or df is None:
                    st.warning(f"{sym}: {err}")
                    continue

                latest = float(df["close"].iloc[-1])
                st.metric(f"{sym}", f"${latest:.2f}")

                plot_df = df.copy()
                plot_df["time"] = pd.to_datetime(plot_df["time"]).dt.tz_localize(None)
                plot_df = plot_df.sort_values("time")
                # Use Altair with data-driven y-axis so ups/downs are visible (not flat)
                y_min, y_max = plot_df["close"].min(), plot_df["close"].max()
                padding = (y_max - y_min) * 0.1 or 1
                chart = alt.Chart(plot_df).mark_line(strokeWidth=2.5, point=True).encode(
                    x=alt.X("time:T", title="Date"),
                    y=alt.Y("close:Q", title="Price ($)", scale=alt.Scale(domain=[y_min - padding, y_max + padding], nice=False)),
                ).properties(height=220)
                st.altair_chart(chart, use_container_width=True)
