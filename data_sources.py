"""
Data sources for LexiSnap: Google Trends, News, Ollama Mistral summarization.
"""

import os
from datetime import datetime, timedelta
import requests

# Load .env if present (for local dev)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# --- Config (set via env vars: SERPAPI_API_KEY, NEWS_API_KEY) ---
SERPAPI_KEY = os.environ.get("SERPAPI_API_KEY", "")
NEWS_API_KEY = os.environ.get("NEWS_API_KEY", "")

CATEGORIES = ["Business and Finance", "Law and Government", "Politics", "Technology"]

# NewsAPI top-headlines category mapping
NEWSAPI_CATEGORIES = {
    "Business and Finance": "business",
    "Law and Government": "general",
    "Politics": "general",
    "Technology": "technology",
}

# Domain-specific keywords for news search (no Google Trends - avoids sports/entertainment)
CATEGORY_SEARCH_TERMS = {
    "Business and Finance": "stock market Fed earnings economy finance",
    "Law and Government": "law regulation court government policy",
    "Politics": "politics election congress policy",
    "Technology": "technology AI software tech",
}

# Exclude from Business and Finance (sports/entertainment)
BUSINESS_EXCLUDE = ["hockey", "basketball", "nba", "nhl", "mlb", "shootout", "winning streak", "beat ", " vs ", "avalanche", "stars "]


def _filter_off_topic(items: list[dict], category: str) -> list[dict]:
    """Remove sports/entertainment from Business and Finance."""
    if "business" not in category.lower() and "finance" not in category.lower():
        return items
    filtered = [i for i in items if not any(
        ex in f"{i.get('title','')} {i.get('snippet','')}".lower() for ex in BUSINESS_EXCLUDE
    )]
    return filtered


def get_search_query_for_category(category: str) -> str:
    """Return domain-specific keywords only. No Google Trends."""
    return CATEGORY_SEARCH_TERMS.get(category, category)


# Keywords that suggest market-moving news (stocks, bonds, economy, policy)
MARKET_MOVING_KEYWORDS = [
    "fed", "interest rate", "earnings", "gdp", "inflation", "recession",
    "stock", "market", "ipo", "merger", "acquisition", "layoff",
    "regulation", "sec", "tariff", "trade", "economy", "revenue",
    "profit", "loss", "forecast", "guidance", "dividend", "bond",
    "oil", "commodity", "crypto", "bitcoin", "central bank",
    "employment", "jobs report", "consumer", "retail sales",
]


def _keyword_market_moving(text: str) -> bool:
    """Fast keyword check for market-moving content."""
    t = text.lower()
    return any(kw in t for kw in MARKET_MOVING_KEYWORDS)


def _ollama_market_moving(title: str) -> bool:
    """Use Ollama to classify if headline is market-moving."""
    try:
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral",
                "prompt": f"""Does this news headline have potential to move financial markets (stocks, bonds, commodities)?
Consider: earnings, Fed policy, regulations, economic data, major company news, geopolitical events.
Headline: "{title}"
Answer only YES or NO.""",
                "stream": False,
            },
            timeout=15,
        )
        if r.status_code == 200:
            ans = r.json().get("response", "").strip().upper()
            return "YES" in ans[:10]
    except Exception:
        pass
    return False


def is_market_moving(title: str, snippet: str = "") -> bool:
    """Check if headline is likely to move financial markets."""
    text = f"{title} {snippet}"
    if _keyword_market_moving(text):
        return True
    return _ollama_market_moving(title)


def fetch_news(query: str, limit: int = 5, category: str = "") -> list[dict]:
    """Fetch latest news from NewsAPI. Tries 7d, 30d, then no date filter."""
    url = "https://newsapi.org/v2/everything"
    date_ranges = [
        (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d"),
        (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d"),
        None,  # No date filter - get most recent available
    ]
    for from_date in date_ranges:
        params = {
            "q": query,
            "apiKey": NEWS_API_KEY,
            "pageSize": limit,
            "sortBy": "publishedAt",
            "language": "en",
        }
        if from_date:
            params["from"] = from_date
        try:
            r = requests.get(url, params=params, timeout=10)
            if r.status_code != 200:
                continue
            data = r.json()
            if data.get("status") == "error":
                continue
            articles = data.get("articles", [])
            out = []
            for a in articles:
                if a.get("title") and a.get("url"):  # Skip removed/deleted
                    out.append({
                        "source": "News",
                        "title": a.get("title", ""),
                        "url": a.get("url", ""),
                        "snippet": (a.get("description") or "")[:300],
                        "publishedAt": a.get("publishedAt", ""),
                    })
            if out:
                filtered = _filter_off_topic(out, category)
                if filtered:
                    return filtered
        except Exception:
            continue
    # Fallback: top headlines (no query needed, often has results)
    cat = NEWSAPI_CATEGORIES.get(category, "general")
    try:
        r = requests.get(
            "https://newsapi.org/v2/top-headlines",
            params={"apiKey": NEWS_API_KEY, "country": "us", "category": cat, "pageSize": limit},
            timeout=10,
        )
        if r.status_code == 200:
            articles = r.json().get("articles", [])
            out = [{
                "source": "News",
                "title": a.get("title", ""),
                "url": a.get("url", ""),
                "snippet": (a.get("description") or "")[:300],
                "publishedAt": a.get("publishedAt", ""),
            } for a in articles if a.get("title") and a.get("url")]
            return _filter_off_topic(out, category)
    except Exception:
        pass
    return []


def filter_market_moving(news_items: list[dict]) -> list[dict]:
    """Keep only headlines that might move the market. Returns top 1."""
    # Keyword pass (fast, no API)
    for n in news_items:
        if _keyword_market_moving(f"{n.get('title', '')} {n.get('snippet', '')}"):
            return [n]
    # No keyword match: use Ollama on top item only (1 API call)
    if news_items and _ollama_market_moving(news_items[0].get("title", "")):
        return [news_items[0]]
    return news_items[:1]  # Fallback: show top 1 anyway


def summarize_with_ollama(text: str, max_chars: int = 500) -> str:
    """Summarize text using Ollama Mistral. Returns original if Ollama unavailable."""
    if not text or len(text.strip()) < 20:
        return text or "No content."
    prompt = f"Summarize this in 1-2 short sentences (max 100 words):\n\n{text[:1500]}"
    try:
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "mistral", "prompt": prompt, "stream": False},
            timeout=30,
        )
        if r.status_code == 200:
            out = r.json().get("response", "").strip()
            return out[:max_chars] if out else text[:max_chars]
    except Exception:
        pass
    return text[:max_chars] + ("..." if len(text) > max_chars else "")
