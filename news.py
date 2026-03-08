import requests
import feedparser
import pandas as pd
import urllib.parse
import os
import time
import random
import trafilatura
from serpapi import GoogleSearch

# --- CONFIGURATION ---
SERPAPI_KEY = "9bdb9d8d83a93b38fa2eb16730baa6fe5a1fd399a8d8b5e371ff5bae70ce561d"
NEWS_API_KEY = "2e1e7b09472b49a8bcf0d205c2721cf2"
OUTPUT_FILE = "nlp_dataset.csv"

# ==========================================
# 1. GET MOST TRENDING TOPIC FROM GOOGLE
# ==========================================

def get_top_trend():
    params = {
        "engine": "google_trends_trending_now",
        "geo": "US",
        "api_key": SERPAPI_KEY
    }
    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        trending_searches = results.get("trending_searches", [])

        if trending_searches:
            top_trend = trending_searches[0]
            query = top_trend.get("query")
            categories_list = top_trend.get("categories", [])
            category_names = [cat.get("name") for cat in categories_list if cat.get("name")]
            category_label = ", ".join(category_names) if category_names else "General"

            print(f"\n🔥 Most Active Trend: {query}")
            print(f"📂 Category: {category_label}")
            return query
        else:
            print("⚠️ No trending data found. Falling back to default query.")
            return "Technology"
    except Exception as e:
        print(f"SerpAPI error: {e}. Falling back to default query.")
        return "Technology"


# ==========================================
# 2. HELPER: EXTRACT FULL ARTICLE TEXT
# ==========================================

def get_full_text(url):
    """Uses trafilatura to extract clean body text from a URL."""
    try:
        downloaded = trafilatura.fetch_url(url)
        return trafilatura.extract(downloaded) if downloaded else ""
    except Exception:
        return ""


# ==========================================
# 3. FETCH REDDIT DATA (JSON endpoint)
# ==========================================

def fetch_reddit_json(query):
    """Fetches Reddit posts via the JSON search endpoint (more reliable than RSS)."""
    encoded = urllib.parse.quote(query)
    url = f"https://www.reddit.com/search.json?q={encoded}&sort=new&limit=5"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.5",
        "DNT": "1"
    }

    try:
        time.sleep(random.uniform(2, 4))
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code == 200:
            posts = response.json().get("data", {}).get("children", [])
            data = []
            for p in posts:
                item = p.get("data", {})
                full_url = f"https://www.reddit.com{item.get('permalink', '')}"
                data.append({
                    "Source": "Reddit",
                    "Trend": query,
                    "Title": item.get("title", ""),
                    "URL": full_url,
                    "Content": get_full_text(full_url),
                    "Timestamp": item.get("created_utc", "")
                })
            print(f"  ✅ Reddit: {len(data)} posts collected")
            return data
        else:
            print(f"  ⚠️ Reddit returned status code: {response.status_code}")
    except Exception as e:
        print(f"  ❌ Reddit error: {e}")

    return []


# ==========================================
# 4. FETCH NEWS DATA (NewsAPI)
# ==========================================

def fetch_news_data(query):
    """Fetches news articles from NewsAPI with full content extraction."""
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "apiKey": NEWS_API_KEY,
        "pageSize": 5,
        "sortBy": "relevancy",
        "language": "en"
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            articles = response.json().get("articles", [])
            data = [{
                "Source": "NewsAPI",
                "Trend": query,
                "Title": a.get("title", ""),
                "URL": a.get("url", ""),
                "Content": get_full_text(a.get("url", "")),
                "Timestamp": a.get("publishedAt", "")
            } for a in articles]
            print(f"  ✅ NewsAPI: {len(data)} articles collected")
            return data
        else:
            print(f"  ⚠️ NewsAPI returned status code: {response.status_code}")
    except Exception as e:
        print(f"  ❌ NewsAPI error: {e}")

    return []


# ==========================================
# 5. SAVE TO CSV
# ==========================================

def save_to_csv(data, filepath=OUTPUT_FILE):
    """Appends collected data to CSV, writing header only if file doesn't exist."""
    if not data:
        print("⚠️ No data to save.")
        return

    df = pd.DataFrame(data)
    file_exists = os.path.isfile(filepath)
    df.to_csv(filepath, mode="a", index=False, header=not file_exists)
    print(f"\n💾 Saved {len(df)} records to '{filepath}'")


# ==========================================
# 6. MAIN PIPELINE
# ==========================================

def main():
    print("=" * 50)
    print("       📡 TREND DATA COLLECTION PIPELINE")
    print("=" * 50)

    # Step 1: Get trending query
    query = get_top_trend()

    # Step 2: Collect from all sources
    print(f"\n🔎 Collecting content for: '{query}'\n")
    all_data = []

    reddit_data = fetch_reddit_json(query)
    news_data = fetch_news_data(query)

    all_data = reddit_data + news_data

    # Step 3: Save combined results
    save_to_csv(all_data)

    # Step 4: Summary
    print(f"\n📊 Summary:")
    print(f"   Reddit posts : {len(reddit_data)}")
    print(f"   News articles: {len(news_data)}")
    print(f"   Total records: {len(all_data)}")
    print("=" * 50)


if __name__ == "__main__":
    main()