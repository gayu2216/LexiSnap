# LexiSnap

Real-time market intelligence from domain-specific news with sentiment analysis and ETF charts.

## Features

- **4 domains**: Business & Finance, Law & Government, Politics, Technology
- **Market-moving filter**: Keywords + Ollama Mistral to surface relevant news
- **Domain sentiment model**: Multi-domain RoBERTa adapter for finance/tech/law/politics
- **ETF charts**: SPY, QQQ, TLT, GLD (last 10 days)

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment variables

Copy `.env.example` to `.env` and add your API keys:

- **SERPAPI_API_KEY** – [serpapi.com](https://serpapi.com) (for Google Trends)
- **NEWS_API_KEY** – [newsapi.org](https://newsapi.org)

### 3. Train the domain sentiment model

The model file (`domain_sentiment_model.pt`) is too large for GitHub. Train it locally:

```bash
python domain_updated.py
```

This creates `domain_sentiment_model.pt` in the project root.

### 4. (Optional) Ollama for summarization

For Mistral summaries, run [Ollama](https://ollama.com) with Mistral:

```bash
ollama pull mistral
ollama serve
```

## Run

```bash
./run_app.sh
```

Or:

```bash
streamlit run app.py
```

## Project structure

- `app.py` – Streamlit app
- `data_sources.py` – News API, filters, Ollama summarization
- `domain_updated.py` – Domain sentiment model (train & inference)
- `domain_sentiment_model.pt` – Trained model (create via step 3)
