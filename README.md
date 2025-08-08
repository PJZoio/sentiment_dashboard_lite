# 📊 AI Company Sentiment Dashboard (Lite)

**Deployment-friendly** build that avoids heavy ML dependencies. Uses **VADER** + **deep-translator** to handle multilingual sentiment by translating text to English before scoring.

## Install & Run
```bash
pip install -r requirements.txt
export NEWSAPI_KEY=your_key_here   # or set in Streamlit secrets
streamlit run app.py
```

## Notes
- Sources: **NewsAPI** + **Google News RSS**
- Storage: **SQLite** (`sentiment_data.db`)
- Charts: candlestick with sentiment overlay, rolling correlation, returns vs sentiment scatter, lag analysis