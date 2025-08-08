import os
import time
import streamlit as st
from langdetect import detect
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from deep_translator import GoogleTranslator
import pandas as pd
import datetime as dt
import requests
import feedparser
import sqlite3
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

st.set_page_config(page_title="AI Company Sentiment (Lite)", page_icon="📊", layout="wide")

# Optional auto-refresh if installed (won't break if missing)
_AUTORF_OK = False
try:
    from streamlit_autorefresh import st_autorefresh
    _AUTORF_OK = True
except Exception:
    _AUTORF_OK = False

NEWSAPI_KEY = st.secrets.get("NEWSAPI_KEY", os.getenv("NEWSAPI_KEY", ""))

with st.sidebar:
    st.header("⚙️ Settings")
    if not NEWSAPI_KEY:
        NEWSAPI_KEY = st.text_input("Enter your NewsAPI key", type="password")
        st.info("Tip: In Streamlit Cloud, set this in App > Settings > Secrets as NEWSAPI_KEY")

    st.subheader("⏱️ Auto-refresh")
    auto_refresh = st.checkbox("Enable auto-refresh", value=False)
    interval_min = st.number_input("Interval (minutes)", min_value=5, max_value=120, value=15, step=5)
    if auto_refresh and _AUTORF_OK:
        st_autorefresh(interval=interval_min * 60 * 1000, key="autorefresh-lite")

    st.subheader("Alphabet Ticker")
    alpha_choice = st.radio(
        "Choose Alphabet share class",
        options=["GOOGL (Class A)", "GOOG (Class C)"],
        index=0,
        horizontal=True,
    )
    selected_alphabet_ticker = "GOOGL" if "GOOGL" in alpha_choice else "GOOG"

    st.subheader("Correlation Window")
    corr_window = st.number_input("Rolling window (days)", min_value=5, max_value=60, value=14, step=1)

    st.subheader("Lag Analysis")
    lag_days = st.number_input(
        "Sentiment lead/lag (days)",
        min_value=-5, max_value=5, value=1, step=1,
    )

analyzer = SentimentIntensityAnalyzer()

DB_PATH = "sentiment_data.db"
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
c = conn.cursor()
c.execute(
    '''CREATE TABLE IF NOT EXISTS sentiment (
        company TEXT,
        date TEXT,
        title TEXT,
        snippet TEXT,
        language TEXT,
        sentiment TEXT,
        score REAL,
        source TEXT,
        url TEXT
    )'''
)
conn.commit()

COMPANIES = ["Palantir", "Nvidia", "Alphabet", "Meta", "Microsoft"]
TICKERS = {"Palantir": "PLTR", "Nvidia": "NVDA", "Alphabet": "GOOGL", "Meta": "META", "Microsoft": "MSFT"}

def translate_to_en(text: str) -> str:
    try:
        return GoogleTranslator(source="auto", target="en").translate(text)
    except Exception:
        return text

def analyze_text(text: str):
    try:
        lang = detect(text)
    except Exception:
        lang = "en"
    raw = text if lang == "en" else translate_to_en(text)
    vs = analyzer.polarity_scores(raw)
    label = "positive" if vs["compound"] > 0.05 else "negative" if vs["compound"] < -0.05 else "neutral"
    score = abs(vs["compound"])
    return lang, label, score

def fetch_newsapi(company: str, page_size: int = 30):
    if not NEWSAPI_KEY:
        return []
    url = "https://newsapi.org/v2/everything"
    params = {"q": company, "pageSize": page_size, "sortBy": "publishedAt", "language": "en", "apiKey": NEWSAPI_KEY}
    try:
        data = requests.get(url, params=params, timeout=20).json()
    except Exception:
        return []
    results = []
    for art in data.get("articles", []):
        title = art.get("title") or ""
        desc = art.get("description") or ""
        snippet = (title + " — " + desc).strip(" —")
        if not snippet:
            continue
        date = (art.get("publishedAt") or "")[:10] or str(dt.date.today())
        url = art.get("url") or ""
        lang, label, score = analyze_text(snippet)
        results.append((company, date, title, desc, lang, label, score, "NewsAPI", url))
    return results

def fetch_google_news_rss(company: str, limit: int = 30):
    feed_url = f"https://news.google.com/rss/search?q={requests.utils.quote(company)}&hl=pt-PT&gl=PT&ceid=PT:pt"
    feed = feedparser.parse(feed_url)
    results = []
    for entry in feed.entries[:limit]:
        title = getattr(entry, "title", "")
        summary = getattr(entry, "summary", "")
        url = getattr(entry, "link", "")
        published = getattr(entry, "published", "")
        date = published[:10] if published else str(dt.date.today())
        snippet = (title + " — " + summary).strip(" —")
        if not snippet:
            continue
        lang, label, score = analyze_text(snippet)
        results.append((company, date, title, summary, lang, label, score, "GoogleRSS", url))
    return results

def upsert_rows(rows):
    for r in rows:
        c.execute("SELECT 1 FROM sentiment WHERE company=? AND date=? AND title=? AND source=?", (r[0], r[1], r[2], r[7]))
        if c.fetchone() is None:
            c.execute(
                "INSERT INTO sentiment (company, date, title, snippet, language, sentiment, score, source, url) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                r,
            )
    conn.commit()

@st.cache_data(show_spinner=False)
def get_stock_history(ticker: str, start, end):
    try:
        data = yf.download(ticker, start=start, end=end, progress=False)
        if not data.empty:
            data = data.reset_index().rename(columns={"Date": "date"})
            data["date"] = pd.to_datetime(data["date"]).dt.tz_localize(None)
        return data
    except Exception:
        return pd.DataFrame()

st.title("📊 AI Company Sentiment Dashboard (Lite)")
st.caption("Lightweight build: VADER sentiment + deep_translator, no heavy ML deps.")

colA, colB, colC = st.columns([2, 1, 1])
with colA:
    st.markdown("**Tracked companies:** " + ", ".join(COMPANIES))
with colB:
    fetch_count = st.number_input("Articles per source per company", min_value=5, max_value=100, value=30, step=5)
with colC:
    run_fetch = st.button("🔄 Fetch latest & analyze")

if "last_auto_fetch" not in st.session_state:
    st.session_state["last_auto_fetch"] = 0.0

def auto_fetch_if_due():
    if not auto_refresh:
        return 0
    now = time.time()
    due = (now - st.session_state["last_auto_fetch"]) >= (interval_min * 60)
    if due:
        all_rows = []
        for comp in COMPANIES:
            all_rows.extend(fetch_newsapi(comp, page_size=fetch_count))
            all_rows.extend(fetch_google_news_rss(comp, limit=fetch_count))
        upsert_rows(all_rows)
        st.session_state["last_auto_fetch"] = now
        st.toast(f"Auto-refresh fetched {len(all_rows)} new items.", icon="🔁")
        return len(all_rows)
    return 0

if run_fetch:
    all_rows = []
    for comp in COMPANIES:
        all_rows.extend(fetch_newsapi(comp, page_size=fetch_count))
        all_rows.extend(fetch_google_news_rss(comp, limit=fetch_count))
    upsert_rows(all_rows)
    st.success(f"Fetched & analyzed {len(all_rows)} items. Stored unique rows in SQLite.")
else:
    auto_fetch_if_due()

df = pd.read_sql_query("SELECT * FROM sentiment", conn, parse_dates=["date"])
if df.empty:
    st.info("No data yet. Click **Fetch latest & analyze** to get started.")
    st.stop()

st.sidebar.subheader("🔎 Filters")
companies_sel = st.sidebar.multiselect("Companies", COMPANIES, default=COMPANIES)
sentiments_sel = st.sidebar.multiselect("Sentiment", sorted(df["sentiment"].dropna().unique().tolist()), default=sorted(df["sentiment"].dropna().unique().tolist()))
date_min = df["date"].min().date()
date_max = df["date"].max().date()
date_range = st.sidebar.date_input("Date range", (date_min, date_max), min_value=date_min, max_value=date_max)

df_f = df[(df["company"].isin(companies_sel)) & (df["sentiment"].isin(sentiments_sel)) & (df["date"].between(pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])))].copy()

st.subheader("🗂️ Articles & Sentiment")
st.dataframe(df_f.sort_values("date", ascending=False)[["date", "company", "sentiment", "score", "title", "language", "source", "url"]], use_container_width=True, hide_index=True)

# Language breakdown
st.subheader("🌍 Sentiment by Language")
lang_counts = (df_f.groupby(["language", "sentiment"]).size().reset_index(name="count").pivot(index="language", columns="sentiment", values="count").fillna(0).astype(int)).sort_index()
if not lang_counts.empty:
    st.dataframe(lang_counts, use_container_width=True)
    st.bar_chart(lang_counts, use_container_width=True, height=300)

# Company comparison
st.subheader("🏁 Company Comparison (filtered period)")
share = df_f.groupby(["company", "sentiment"]).size().reset_index(name="count")
total_by_company = share.groupby("company")["count"].sum().rename("total")
share = share.merge(total_by_company, on="company")
share["share_%"] = (share["count"] / share["total"] * 100).round(1)
share_pivot = share.pivot(index="company", columns="sentiment", values="share_%").fillna(0)
st.dataframe(share_pivot.sort_index(), use_container_width=True)
pos_share = share[share["sentiment"] == "positive"][["company", "share_%"]].set_index("company").sort_values("share_%", ascending=False)
if not pos_share.empty: st.bar_chart(pos_share, height=250, use_container_width=True)
mean_score = df_f.groupby("company")["score"].mean().sort_values(ascending=False).to_frame("mean_confidence")
if not mean_score.empty: st.bar_chart(mean_score, height=250, use_container_width=True)

# Price vs Sentiment (candlestick + sentiment overlay + rolling corr + scatter + lag)
st.subheader("💹 Stock Price (Candlestick) vs Sentiment")
st.caption("Candlestick from Yahoo Finance; overlay shows daily mean sentiment index (-1..1).")
label_to_num = {"positive": 1, "neutral": 0, "negative": -1}
sent_idx = df_f.assign(sent_idx=df_f["sentiment"].map(label_to_num)).dropna(subset=["sent_idx"]).groupby(["company", "date"])["sent_idx"].mean().reset_index()

start_date = pd.to_datetime(date_range[0])
end_date = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)

for comp in companies_sel:
    st.markdown(f"**{comp}**")
    ticker = TICKERS.get(comp)
    if comp == "Alphabet": ticker = selected_alphabet_ticker

    price_df = get_stock_history(ticker, start=start_date, end=end_date)
    if price_df is None or price_df.empty:
        st.warning("No price data found for the selected period.")
        continue

    s_df = sent_idx[sent_idx["company"] == comp].copy()
    s_daily = s_df.groupby("date")["sent_idx"].mean().reindex(price_df["date"]).fillna(method="ffill").fillna(0)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Candlestick(x=price_df["date"], open=price_df["Open"], high=price_df["High"], low=price_df["Low"], close=price_df["Close"], name="Price"), secondary_y=False)
    fig.add_trace(go.Scatter(x=price_df["date"], y=s_daily.values, mode="lines", name="Sentiment Index"), secondary_y=True)
    fig.update_yaxes(title_text="Price", secondary_y=False)
    fig.update_yaxes(title_text="Sentiment Index", range=[-1, 1], secondary_y=True)
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Rolling correlation
    ret = price_df["Close"].pct_change()
    corr = ret.rolling(int(corr_window)).corr(pd.Series(s_daily.values, index=price_df["date"]).pct_change())
    corr_fig = go.Figure()
    corr_fig.add_trace(go.Scatter(x=price_df["date"], y=corr, mode="lines", name="Rolling Corr"))
    corr_fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=250)
    corr_fig.update_yaxes(title_text="Correlation (-1..1)", range=[-1, 1])
    st.plotly_chart(corr_fig, use_container_width=True)

    # Scatter: returns vs sentiment
    ret_aligned = ret.reindex(price_df["date"]).values
    sent_aligned = pd.Series(s_daily.values, index=price_df["date"]).values
    mask = ~(pd.isna(ret_aligned) | pd.isna(sent_aligned))
    x = sent_aligned[mask]; y = ret_aligned[mask]
    if len(x) > 2:
        a, b = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 50); y_line = a * x_line + b
        pearson = pd.Series(x).corr(pd.Series(y), method="pearson")
        spearman = pd.Series(x).corr(pd.Series(y), method="spearman")
        sc_fig = go.Figure()
        sc_fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Points", opacity=0.7))
        sc_fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines", name="Fit (OLS)"))
        sc_fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=280)
        sc_fig.update_xaxes(title_text="Sentiment Index (-1..1)"); sc_fig.update_yaxes(title_text="Daily Return")
        st.plotly_chart(sc_fig, use_container_width=True)
        st.caption(f"Pearson r = {pearson:.3f} • Spearman ρ = {spearman:.3f}")
    else:
        st.info("Not enough aligned data points for scatter analysis.")

    # Lag analysis
    sent_series = pd.Series(s_daily.values, index=price_df["date"])
    ret_series = price_df["Close"].pct_change()
    if int(lag_days) > 0:
        x_lag = sent_series.shift(int(lag_days)); y_lag = ret_series
    elif int(lag_days) < 0:
        x_lag = sent_series; y_lag = ret_series.shift(abs(int(lag_days)))
    else:
        x_lag = sent_series; y_lag = ret_series
    align = pd.concat([x_lag, y_lag], axis=1, keys=["sent", "ret"]).dropna()
    if not align.empty and len(align) > 5:
        px = align["sent"].values; py = align["ret"].values
        try:
            a_lag, b_lag = np.polyfit(px, py, 1)
            xfit = np.linspace(px.min(), px.max(), 50); yfit = a_lag * xfit + b_lag
        except Exception:
            xfit, yfit = None, None
        pear_lag = pd.Series(px).corr(pd.Series(py), method="pearson")
        spear_lag = pd.Series(px).corr(pd.Series(py), method="spearman")
        lag_fig = go.Figure()
        lag_fig.add_trace(go.Scatter(x=px, y=py, mode="markers", name="Lagged Points", opacity=0.7))
        if xfit is not None: lag_fig.add_trace(go.Scatter(x=xfit, y=yfit, mode="lines", name="Lagged Fit (OLS)"))
        lag_fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=280)
        lag_fig.update_xaxes(title_text=f"Sentiment Index (shift={int(lag_days)})"); lag_fig.update_yaxes(title_text="Daily Return")
        st.plotly_chart(lag_fig, use_container_width=True)
        st.caption(f"Lagged Pearson r = {pear_lag:.3f} • Lagged Spearman ρ = {spear_lag:.3f}")
    else:
        st.info("Not enough data points for lag analysis with current settings.")

st.success("Ready. Use the sidebar to filter. Lite build avoids heavy ML dependencies.")