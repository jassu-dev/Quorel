import yfinance as yf
import requests
from bs4 import BeautifulSoup
import pandas as pd
import feedparser
import datetime

EXPECTED = ["Open","High","Low","Close","Volume"]
def _ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Make sure standard OHLCV columns exist; build them from what's available."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["Date"] + EXPECTED)

    df = df.copy()

    # Fix Close if only Adj Close present
    if "Adj Close" in df.columns and "Close" not in df.columns:
        df["Close"] = df["Adj Close"]

    # Fill missing columns
    for col in EXPECTED:
        if col not in df.columns:
            if col in ("Open", "High", "Low", "Close"):
                if "Close" in df.columns:
                    df[col] = df["Close"]
                else:
                    df[col] = 0.0
            elif col == "Volume":
                df[col] = 0

    # Enforce numeric safely
# Enforce numeric safely
    for col in ["Open", "High", "Low", "Close"]:
        if col in df.columns:
            if isinstance(df[col], (pd.Series, list)):
                df[col] = (
                    pd.to_numeric(df[col], errors="coerce")
                    .ffill()
                    .bfill()
                    .fillna(0.0)
                )
            else:
                df[col] = pd.Series([0.0] * len(df))

    if "Volume" in df.columns:
        if isinstance(df["Volume"], (pd.Series, list)):
            df["Volume"] = (
                pd.to_numeric(df["Volume"], errors="coerce").fillna(0).astype(int)
            )
        else:
            df["Volume"] = pd.Series([0] * len(df))

    # Ensure Date column
    if "Date" not in df.columns and getattr(df.index, "name", None) is not None:
        df = df.reset_index()

    # Keep only relevant
    return df[[c for c in (["Date"] + EXPECTED) if c in df.columns]]

def fetch_yahoo_finance(ticker="AAPL"):
    """Fetch OHLCV for last 60 days daily, robust to missing columns."""
    try:
        data = yf.download(
            ticker, period="2mo", interval="1d", auto_adjust=False, progress=False
        )
        if data is None or data.empty:
            print(f"[WARN] No data returned for {ticker}")
            return pd.DataFrame(columns=["Date"] + EXPECTED)

        # Handle MultiIndex (e.g., ("Close", "AAPL"))
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]

        data = data.reset_index()
    except Exception as e:
        print(f"[ERROR] yfinance error for {ticker}:", e)
        data = pd.DataFrame(columns=["Date"] + EXPECTED)

    return _ensure_ohlcv(data)

def _rss_headlines(url):
    try:
        feed = feedparser.parse(url)
        return [e.get("title","").strip() for e in feed.entries][:20]
    except Exception:
        return []

def _scrape_simple_headlines(url, selector):
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        items = [el.get_text(strip=True) for el in soup.select(selector)]
        return [h for h in items if h]
    except Exception:
        return []

def fetch_news_headlines(ticker: str = "AAPL", limit: int = 10):
    """
    Return recent news headlines for a ticker.
    Falls back to general market/business news if ticker-specific fails.
    Always returns list of dicts with {date, headline}.
    """
    headlines = []

    # Try Yahoo Finance RSS for ticker
    feeds = [
        f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
        "https://feeds.reuters.com/reuters/businessNews",
        "https://feeds.reuters.com/reuters/marketsNews",
    ]

    for url in feeds:
        headlines += _rss_headlines(url)

    if not headlines:
        # Fallback HTML scraping
        headlines = _scrape_simple_headlines("https://www.reuters.com/markets/", "h2, h3")
        if not headlines:
            headlines = _scrape_simple_headlines("https://finance.yahoo.com", "h2, h3")

    # Deduplicate + clean
    seen = set()
    uniq = []
    today = datetime.date.today().isoformat()

    for h in headlines:
        if not h:
            continue
        if h in seen:
            continue
        if len(h) < 20:
            continue
        uniq.append({"date": today, "headline": h})
        seen.add(h)
        if len(uniq) >= limit:
            break

    if not uniq:
        uniq = [{"date": today, "headline": f"No recent headlines found for {ticker}"}]

    return uniq
# data_pipeline/ingest.py

def fetch_stock_data(symbol, start="2020-01-01", end=None):
    df = yf.Ticker(symbol).history(start=start, end=end)
    df["Range"] = df["High"] - df["Low"]
    df["Volatility5"] = df["Close"].pct_change().rolling(5).std()
    df["Momentum5"] = df["Close"].pct_change(periods=5)
    return df
