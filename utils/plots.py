# utils/plots.py
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PLOT_DIR = os.path.join("static", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

def plot_feature_importances(ticker, importances: dict):
    """
    Bar chart of feature importances.
    Saves: static/plots/{ticker}_features.png
    """
    keys = list(importances.keys())
    vals = [importances[k] for k in keys]
    fig, ax = plt.subplots(figsize=(6,3))
    ax.barh(keys[::-1], vals[::-1])  # horizontal, descending
    ax.set_title(f"{ticker} - Feature importances")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"{ticker}_features.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path

def plot_price_with_mas(ticker, df, short=5, long=30):
    """
    Line chart of Close price with short/long moving averages.
    Saves: static/plots/{ticker}_price.png
    """
    if df is None or df.empty or "Close" not in df.columns:
        return None
    s = df["Close"].astype(float)
    ma_short = s.rolling(short, min_periods=1).mean()
    ma_long = s.rolling(long, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(7,3))
    ax.plot(s.index, s.values, label="Close", linewidth=1)
    ax.plot(s.index, ma_short.values, label=f"{short}-day MA", linewidth=1)
    ax.plot(s.index, ma_long.values, label=f"{long}-day MA", linewidth=1)
    ax.set_title(f"{ticker} Price & MAs")
    ax.set_ylabel("Price")
    ax.legend(loc="best", fontsize="small")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"{ticker}_price.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path

def plot_event_sentiment(ticker, pos, neu, neg):
    plt.figure(figsize=(5,4))
    labels = ["Positive", "Neutral", "Negative"]
    values = [pos, neu, neg]

    plt.bar(labels, values, color=["green", "gray", "red"])
    plt.title(f"News Sentiment for {ticker}")
    plt.ylabel("Number of Headlines")

    plot_path = os.path.join("static", "plots", f"{ticker}_sentiment.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    return plot_path
