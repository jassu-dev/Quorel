import yfinance as yf
import numpy as np
import pandas as pd
from .peers import INDUSTRY_PEERS


def _has_valid_target(symbol: str) -> bool:
    """
    Check if a symbol has enough variation in the target variable.
    Target is defined as price movement (up/down).
    """
    try:
        df = yf.download(symbol, period="6mo", interval="1d")
        if df.empty or "Close" not in df:
            return False

        # Create simple target: 1 if price went up, 0 if went down
        df["Return"] = df["Close"].pct_change()
        df = df.dropna()
        df["Target"] = (df["Return"] > 0).astype(int)

        # Check if we have at least 2 unique classes
        return len(np.unique(df["Target"])) > 1
    except Exception:
        return False


def corr_to_rating(corr: float) -> int:
    """
    Convert correlation coefficient into 1–5 rating.
    """
    strength = abs(corr)
    if strength >= 0.8:
        return 5
    elif strength >= 0.6:
        return 4
    elif strength >= 0.4:
        return 3
    elif strength >= 0.2:
        return 2
    else:
        return 1


def get_peers(symbol: str):
    symbol = symbol.upper()

    # First check static INDUSTRY_PEERS dictionary
    for industry, tickers in INDUSTRY_PEERS.items():
        if symbol in tickers:
            peers = tickers
            break
    else:
        peers = []

        # If not found, try sector/industry lookup via yfinance
        try:
            info = yf.Ticker(symbol).info
            industry = info.get("industry", "")
            sector = info.get("sector", "")

            if industry in INDUSTRY_PEERS:
                peers = INDUSTRY_PEERS[industry]
            elif sector in INDUSTRY_PEERS:
                peers = INDUSTRY_PEERS[sector]
        except Exception:
            pass

    # Fallback: just return the ticker itself
    if not peers:
        peers = [symbol]

    # Filter peers that don’t have valid target variation
    valid_peers = [p for p in peers if _has_valid_target(p)]

    # If none are valid, at least return the main symbol
    peers = valid_peers if valid_peers else [symbol]

    # --- New: Calculate influence rating based on correlation ---
    influence = {}
    try:
        base_df = yf.download(symbol, period="6mo", interval="1d")
        base_returns = base_df["Close"].pct_change().dropna()

        for peer in peers:
            if peer == symbol:
                continue
            try:
                peer_df = yf.download(peer, period="6mo", interval="1d")
                peer_returns = peer_df["Close"].pct_change().dropna()

                # align dates
                combined = pd.concat([base_returns, peer_returns], axis=1, join="inner")
                combined.columns = ["base", "peer"]

                if not combined.empty:
                    corr = combined["base"].corr(combined["peer"])
                    influence[peer] = {
                        "corr": round(corr, 2),
                        "rating": corr_to_rating(corr),
                    }
            except Exception:
                influence[peer] = {"corr": None, "rating": None}
    except Exception:
        pass

    return peers, influence
