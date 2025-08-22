import pandas as pd

NEEDED = ["Open", "High", "Low", "Close", "Volume"]

def clean_financial_data(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=NEEDED)
    
    # only keep expected columns if present
    keep = [c for c in df.columns if c in (["Date"] + NEEDED)]
    df = df[keep].copy()
    
    # drop rows where all OHLCV are missing
    df = df.dropna(how="all", subset=[c for c in NEEDED if c in df.columns])
    
    # if any columns missing, create them with safe defaults
    for col in NEEDED:
        if col not in df.columns:
            df[col] = 0.0 if col != "Volume" else 0
    
    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    if "Close" in df.columns:
        df = df.copy()
        
        # Daily returns
        df["Return"] = df["Close"].pct_change().fillna(0.0)
        
        # Daily range (high-low relative to close)
        if "High" in df.columns and "Low" in df.columns:
            df["Range"] = (df["High"] - df["Low"]) / df["Close"].replace(0, 1)
        else:
            df["Range"] = 0.0
        
        # Rolling volatility (5-day)
        df["Volatility5"] = df["Return"].rolling(5, min_periods=1).std().fillna(0.0)
        
        # 5-day momentum
        df["Momentum5"] = df["Close"].pct_change(5).fillna(0.0)
    
    return df
