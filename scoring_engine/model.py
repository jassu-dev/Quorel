from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

FEATURES = ["Open","High","Low","Close","Volume","Range","Volatility5","Momentum5"]

class CreditScoringModel:
    def __init__(self, max_depth=4, random_state=42):
        self.model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
        self.scaler = StandardScaler()
        self.fitted = False
        self.timeline = []   # store scores over time

    def _prepare(self, df: pd.DataFrame):
        df = df.copy()
        for col in FEATURES:
            if col not in df.columns:
                df[col] = 0.0
        X = df[FEATURES].fillna(0.0).values
        if "Return" in df.columns:
            y = (df["Return"].shift(-1).fillna(0.0) >= 0).astype(int).values
        else:
            y = (pd.Series([0]*len(df))).values
        return X, y

    def fit(self, df: pd.DataFrame, symbol: str = None):
        if df is None or df.empty or len(df) < 10:
            raise ValueError(f"{symbol if symbol else 'Data'} → Not enough data to fit.")
        X, y = self._prepare(df)
        if len(set(y)) < 2:
            raise ValueError(f"{symbol if symbol else 'Data'} → Only one class found in target variable.")
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs, y)
        self.fitted = True

    def score(self, df: pd.DataFrame, return_details=False):
        if not self.fitted:
            raise ValueError("Model not trained")
        X, _ = self._prepare(df)
        Xs = self.scaler.transform(X)

        proba = self.model.predict_proba(Xs)
        if proba.shape[1] == 1:
            p = float(proba[-1, 0])
        else:
            p = float(proba[-1, 1])

        score = round(p * 100.0, 1)
        importances = {f: float(w) for f, w in zip(FEATURES, self.model.feature_importances_)}

        if return_details:
            return score, p, importances
        return score

    def get_timeline(self):
        """Return timeline of creditworthiness scores"""
        return self.timeline
