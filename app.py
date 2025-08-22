import os
import smtplib
import sqlite3
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for, session
from flask_bcrypt import Bcrypt
from flask_mail import Mail, Message
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import time
from data_pipeline.ingest import fetch_yahoo_finance, fetch_news_headlines
from data_pipeline.preprocess import clean_financial_data, feature_engineering
from scoring_engine.model import CreditScoringModel
from nlu_events.event_extractor import extract_events_with_sentiment
from utils.explain import make_plain_language_explanation
from scoring_engine.peer_comparison import get_peers
import markdown
from werkzeug.exceptions import Forbidden
import yfinance as yf
# -----------------------
# Flask setup
# -----------------------

app = Flask(__name__)
app.secret_key = "supersecretkey"
bcrypt = Bcrypt(app)

DB_PATH = "score_history.db"
USER_DB = "users.db"

# Email config
app.config.update(
    MAIL_SERVER="smtp.gmail.com",
    MAIL_PORT=587,
    MAIL_USE_TLS=True,
    MAIL_USERNAME="quorelnoreply@gmail.com",
    MAIL_PASSWORD="yjya nmmj uoul ijij"
)
mail = Mail(app)

def init_score_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS score_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            time TEXT NOT NULL,
            score REAL NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def init_top_scores():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS top_scores (
            ticker TEXT PRIMARY KEY,
            score REAL NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()
init_top_scores()



def init_user_db():
    conn = sqlite3.connect(USER_DB)
    cur = conn.cursor()
    
    # --- Users table ---
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            is_verified INTEGER DEFAULT 0,
            verification_token TEXT,
            reset_token TEXT,
            reset_expiry TEXT
        )
    """)
    conn.commit()
    conn.close()
init_score_db()
init_user_db()


def init_admin_db():
    conn = sqlite3.connect(USER_DB)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS admins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            designation TEXT DEFAULT 'Admin'
        )
    """)
    # ensure super admin always exists
    cur.execute("INSERT OR IGNORE INTO admins (email, designation) VALUES (?, ?)", 
                ("maddalajashwanth69@gmail.com", "Co-Founder"))
    conn.commit()
    conn.close()

# call it on startup
init_admin_db()



def init_ultra_notice_db():
    conn = sqlite3.connect(USER_DB)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS ultra_notice (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            ticker1 TEXT,
            ticker2 TEXT,
            ticker3 TEXT,
            threshold REAL DEFAULT 5,
            direction TEXT DEFAULT 'up',
            active INTEGER DEFAULT 1
        )
    """)
    conn.commit()
    conn.close()

init_ultra_notice_db()




# -----------------------
# Forecast helpers
# -----------------------

def forecast_prices(df, steps=5):
    """
    Plain ARIMA on Close using all available data.
    Returns list of floats or None.
    """
    if "Close" not in df.columns or len(df) < 30:
        return None
    series = df["Close"].astype(float)
    try:
        model = ARIMA(series, order=(5, 1, 0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=steps)
        return forecast.tolist()
    except Exception as e:
        print("Forecast error:", e)
        return None

def forecast_scores(ticker, steps=5):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT score FROM score_history WHERE ticker=? ORDER BY time ASC", (ticker,))
    rows = [r[0] for r in cur.fetchall()]
    conn.close()

    if len(rows) < 10:
        return None

    try:
        model = ARIMA(rows, order=(2, 1, 2))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=steps)
        return forecast.tolist()
    except Exception as e:
        print("Score forecast error:", e)
        return None

# ---------- News-aware helpers (for interactive forecast chart) ----------

def daily_sentiment_series(ticker, start_date=None, end_date=None):
    """
    Build a daily sentiment score series (mean compound per date).
    If no news that day → 0.0. Returns a pandas Series indexed by date (YYYY-MM-DD).
    """
    headlines = fetch_news_headlines(ticker)
    events = extract_events_with_sentiment(headlines) if headlines else []

    if not events:
        return pd.Series(dtype=float)

    rows = []
    for e in events:
        if not isinstance(e, dict):
            continue
        t = e.get("time") or datetime.utcnow().isoformat()
        date_key = str(t)[:10]
        score = float(e.get("sentiment_score", 0.0))
        rows.append({"date": date_key, "score": score})

    if not rows:
        return pd.Series(dtype=float)

    df_ev = pd.DataFrame(rows)
    daily = df_ev.groupby("date")["score"].mean().sort_index()
    if start_date:
        daily = daily[daily.index >= str(start_date)]
    if end_date:
        daily = daily[daily.index <= str(end_date)]
    return daily

def build_exog(price_df, ticker, steps):
    """
    Create exogenous features aligned to Close series:
    - 7D realized volatility (rolling std of returns)
    - daily sentiment (filled missing with 0, 7D rolling mean)
    Returns exog_train (len = len(price_df)) and exog_future (len = steps)
    """
    close = price_df["Close"].astype(float).copy()
    rets = close.pct_change()
    vol7 = rets.rolling(7).std().fillna(0.0)

    # news → daily & align to price index
    sent_daily = daily_sentiment_series(ticker)
    idx_dates = price_df.index.astype(str)
    sent_aligned = pd.Series(0.0, index=idx_dates)
    if not sent_daily.empty:
        # reindex with fill_value=0.0 then clip to index
        sent_aligned.loc[sent_daily.index.intersection(sent_aligned.index)] = sent_daily.reindex(
            sent_aligned.index, fill_value=0.0
        )
    sent7 = sent_aligned.rolling(7).mean().fillna(0.0)

    exog_train = pd.DataFrame({"vol7": vol7.values, "sent7": sent7.values}, index=price_df.index)

    # Future exog: keep vol steady, decay sentiment
    last_vol = float(exog_train["vol7"].iloc[-1]) if len(exog_train) else 0.0
    last_sent = float(exog_train["sent7"].iloc[-1]) if len(exog_train) else 0.0
    decay = np.exp(-0.5 * np.arange(1, steps + 1))
    future_vol = np.full(steps, last_vol)
    future_sent = last_sent * decay

    exog_future = pd.DataFrame({"vol7": future_vol, "sent7": future_sent})
    return exog_train, exog_future

def forecast_prices_news_aware(ticker, price_df, steps=5):
    """
    SARIMAX with exogenous vars (volatility & news sentiment).
    Returns dict: {"forecast": list, "lower": list, "upper": list}
    """
    if "Close" not in price_df.columns or len(price_df) < 40:
        return {"forecast": [], "lower": [], "upper": []}

    series = price_df["Close"].astype(float)
    exog_train, exog_future = build_exog(price_df, ticker, steps)

    try:
        model = SARIMAX(
            series,
            order=(5, 1, 1),
            trend="c",
            exog=exog_train,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        fit = model.fit(disp=False)
        pred = fit.get_forecast(steps=steps, exog=exog_future)
        mean = pred.predicted_mean.tolist()
        conf = pred.conf_int(alpha=0.2)  # 80% CI
        lower = conf.iloc[:, 0].tolist()
        upper = conf.iloc[:, 1].tolist()
        return {"forecast": mean, "lower": lower, "upper": upper}
    except Exception as e:
        print("SARIMAX error:", e)
        return {"forecast": [], "lower": [], "upper": []}

# -----------------------
# Auth routes
# -----------------------

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if "user" in session:
        return redirect(url_for("index"))
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        confirm = request.form["confirm_password"]

        if password != confirm:
            flash("❌ Passwords do not match", "error")
            return redirect(url_for("signup"))

        pw_hash = bcrypt.generate_password_hash(password).decode("utf-8")
        token = os.urandom(16).hex()

        try:
            conn = sqlite3.connect(USER_DB)
            cur = conn.cursor()
            cur.execute("INSERT INTO users (email, password_hash, verification_token) VALUES (?, ?, ?)",
                        (email, pw_hash, token))
            conn.commit()
            conn.close()

            link = url_for("verify_email", token=token, _external=True)
            msg = Message("Verify your Quorel account", sender=app.config["MAIL_USERNAME"], recipients=[email])
            msg.body = f"Click to verify your account: {link}"
            mail.send(msg)

            flash("✅ Account created! Check your email to verify.", "success")
            return redirect(url_for("login"))
        except Exception:
            flash("❌ Email already exists", "error")
            return redirect(url_for("signup"))
    return render_template("signup.html")

@app.route("/verify/<token>")
def verify_email(token):
    conn = sqlite3.connect(USER_DB)
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE verification_token=?", (token,))
    row = cur.fetchone()
    if row:
        cur.execute("UPDATE users SET is_verified=1, verification_token=NULL WHERE id=?", (row[0],))
        conn.commit()
        conn.close()
        flash("✅ Email verified! You can now log in.", "success")
    else:
        flash("❌ Invalid or expired verification link.", "error")
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if "user" in session:
        return redirect(url_for("index"))
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        conn = sqlite3.connect(USER_DB)
        cur = conn.cursor()
        cur.execute("SELECT id, password_hash, is_verified FROM users WHERE email=?", (email,))
        user = cur.fetchone()
        conn.close()

        if not user:
            flash("❌ Invalid credentials", "error")
            return redirect(url_for("login"))

        uid, pw_hash, verified = user
        if not bcrypt.check_password_hash(pw_hash, password):
            flash("❌ Invalid credentials", "error")
            return redirect(url_for("login"))

        if not verified:
            flash("⚠️ Please verify your email before logging in.", "warning")
            return redirect(url_for("login"))

        session["user"] = email
        flash("✅ Logged in successfully!", "success")
        return redirect(url_for("index"))
    return render_template("login.html")

from flask import get_flashed_messages

@app.route("/logout")
def logout():
    session.pop("user", None)
    _ = get_flashed_messages()
    flash("Logged out successfully.", "info")
    return redirect(url_for("login"))
@app.route("/reset_password", methods=["GET", "POST"])
def reset_password():
    if request.method == "POST":
        email = request.form["email"]
        token = os.urandom(16).hex()
        expiry = (datetime.now() + timedelta(hours=1)).isoformat()

        try:
            with sqlite3.connect(USER_DB, timeout=10) as conn:
                cur = conn.cursor()
                cur.execute(
                    "UPDATE users SET reset_token=?, reset_expiry=? WHERE email=?",
                    (token, expiry, email),
                )
                if cur.rowcount == 0:
                    flash("❌ No account found with that email.", "error")
                    return redirect(url_for("reset_password"))
                conn.commit()
        except sqlite3.Error as e:
            print("DB error in reset_password:", e)
            flash("⚠ Database busy, please try again.", "danger")
            return redirect(url_for("reset_password"))

        link = url_for("do_reset_password", token=token, _external=True)
        msg = Message(
            "Reset your password",
            sender=app.config["MAIL_USERNAME"],
            recipients=[email],
        )
        msg.body = f"Click to reset password: {link}"
        mail.send(msg)

        flash("✅ Password reset link sent to email.", "success")
        return redirect(url_for("login"))

    return render_template("reset_password.html")


@app.route("/")
def land():
    return render_template("perfect.html")

@app.route("/api")
def api():
    return render_template("api.html")


@app.route("/do_reset/<token>", methods=["GET", "POST"])
def do_reset_password(token):
    if request.method == "POST":
        new_pw = request.form["password"]
        pw_hash = bcrypt.generate_password_hash(new_pw).decode("utf-8")

        with sqlite3.connect(USER_DB) as conn:
            cur = conn.cursor()
            cur.execute("SELECT id, reset_expiry FROM users WHERE reset_token=?", (token,))
            row = cur.fetchone()
            if not row:
                flash("❌ Invalid reset link", "error")
                return redirect(url_for("login"))

            uid, expiry = row
            if datetime.fromisoformat(expiry) < datetime.now():
                flash("❌ Reset link expired", "error")
                return redirect(url_for("login"))

            cur.execute(
                "UPDATE users SET password_hash=?, reset_token=NULL, reset_expiry=NULL WHERE id=?",
                (pw_hash, uid),
            )
            conn.commit()

        flash("✅ Password updated! Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("do_reset.html", token=token)

# -----------------------
# App pages
# -----------------------
@app.route("/dashboard")
def index():
    if "user" in session:
        return render_template("index.html", default_ticker="AAPL")
    else:
        return redirect(url_for('login'))

@app.route("/score")
def score_page():
    if "user" in session:
        ticker = request.args.get("ticker", "AAPL").upper().strip()
        df = fetch_yahoo_finance(ticker)
        df = clean_financial_data(df)

        if df is None or df.empty or not set(["Open","High","Low","Close","Volume"]).issubset(df.columns):
            return render_template("score.html", ticker=ticker,
                                error="No usable OHLCV data for this ticker right now. Try another symbol or check network."), 200
        df = feature_engineering(df)

        try:
            # ---- Base model ----
            model = CreditScoringModel()
            model.fit(df)
            model_score, prob, importances = model.score(df, return_details=True)

            # cap base model at 60
            base_score = max(0.0, min(60.0, float(model_score)))

            # ---- Market volatility penalty ----
            returns = df["Close"].pct_change().dropna()
            vol30 = returns.rolling(30).std().iloc[-1] if not returns.empty else 0.0
            vol_penalty = min(10.0, vol30 * 500) if pd.notna(vol30) else 0.0

            # ---- Liquidity / volume penalty ----
            avg_vol_long = df["Volume"].tail(60).mean() if len(df) >= 60 else df["Volume"].mean()
            avg_vol_short = df["Volume"].tail(5).mean()
            if avg_vol_long > 0:
                liquidity_ratio = avg_vol_short / avg_vol_long
                if liquidity_ratio < 0.5:
                    liq_penalty = 5.0
                elif liquidity_ratio < 0.8:
                    liq_penalty = 2.0
                else:
                    liq_penalty = 0.0
            else:
                liquidity_ratio, liq_penalty = 1.0, 0.0

            # ---- Events / sentiment ----
            headlines = fetch_news_headlines(ticker)
            events = extract_events_with_sentiment(headlines) if headlines else []

            pos, neu, neg, avg_compound = 0, 0, 0, 0.0
            if events:
                pos = sum(1 for e in events if e.get("sentiment_label") == "pos")
                neu = sum(1 for e in events if e.get("sentiment_label") == "neu")
                neg = sum(1 for e in events if e.get("sentiment_label") == "neg")

                compounds = [e.get("sentiment_score", 0.0) for e in events if isinstance(e, dict)]
                avg_compound = sum(compounds) / len(compounds) if compounds else 0.0

                if pos == 0 and neu == 0 and neg == 0 and compounds:
                    for score in compounds:
                        if score >= 0.05:
                            pos += 1
                        elif score <= -0.05:
                            neg += 1
                        else:
                            neu += 1

            # harsher sentiment adjustment
            if avg_compound <= -0.3:
                news_adj = max(-20, avg_compound * 30)
            elif avg_compound >= 0.3:
                news_adj = min(10, avg_compound * 10)
            else:
                news_adj = avg_compound * 5

            # ---- Final score ----
            final_score = base_score - vol_penalty - liq_penalty + news_adj
            final_score = max(0.0, min(100.0, final_score))

            # Save score to score_history
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute("INSERT INTO score_history (ticker, time, score) VALUES (?, ?, ?)",
                        (ticker, now, float(final_score)))
            conn.commit()
            conn.close()

            # ALSO: save user search to user_history if logged in
            # ---- Explanations ----
            parts = [f"Base model score (capped at 60): {base_score:.1f}."]
            if vol_penalty:
                parts.append(f"Volatility penalty −{vol_penalty:.1f} (30-day vol={vol30:.2%}).")
            if liq_penalty:
                parts.append(f"Liquidity penalty −{liq_penalty:.1f} (volume ratio={liquidity_ratio:.2f}).")
            if news_adj:
                direction = "positive" if news_adj > 0 else "negative"
                parts.append(f"News sentiment ({avg_compound:.2f}) gave {direction} adjustment {news_adj:+.1f}.")
            if not parts:
                parts.append("No significant adjustments applied.")
            event_explain = " ".join(parts)

            from utils.plots import plot_feature_importances, plot_price_with_mas, plot_event_sentiment
            feat_plot = plot_feature_importances(ticker, importances)
            price_plot = plot_price_with_mas(ticker, df)
            sent_plot = plot_event_sentiment(ticker, pos, neu, neg)

            base_explanation = make_plain_language_explanation(importances, df)
            combined_explanation = f"{base_explanation} {event_explain}"

            # ---- Advanced Explainability ----
            advanced_explanation = {
                "feature_contributions": importances,
                "trend_indicators": {
                    "short_term": "Bullish" if df["Close"].tail(5).mean() > df["Close"].tail(20).mean() else "Bearish",
                    "long_term": "Bullish" if len(df) > 200 and df["Close"].tail(50).mean() > df["Close"].tail(200).mean() else "Bearish"
                },
                "structured_events": f"Volatility penalty {vol_penalty:.1f}, Liquidity penalty {liq_penalty:.1f}.",
                "unstructured_events": f"News sentiment adjustment {news_adj:+.1f} (avg sentiment {avg_compound:.2f}).",
                "plain_summary": combined_explanation
            }

            # ---- Forecasts (used on score page narrative) ----
            price_forecast = forecast_prices(df, steps=5)   # next ~5 market days
            score_forecast = forecast_scores(ticker, steps=5)

            forecast_adj = 0.0
            if price_forecast and len(price_forecast) > 1:
                trend = price_forecast[-1] - price_forecast[0]
                if trend > 0:
                    forecast_adj += min(5.0, trend / df["Close"].iloc[-1] * 100)
                elif trend < 0:
                    forecast_adj -= min(5.0, abs(trend) / df["Close"].iloc[-1] * 100)

            if score_forecast and len(score_forecast) > 1:
                score_trend = score_forecast[-1] - score_forecast[0]
                if score_trend > 0:
                    forecast_adj += min(5.0, score_trend / 20)
                elif score_trend < 0:
                    forecast_adj -= min(5.0, abs(score_trend) / 20)

            final_score = final_score + forecast_adj
            final_score = max(0.0, min(100.0, final_score))
            if forecast_adj:
                parts.append(f"Forecast adjustment {forecast_adj:+.1f} based on predicted trends.")

            # Friendly narrative (Markdown → HTML)
            def build_friendly_explanation(base_score, vol_penalty, liq_penalty, news_adj, forecast_adj, final_score):
                explanation = []
                explanation.append(f"The stock’s base model score started at **{base_score:.1f}/100**.")
                if vol_penalty > 0:
                    explanation.append(f"Because of recent market swings, volatility reduced the score by **{vol_penalty:.1f} points**.")
                else:
                    explanation.append("Market volatility is stable, so no penalty was applied here.")
                if liq_penalty > 0:
                    explanation.append(f"Trading activity (liquidity) was lower than usual, costing about **{liq_penalty:.1f} points**.")
                else:
                    explanation.append("Liquidity levels look healthy, no deduction needed.")
                if news_adj > 0:
                    explanation.append(f"Recent news coverage was **positive**, which improved the score by **{news_adj:.1f} points**.")
                elif news_adj < 0:
                    explanation.append(f"Recent news sentiment was **negative**, reducing the score by **{abs(news_adj):.1f} points**.")
                else:
                    explanation.append("News coverage was neutral, so no adjustment here.")
                if forecast_adj > 0:
                    explanation.append(f"Our forecast model predicts an **uptrend**, adding about **{forecast_adj:.1f} points**.")
                elif forecast_adj < 0:
                    explanation.append(f"Our forecast suggests a possible **downtrend**, deducting **{abs(forecast_adj):.1f} points**.")
                else:
                    explanation.append("Forecasts suggest a stable outlook, no adjustment applied.")
                explanation.append(f"📊 Pulling this together, the **final creditworthiness score is {final_score:.1f}/100**.")
                return " ".join(explanation)

            friendly_explanation = markdown.markdown(
                build_friendly_explanation(base_score, vol_penalty, liq_penalty, news_adj, forecast_adj, final_score)
            )
            return render_template(
                "score.html",
                ticker=ticker,
                score=round(final_score, 1),
                underlying_model_score=round(float(model_score),1),
                prob=round(prob,3),
                importances=importances,
                explanation=combined_explanation,
                advanced_explanation=advanced_explanation,
                events=events,
                latest_close=float(df["Close"].iloc[-1]) if "Close" in df.columns and len(df) else None,
                feat_plot=os.path.basename(feat_plot) if feat_plot else None,
                price_plot=os.path.basename(price_plot) if price_plot else None,
                sent_plot=os.path.basename(sent_plot) if sent_plot else None,
                price_forecast=price_forecast,
                score_forecast=score_forecast,
                friendly_explanation=friendly_explanation,
                forecast_adj=round(forecast_adj, 2)
            )

        except Exception as e:
            return render_template("score.html", ticker=ticker, error=f"Model error: {e}"), 200
    else:
        return redirect(url_for("login"))

@app.route("/events")
def events_page():
    if "user" in session:
        headlines = fetch_news_headlines()
        events = extract_events_with_sentiment(headlines) if headlines else []
        return render_template("events.html", events=events)
    else:
        return redirect(url_for("login"))

@app.route("/api/score/<ticker>")
def api_score(ticker):
    ticker = ticker.upper().strip()
    df = fetch_yahoo_finance(ticker)
    df = clean_financial_data(df)
    if df is None or df.empty or not set(["Open","High","Low","Close","Volume"]).issubset(df.columns):
        return jsonify({"error": "No usable OHLCV data"}), 400
    df = feature_engineering(df)
    model = CreditScoringModel()
    model.fit(df)
    score, prob, importances = model.score(df, return_details=True)
    explanation = make_plain_language_explanation(importances, df)
    return jsonify({
        "ticker": ticker,
        "score_0_to_100": round(score,1),
        "p_good_last_step": round(prob,3),
        "feature_importances": importances,
        "explanation": explanation
    })

@app.route("/api/events")
def api_events():
    headlines = fetch_news_headlines()
    events = extract_events_with_sentiment(headlines) if headlines else []
    return jsonify({"events": events})

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

# -----------------------
# Peer comparison
# -----------------------

@app.route("/peer")
def peer_page():
    if "user" in session:
        symbol = request.args.get("symbol", "AAPL").upper()
        peers, influence = get_peers(symbol)

        # remove self from peers
        peers = [p for p in peers if p != symbol]

        results = {}
        for peer in peers:
            try:
                df = fetch_yahoo_finance(peer)
                if df is None or df.empty:
                    results[peer] = "No data available"
                    continue

                df["Return"] = df["Close"].pct_change()
                df = df.dropna()

                model = CreditScoringModel()
                model.fit(df, symbol=peer)
                score, p, _ = model.score(df, return_details=True)
                results[peer] = round(score, 1)

            except ValueError as e:
                msg = str(e)
                if "Only one class" in msg:
                    results[peer] = 50.0
                elif "Not enough data" in msg:
                    results[peer] = "Insufficient historical data to evaluate."
                else:
                    results[peer] = "Could not compute score."
            except Exception:
                results[peer] = "Unexpected error during evaluation."

        numeric_scores = {k: v for k, v in results.items() if isinstance(v, (int, float))}
        if numeric_scores:
            plt.figure(figsize=(8, 4))
            plt.bar(numeric_scores.keys(), numeric_scores.values())
            plt.title(f"Peer Comparison for {symbol}")
            plt.ylabel("Creditworthiness Score")
            os.makedirs("static/plots", exist_ok=True)
            plot_path = f"static/plots/{symbol}_peers.png"
            plt.savefig(plot_path)
            plt.close()
        else:
            plot_path = None

        return render_template(
            "peer.html",
            symbol=symbol,
            peers=peers,
            plot=os.path.basename(plot_path) if plot_path else None,
            scores=results,
            influence=influence
        )
    else:
        return redirect(url_for('login'))

# -----------------------
# Glossary
# -----------------------

@app.route("/glossary")
def glossary_page():
    if "user" in session:
        terms = {
            "Bullish": "A belief or trend indicating that prices will rise.",
            "Bearish": "A belief or trend indicating that prices will fall.",
            "Volatility": "How much and how quickly the price of an asset changes.",
            "Liquidity": "How easily an asset can be bought or sold without affecting its price.",
            "Creditworthiness": "An estimate of how reliable or stable a company/stock is financially.",
            "Sentiment": "The general feeling (positive, negative, neutral) from news or events.",
            "Momentum": "The speed of price changes over time, used to predict trends.",
            "Moving Average": "An average of prices over a set period (short-term or long-term).",
            "Penalty": "A reduction applied to the score due to risk factors (like volatility or low liquidity).",
            "Adjustment": "An increase or decrease in the score based on sentiment or events.",
            "Feature Importance": "How much each factor contributed to the model’s decision.",
            "Structured Data": "Well-organized data such as prices, volumes, and volatility metrics.",
            "Unstructured Data": "Textual or messy data like news articles, tweets, and press releases."
        }
        return render_template("glossary.html", terms=terms)
    else:
        return redirect(url_for("login"))

# -----------------------
# APIs
# -----------------------

@app.route("/api/score_history/<ticker>")
def api_score_history(ticker):
    ticker = ticker.upper().strip()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT time, score FROM score_history WHERE ticker = ? ORDER BY time ASC", (ticker,))
    rows = cur.fetchall()
    conn.close()
    history = [{"time": t, "score": s} for (t, s) in rows]
    return jsonify(history)

@app.route("/api/price_forecast")
def api_price_forecast():
    """
    JSON for Chart.js:
    params: ticker, window (default 30 actual points), days (5|15|30)
    """
    ticker = request.args.get("ticker", "AAPL").upper().strip()
    window = int(request.args.get("window", 30))
    days = int(request.args.get("days", 5))
    if days not in (5, 15, 30):
        days = 5

    df = fetch_yahoo_finance(ticker)
    df = clean_financial_data(df)
    if df is None or df.empty or "Close" not in df.columns:
        return jsonify({"error": "no_price_data"}), 400

    # Tail (for display), but fit SARIMAX on full df
    tail = df.tail(window).copy()
    actual = tail["Close"].astype(float).tolist()
    labels = list(range(1, len(actual) + 1))

    news_fore = forecast_prices_news_aware(ticker, df, steps=days)
    fcast = news_fore["forecast"]
    lower = news_fore["lower"]
    upper = news_fore["upper"]

    future_labels = list(range(len(labels) + 1, len(labels) + days + 1))

    return jsonify({
        "ticker": ticker,
        "labels": labels,
        "actual": actual,
        "future_labels": future_labels,
        "forecast": fcast,
        "lower": lower,
        "upper": upper
    })

# -----------------------
# Score Trend page (interactive chart)
# -----------------------

@app.route("/score_trend/<ticker>")
def score_trend_page(ticker):
    """
    Interactive price trend & forecast.
    - Shows last 30 actual closes (solid) and next N-day forecast (dotted).
    - X-axis: sequential days (no timestamps shown).
    - Controls for 5/15/30 days and ticker search.
    """
    if "user" in session:
        ticker = ticker.upper().strip()
        return render_template("score_trend.html", ticker=ticker)
    else:
        return redirect(url_for("login"))

'''@app.route("/ultra_notice", methods=["GET", "POST"])
def ultra_notice_page():
    if "user" not in session:
        flash("Please log in first.", "warning")
        return redirect(url_for("login"))

    email = session["user"]

    if request.method == "POST":
        ticker1 = request.form.get("ticker1", "").upper()
        ticker2 = request.form.get("ticker2", "").upper()
        ticker3 = request.form.get("ticker3", "").upper()
        threshold = float(request.form.get("threshold", 5))
        direction = request.form.get("direction", "up")
        active = 1 if request.form.get("active") == "on" else 0

        conn = sqlite3.connect(USER_DB)
        cur = conn.cursor()
        cur.execute("DELETE FROM ultra_notice WHERE email=?", (email,))
        cur.execute("""
            INSERT INTO ultra_notice (email, ticker1, ticker2, ticker3, threshold, direction, active)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (email, ticker1, ticker2, ticker3, threshold, direction, active))
        conn.commit()
        conn.close()

        flash("✅ Ultra Notice preferences saved!", "success")
        return redirect(url_for("ultra_notice_page"))

    # fetch saved prefs
    conn = sqlite3.connect(USER_DB)
    cur = conn.cursor()
    cur.execute("SELECT ticker1,ticker2,ticker3,threshold,direction,active FROM ultra_notice WHERE email=?", (email,))
    row = cur.fetchone()
    conn.close()

    prefs = None
    if row:
        prefs = {
            "ticker1": row[0] or "",
            "ticker2": row[1] or "",
            "ticker3": row[2] or "",
            "threshold": row[3],
            "direction": row[4],
            "active": bool(row[5])
        }

    return render_template("ultra_notice.html", prefs=prefs)

def monitor_ultra_notice():
    """Background job: monitors tickers & sends alerts if threshold is exceeded."""
    while True:
        try:
            conn = sqlite3.connect(USER_DB)
            cur = conn.cursor()
            cur.execute("SELECT email,ticker1,ticker2,ticker3,threshold,direction,active FROM ultra_notice WHERE active=1")
            rows = cur.fetchall()
            conn.close()

            for row in rows:
                email, t1, t2, t3, threshold, direction, active = row
                tickers = [t for t in [t1,t2,t3] if t]

                for ticker in tickers:
                    conn = sqlite3.connect(DB_PATH)
                    cur = conn.cursor()
                    cur.execute("SELECT time,score FROM score_history WHERE ticker=? ORDER BY time DESC LIMIT 2", (ticker,))
                    scores = cur.fetchall()
                    conn.close()

                    if len(scores) == 2:
                        latest, prev = scores[0][1], scores[1][1]
                        change = latest - prev

                        if (direction=="up" and change >= threshold) or (direction=="down" and change <= -threshold):
                            # send email
                            msg = Message(f"⚡ Ultra Notice Alert for {ticker}",
                                          sender=app.config["MAIL_USERNAME"],
                                          recipients=[email])
                            msg.body = f"The creditworthiness score of {ticker} changed from {prev:.1f} to {latest:.1f} ({change:+.1f})."
                            try:
                                mail.send(msg)
                                print(f"Alert sent to {email} for {ticker}")
                            except Exception as e:
                                print("Mail error:", e)

            time.sleep(60)  # check every 1 min
        except Exception as e:
            print("Monitor loop error:", e)
            time.sleep(60)
        threading.Thread(target=monitor_ultra_notice, daemon=True).start()
'''

def is_admin(email):
    conn = sqlite3.connect(USER_DB)
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM admins WHERE email=?", (email,))
    result = cur.fetchone()
    conn.close()
    return result is not None

def require_admin():
    if "user" not in session or not is_admin(session["user"]):
        return render_template("403.html"), 403
    return None


@app.route("/admin-panel", methods=["GET", "POST"])
def admin_panel():
    denied = require_admin()
    if denied: return denied
    conn = sqlite3.connect(USER_DB)
    cur = conn.cursor()
    cur.execute("SELECT id, email FROM users")
    users = cur.fetchall()
    cur.execute("SELECT id, email, designation FROM admins")
    admins = cur.fetchall()
    conn.close()
    return render_template("admin_panel.html", users=users, admins=admins)

@app.route("/add_admin", methods=["GET", "POST"])
def add_admin():
    denied = require_admin()
    if denied: return denied
    if request.method == "POST":
        new_email = request.form["email"]
        designation = request.form.get("designation", "Admin")

        conn = sqlite3.connect(USER_DB)
        cur = conn.cursor()
        cur.execute("INSERT OR IGNORE INTO admins (email, designation) VALUES (?, ?)", (new_email, designation))
        conn.commit()
        conn.close()

        flash(f"{new_email} added as {designation}", "success")
        return redirect(url_for("admin_panel"))

    return render_template("add_admin.html")

@app.route("/remove_user", methods=["POST"])
def remove_user():
    denied = require_admin()
    if denied: return denied  

    user_id = request.form["user_id"]

    conn = sqlite3.connect(USER_DB)
    cur = conn.cursor()
    cur.execute("DELETE FROM users WHERE id=?", (user_id,))
    conn.commit()
    conn.close()

    flash("User removed successfully.", "success")
    return redirect(url_for("admin_panel"))


@app.route("/send_email_to_all", methods=["POST"])
def send_email_to_all():
    denied = require_admin()
    if denied: 
        return denied  

    subject = request.form["subject"]
    body = request.form["body"]

    conn = sqlite3.connect(USER_DB)
    cur = conn.cursor()
    cur.execute("SELECT email FROM users")
    emails = [row[0] for row in cur.fetchall()]
    conn.close()

    # Use your existing email function
    for email in emails:
            msg = Message("Verify your Quorel account", sender=app.config["MAIL_USERNAME"], recipients=[email])
            msg.subject = subject
            msg.body = body
            mail.send(msg)

    flash("✅ Email sent to all users!", "success")
    return redirect(url_for("admin_panel"))

TICKERS_TO_MONITOR = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSM", "UNH", "V", "WMT",
    "JNJ", "JPM", "XOM", "MA", "NVO", "PG", "AVGO", "ORCL", "HD", "CVX",
]


@app.route("/api/top_stocks")
def api_top_stocks():
    data = []
    for ticker in TICKERS_TO_MONITOR:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="5d", interval="1d")  # last 5 daily candles
            if hist.empty:
                continue

            latest = hist.iloc[-1]
            prev_close = hist["Close"].iloc[-2] if len(hist) > 1 else latest["Close"]

            volume = int(latest["Volume"])
            price = round(latest["Close"], 2)
            pct_change = round(((latest["Close"] - prev_close) / prev_close) * 100, 2)

            data.append({
                "ticker": ticker,
                "price": price,
                "volume": volume,
                "pct_change": pct_change,
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        except Exception as e:
            print(f"❌ Error fetching {ticker}: {e}")

    # Sort by percentage gainers (top 10 movers)
    data = sorted(data, key=lambda x: abs(x["pct_change"]), reverse=True)[:10]

    return jsonify(data)

@app.route("/about")
def about():
    return render_template("ab.html")

from email.mime.text import MIMEText

@app.route("/contact", methods=["POST"])
def contact():
    try:
        name = request.form.get("name")
        email = request.form.get("email")
        topic = request.form.get("topic")
        message = request.form.get("message")

        # Format email body
        body = f"{message}\n\n---\nFrom: {name}\nEmail: {email}"

        msg = MIMEText(body)
        msg["Subject"] = topic
        msg["From"] = email
        msg["To"] = "quorel.connect@gmail.com"

        # Gmail SMTP (requires App Password if 2FA is enabled)
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login("quorel.connect@gmail.com", "yfvh jgef uzkk myai")  # ⚠️ Use app password, not normal password
            server.sendmail(email, "quorel.connect@gmail.com", msg.as_string())

        flash("✅ Your message has been sent!", "success")
    except Exception as e:
        print("❌ Error sending email:", e)
        flash("❌ Failed to send message. Try again later.", "danger")

    return redirect(url_for("about"))
if __name__ == "__main__":
    os.makedirs("static/plots", exist_ok=True)
    init_admin_db()
    app.run(host="0.0.0.0", port=5000)