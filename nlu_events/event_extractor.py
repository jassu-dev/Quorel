from nltk.sentiment.vader import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

def extract_events_with_sentiment(headlines):
    events = []
    for h in headlines:
        text = h["headline"] if isinstance(h, dict) and "headline" in h else str(h)
        scores = sia.polarity_scores(text)
        compound = scores["compound"]
        # Label sentiment
        if compound >= 0.05:
            label = "Positive"
        elif compound <= -0.05:
            label = "Negative"
        else:
            label = "Neutral"

        events.append({
            "date": h["date"] if isinstance(h, dict) and "date" in h else "",
            "headline": text,
            "sentiment": scores,          # full dict
            "sentiment_score": compound,  # single number
            "sentiment_label": label      # string
        })
    return events
