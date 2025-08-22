def make_plain_language_explanation(importances: dict, df):
    if df is None or df.empty:
        return "Insufficient recent data to generate explanation."
    latest = df.iloc[-1].to_dict()
    top = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:3]
    if not top:
        return "Model has low discriminative importances; relying on base probability."
    parts = []
    for name, weight in top:
        val = latest.get(name, 0)
        try:
            parts.append(f"{name}~{float(val):.4f} (weight {weight:.2f})")
        except Exception:
            parts.append(f"{name} (weight {weight:.2f})")
    text = "Top drivers: " + "; ".join(parts) + ". "
    try:
        ret5 = float(df.get("Momentum5", [0])[-1])
        vol5 = float(df.get("Volatility5", [0])[-1])
        trend = "improving momentum" if ret5 > 0 else "softening momentum"
        risk = "stable volatility" if vol5 < 0.03 else "elevated short-term volatility"
        text += f"Recent trend shows {trend} with {risk}."
    except Exception:
        pass
    return text
