"""
Financial data layer.
Attempts real API calls (Alpha Vantage, Finnhub, NewsAPI) when keys are
configured; falls back to synthetic data transparently.
"""

from __future__ import annotations
import math
import requests
from datetime import datetime, timedelta
from typing import Any

from config import (
    ALPHA_VANTAGE_API_KEY,
    FINNHUB_API_KEY,
    NEWSAPI_KEY,
)
import synthetic_data as syn


# ── Helpers ───────────────────────────────────────────────────────────────────
def _has_key(key: str) -> bool:
    return bool(key and key.strip())


# ── Price history ─────────────────────────────────────────────────────────────
def get_price_history(ticker: str, days: int = 365) -> list[dict[str, Any]]:
    """Return OHLCV list. Uses Alpha Vantage if key present, else synthetic."""
    if _has_key(ALPHA_VANTAGE_API_KEY):
        try:
            url = (
                "https://www.alphavantage.co/query"
                f"?function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker}"
                f"&outputsize=full&apikey={ALPHA_VANTAGE_API_KEY}"
            )
            resp = requests.get(url, timeout=12).json()
            ts   = resp.get("Time Series (Daily Adjusted)", {})
            records = []
            cutoff  = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            for date_str in sorted(ts.keys(), reverse=True):
                if date_str < cutoff:
                    break
                row = ts[date_str]
                records.append({
                    "date":   date_str,
                    "open":   float(row["1. open"]),
                    "high":   float(row["2. high"]),
                    "low":    float(row["3. low"]),
                    "close":  float(row["5. adjusted close"]),
                    "volume": int(row["6. volume"]),
                    "data_source": "alpha_vantage",
                })
            if records:
                return list(reversed(records))
        except Exception:
            pass
    return syn.generate_price_history(ticker, days=days)


def get_fundamentals(ticker: str) -> dict[str, Any]:
    """Return company fundamentals. Uses Finnhub if key present."""
    if _has_key(FINNHUB_API_KEY):
        try:
            headers = {"X-Finnhub-Token": FINNHUB_API_KEY}
            q_data = requests.get(
                f"https://finnhub.io/api/v1/quote?symbol={ticker}",
                headers=headers, timeout=8
            ).json()
            m_data = requests.get(
                f"https://finnhub.io/api/v1/stock/metric?symbol={ticker}&metric=all",
                headers=headers, timeout=8
            ).json().get("metric", {})
            p_data = requests.get(
                f"https://finnhub.io/api/v1/stock/profile2?symbol={ticker}",
                headers=headers, timeout=8
            ).json()
            return {
                "ticker":         ticker,
                "name":           p_data.get("name", ticker),
                "sector":         p_data.get("finnhubIndustry", "Unknown"),
                "price":          q_data.get("c", 0),
                "market_cap_b":   round((p_data.get("marketCapitalization", 0) or 0) / 1000, 1),
                "pe_ratio":       m_data.get("peBasicExclExtraTTM") or 0,
                "pb_ratio":       m_data.get("pbAnnual") or 0,
                "eps":            m_data.get("epsBasicExclExtraItemsTTM") or 0,
                "revenue_b":      round((m_data.get("revenuePerShareTTM") or 0) * 1e6 / 1e9, 1),
                "profit_margin":  round((m_data.get("netProfitMarginTTM") or 0), 1),
                "debt_equity":    m_data.get("totalDebt/totalEquityAnnual") or 0,
                "roe":            round((m_data.get("roeTTM") or 0) * 100, 1),
                "dividend_yield": m_data.get("dividendYieldIndicatedAnnual") or 0,
                "beta":           m_data.get("beta") or 1.0,
                "52w_high":       m_data.get("52WeekHigh") or 0,
                "52w_low":        m_data.get("52WeekLow") or 0,
                "analyst_rating": "N/A",
                "price_target":   0,
                "data_source":    "finnhub",
            }
        except Exception:
            pass
    return syn.generate_fundamentals(ticker)


def get_news(ticker: str, n: int = 8) -> list[dict[str, Any]]:
    """Return news articles. Uses NewsAPI if key present."""
    if _has_key(NEWSAPI_KEY):
        try:
            from_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            url = (
                "https://newsapi.org/v2/everything"
                f"?q={ticker}&from={from_date}&sortBy=publishedAt"
                f"&language=en&pageSize={n}&apiKey={NEWSAPI_KEY}"
            )
            resp     = requests.get(url, timeout=10).json()
            articles = resp.get("articles", [])
            results  = []
            for art in articles[:n]:
                title   = (art.get("title") or "").lower()
                score   = _simple_sentiment(title)
                results.append({
                    "headline":        art.get("title", ""),
                    "source":          art.get("source", {}).get("name", "Unknown"),
                    "published_date":  (art.get("publishedAt") or "")[:10],
                    "sentiment_score": score,
                    "sentiment_label": _score_to_label(score),
                    "url":             art.get("url", ""),
                    "data_source":     "newsapi",
                })
            if results:
                return results
        except Exception:
            pass
    return syn.generate_news(ticker, n=n)


def get_market_overview() -> dict[str, Any]:
    return syn.generate_market_overview()


# ── Financial ratio calculations ──────────────────────────────────────────────
def calculate_financial_ratios(fundamentals: dict[str, Any]) -> dict[str, Any]:
    """Derive additional computed ratios from a fundamentals dict."""
    pe     = fundamentals.get("pe_ratio",       0) or 0
    pb     = fundamentals.get("pb_ratio",       0) or 0
    de     = fundamentals.get("debt_equity",    0) or 0
    roe    = fundamentals.get("roe",            0) or 0
    beta   = fundamentals.get("beta",           1) or 1
    margin = fundamentals.get("profit_margin",  0) or 0
    div    = fundamentals.get("dividend_yield", 0) or 0
    eps    = fundamentals.get("eps",            0) or 0
    rev_b  = fundamentals.get("revenue_b",      0) or 0
    mcap_b = fundamentals.get("market_cap_b",   0) or 0

    # Graham number proxy: sqrt(22.5 × EPS × BV) — simplified
    graham = round(math.sqrt(max(22.5 * eps * max(pb, 0.1), 0)), 2) if eps > 0 else 0

    # Price-to-Sales proxy
    ps_ratio = round(mcap_b / rev_b, 2) if rev_b > 0 else 0

    # Quality score (0-100): rewards high ROE, high margin, low D/E, positive EPS
    # Each component capped at 25 points
    roe_capped    = min(roe, 60)    # cap to avoid outliers like HD
    margin_capped = min(margin, 40)
    q_score = min(100, max(0,
        (roe_capped   / 60) * 25 +
        (margin_capped / 40) * 25 +
        max(0, (3.0 - de) / 3.0) * 25 +
        (min(abs(eps), 20) / 20) * 25
    ))

    # Value score: rewards low P/E, low P/B relative to benchmarks
    v_score = min(100, max(0,
        max(0, (45 - pe) / 45) * 50 +
        max(0, (12 - pb) / 12) * 50
    ))

    # Momentum score: derived from dividend yield bonus + earnings power
    momentum = min(100, max(0,
        min(div * 15, 30) +          # up to 30 pts for dividend yield
        (min(margin, 35) / 35) * 40 +  # up to 40 pts for profitability
        max(0, (2 - de) / 2) * 30     # up to 30 pts for low leverage
    ))

    # Risk rating 1–5
    risk = _compute_risk_rating(beta, de, margin)

    # Upside to price target
    price  = fundamentals.get("price", 0)
    target = fundamentals.get("price_target", 0)
    upside_pct = round((target - price) / price * 100, 1) if price > 0 and target > 0 else 0

    return {
        "pe_ratio":       round(pe, 1),
        "pb_ratio":       round(pb, 1),
        "ps_ratio":       ps_ratio,
        "debt_equity":    round(de, 2),
        "roe":            round(roe, 1),
        "profit_margin":  round(margin, 1),
        "dividend_yield": round(div, 2),
        "beta":           round(beta, 2),
        "graham_number":  graham,
        "quality_score":  round(q_score),
        "value_score":    round(v_score),
        "momentum_score": round(momentum),
        "risk_rating":    risk,
        "upside_pct":     upside_pct,
    }


def _compute_risk_rating(beta: float, de: float, margin: float) -> int:
    score = 0
    if beta > 1.6:   score += 2
    elif beta > 1.1: score += 1
    if de > 2.5:     score += 2
    elif de > 1.2:   score += 1
    if margin < 5:   score += 1
    return min(5, max(1, score + 1))


def _simple_sentiment(text: str) -> float:
    positives = ["beat", "surge", "record", "growth", "upgrade", "profit",
                 "rise", "gain", "bull", "strong", "expand", "outperform",
                 "exceed", "launch", "partnership", "innovation"]
    negatives = ["miss", "fall", "drop", "loss", "downgrade", "concern",
                 "weak", "decline", "bear", "scrutiny", "warning", "cut",
                 "lawsuit", "recall", "delay", "investigation"]
    pos = sum(w in text for w in positives)
    neg = sum(w in text for w in negatives)
    total = pos + neg
    return round((pos - neg) / total, 3) if total else 0.0


def _score_to_label(score: float) -> str:
    if score > 0.2:  return "Positive"
    if score < -0.2: return "Negative"
    return "Neutral"


# ── Comparative analysis ──────────────────────────────────────────────────────
def compare_stocks(tickers: list[str]) -> list[dict[str, Any]]:
    """Return enriched fundamentals + ratios for a list of tickers."""
    results = []
    for t in tickers:
        f = get_fundamentals(t)
        r = calculate_financial_ratios(f)
        results.append({"ticker": t, **f, **r})
    return results


# ── Portfolio analytics ───────────────────────────────────────────────────────
def calculate_portfolio_metrics(
    tickers: list[str],
    weights: list[float],
    ratios: dict[str, dict],
) -> dict[str, Any]:
    """
    Compute aggregate portfolio-level metrics from individual stock metrics.
    *weights* must sum to 1.0.
    """
    if not tickers or not weights:
        return {}

    w_sum = sum(weights)
    norm  = [w / w_sum for w in weights]  # normalise just in case

    agg: dict[str, float] = {
        "weighted_pe":       0.0,
        "weighted_beta":     0.0,
        "weighted_div":      0.0,
        "weighted_quality":  0.0,
        "weighted_value":    0.0,
        "weighted_risk":     0.0,
    }
    for t, w in zip(tickers, norm):
        r = ratios.get(t, {})
        agg["weighted_pe"]      += w * (r.get("pe_ratio", 0) or 0)
        agg["weighted_beta"]    += w * (r.get("beta", 1) or 1)
        agg["weighted_div"]     += w * (r.get("dividend_yield", 0) or 0)
        agg["weighted_quality"] += w * (r.get("quality_score", 50) or 50)
        agg["weighted_value"]   += w * (r.get("value_score", 50) or 50)
        agg["weighted_risk"]    += w * (r.get("risk_rating", 3) or 3)

    return {
        "portfolio_pe":       round(agg["weighted_pe"], 1),
        "portfolio_beta":     round(agg["weighted_beta"], 2),
        "portfolio_div_yield": round(agg["weighted_div"], 2),
        "portfolio_quality":  round(agg["weighted_quality"]),
        "portfolio_value":    round(agg["weighted_value"]),
        "portfolio_risk":     round(agg["weighted_risk"], 1),
    }
