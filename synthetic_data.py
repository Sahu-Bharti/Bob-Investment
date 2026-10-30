"""
Synthetic financial data generator.
Produces realistic OHLCV prices, fundamentals, news, and market data
so the app works fully offline before real API keys are added.
"""

from __future__ import annotations
import random
import math
from datetime import datetime, timedelta
from typing import Any
from config import DEMO_STOCKS


# ── Reproducible RNG per ticker ───────────────────────────────────────────────
def _rng(ticker: str, seed_offset: int = 0) -> random.Random:
    seed = sum(ord(c) for c in ticker) * 31 + seed_offset
    return random.Random(seed)


# ── Realistic base data per ticker ────────────────────────────────────────────
STOCK_PROFILES: dict[str, dict] = {
    "AAPL": {"base": 185.0, "sector": "Technology",              "vol": 0.22, "drift": 0.12, "beta": 1.20},
    "MSFT": {"base": 415.0, "sector": "Technology",              "vol": 0.21, "drift": 0.15, "beta": 1.10},
    "GOOGL":{"base": 175.0, "sector": "Technology",              "vol": 0.25, "drift": 0.10, "beta": 1.15},
    "AMZN": {"base": 195.0, "sector": "Consumer Discretionary",  "vol": 0.28, "drift": 0.14, "beta": 1.25},
    "NVDA": {"base": 880.0, "sector": "Technology",              "vol": 0.45, "drift": 0.35, "beta": 1.75},
    "META": {"base": 520.0, "sector": "Technology",              "vol": 0.30, "drift": 0.20, "beta": 1.30},
    "TSLA": {"base": 175.0, "sector": "Consumer Discretionary",  "vol": 0.50, "drift": 0.08, "beta": 1.90},
    "JPM":  {"base": 210.0, "sector": "Financials",              "vol": 0.20, "drift": 0.09, "beta": 1.05},
    "V":    {"base": 275.0, "sector": "Financials",              "vol": 0.16, "drift": 0.10, "beta": 0.95},
    "JNJ":  {"base": 155.0, "sector": "Healthcare",              "vol": 0.15, "drift": 0.05, "beta": 0.60},
    "WMT":  {"base": 175.0, "sector": "Consumer Staples",        "vol": 0.16, "drift": 0.07, "beta": 0.55},
    "PG":   {"base": 165.0, "sector": "Consumer Staples",        "vol": 0.14, "drift": 0.06, "beta": 0.50},
    "UNH":  {"base": 520.0, "sector": "Healthcare",              "vol": 0.19, "drift": 0.11, "beta": 0.80},
    "HD":   {"base": 380.0, "sector": "Consumer Discretionary",  "vol": 0.22, "drift": 0.09, "beta": 1.00},
    "BAC":  {"base":  40.0, "sector": "Financials",              "vol": 0.28, "drift": 0.08, "beta": 1.35},
}
DEFAULT_PROFILE = {"base": 100.0, "sector": "Technology", "vol": 0.25, "drift": 0.10, "beta": 1.0}


def _profile(ticker: str) -> dict:
    return STOCK_PROFILES.get(ticker, DEFAULT_PROFILE)


# ── Price history ─────────────────────────────────────────────────────────────
def generate_price_history(
    ticker: str,
    days: int = 365,
    end_date: datetime | None = None,
) -> list[dict[str, Any]]:
    """Return a list of OHLCV dicts for *days* trading days ending at *end_date*."""
    rng  = _rng(ticker)
    prof = _profile(ticker)
    base = prof["base"]
    end  = end_date or datetime.now()

    # Build trading-day calendar
    calendar: list[datetime] = []
    dt = end
    while len(calendar) < days:
        if dt.weekday() < 5:
            calendar.append(dt)
        dt -= timedelta(days=1)
    calendar.reverse()

    price        = base * rng.uniform(0.75, 1.25)
    daily_vol    = prof["vol"] / math.sqrt(252)
    daily_drift  = prof["drift"] / 252

    records: list[dict[str, Any]] = []
    for day in calendar:
        ret   = rng.gauss(daily_drift, daily_vol)
        close = max(price * (1 + ret), 0.01)
        intra = abs(rng.gauss(0, daily_vol * 0.6))
        high  = close * (1 + intra)
        low   = close * (1 - intra * rng.uniform(0.3, 1.0))
        open_ = price * (1 + rng.gauss(0, daily_vol * 0.3))
        open_ = max(min(open_, high), low)
        volume = int(rng.uniform(4_000_000, 90_000_000))
        records.append({
            "date":   day.strftime("%Y-%m-%d"),
            "open":   round(open_, 2),
            "high":   round(max(high, open_, close), 2),
            "low":    round(min(low, open_, close), 2),
            "close":  round(close, 2),
            "volume": volume,
        })
        price = close
    return records


# ── Fundamentals ──────────────────────────────────────────────────────────────
# Per-ticker realistic ranges for key ratios
_FUNDAMENTALS_PARAMS: dict[str, dict] = {
    "AAPL": {"pe": (28, 34), "pb": (45, 55), "eps": (6.0, 7.0),  "roe": (145, 160), "de": (1.5, 2.0),  "margin": (24, 27), "div": (0.5, 0.6),  "rev_b": (380, 400)},
    "MSFT": {"pe": (34, 40), "pb": (12, 15), "eps": (11.0, 13.0),"roe": (35, 45),  "de": (0.3, 0.5),  "margin": (34, 38), "div": (0.7, 0.9),  "rev_b": (210, 230)},
    "GOOGL":{"pe": (24, 30), "pb": (6,  8),  "eps": (7.0, 9.0),  "roe": (22, 28),  "de": (0.05, 0.15),"margin": (22, 26), "div": (0.0, 0.0),  "rev_b": (300, 320)},
    "AMZN": {"pe": (40, 55), "pb": (8, 12),  "eps": (3.0, 5.0),  "roe": (15, 22),  "de": (0.5, 0.9),  "margin": (5, 10),  "div": (0.0, 0.0),  "rev_b": (550, 590)},
    "NVDA": {"pe": (55, 75), "pb": (20, 30), "eps": (16.0, 22.0),"roe": (60, 80),  "de": (0.3, 0.6),  "margin": (50, 57), "div": (0.03, 0.05),"rev_b": (90, 120)},
    "META": {"pe": (24, 30), "pb": (8, 11),  "eps": (16.0, 19.0),"roe": (32, 40),  "de": (0.1, 0.2),  "margin": (28, 33), "div": (0.4, 0.5),  "rev_b": (130, 145)},
    "TSLA": {"pe": (50, 80), "pb": (10, 15), "eps": (2.5, 4.0),  "roe": (10, 18),  "de": (0.1, 0.3),  "margin": (8, 14),  "div": (0.0, 0.0),  "rev_b": (96, 100)},
    "JPM":  {"pe": (11, 14), "pb": (1.8, 2.3),"eps": (16.0, 20.0),"roe": (14, 18), "de": (1.2, 1.8),  "margin": (30, 36), "div": (2.2, 2.8),  "rev_b": (160, 180)},
    "V":    {"pe": (26, 32), "pb": (13, 17), "eps": (9.0, 11.0), "roe": (42, 50),  "de": (0.5, 0.8),  "margin": (50, 56), "div": (0.7, 0.9),  "rev_b": (33, 36)},
    "JNJ":  {"pe": (15, 20), "pb": (5, 8),   "eps": (9.0, 11.0), "roe": (22, 28),  "de": (0.4, 0.7),  "margin": (20, 25), "div": (2.8, 3.1),  "rev_b": (85, 90)},
    "WMT":  {"pe": (26, 32), "pb": (7, 9),   "eps": (6.0, 7.5),  "roe": (18, 24),  "de": (0.5, 0.9),  "margin": (2.5, 4.0),"div": (1.2, 1.5), "rev_b": (640, 660)},
    "PG":   {"pe": (24, 28), "pb": (8, 10),  "eps": (5.5, 6.5),  "roe": (28, 34),  "de": (0.6, 1.0),  "margin": (18, 22), "div": (2.3, 2.6),  "rev_b": (82, 85)},
    "UNH":  {"pe": (18, 24), "pb": (5, 7),   "eps": (22.0, 27.0),"roe": (24, 30),  "de": (0.5, 0.8),  "margin": (5, 8),   "div": (1.3, 1.6),  "rev_b": (360, 380)},
    "HD":   {"pe": (22, 27), "pb": (60, 80), "eps": (14.0, 17.0),"roe": (1000,1200),"de":(5.0, 7.0),  "margin": (10, 14), "div": (2.3, 2.7),  "rev_b": (153, 158)},
    "BAC":  {"pe": (11, 14), "pb": (1.0, 1.5),"eps": (3.0, 4.0), "roe": (10, 14),  "de": (1.0, 1.5),  "margin": (25, 32), "div": (2.0, 2.5),  "rev_b": (93, 100)},
}
_DEFAULT_PARAMS = {"pe": (18, 30), "pb": (2, 8), "eps": (2, 12), "roe": (10, 25), "de": (0.2, 1.5), "margin": (8, 25), "div": (0, 2.5), "rev_b": (10, 200)}


def generate_fundamentals(ticker: str) -> dict[str, Any]:
    """Return a realistic fundamental snapshot for *ticker*."""
    rng  = _rng(ticker, seed_offset=42)
    prof = _profile(ticker)
    prm  = _FUNDAMENTALS_PARAMS.get(ticker, _DEFAULT_PARAMS)

    def _r(lo_hi):
        return round(rng.uniform(*lo_hi), 2)

    price  = prof["base"] * rng.uniform(0.88, 1.12)
    eps    = _r(prm["eps"])
    pe     = _r(prm["pe"])
    pb     = _r(prm["pb"])
    de     = _r(prm["de"])
    roe    = _r(prm["roe"])
    margin = _r(prm["margin"])
    div    = _r(prm["div"])
    rev_b  = _r(prm["rev_b"])
    beta   = round(prof["beta"] * rng.uniform(0.90, 1.10), 2)
    mcap_b = round(price * rng.uniform(0.8e10, 2.2e10) / 1e9, 1)

    return {
        "ticker":          ticker,
        "name":            DEMO_STOCKS.get(ticker, ticker),
        "sector":          prof["sector"],
        "price":           round(price, 2),
        "market_cap_b":    mcap_b,
        "pe_ratio":        pe,
        "pb_ratio":        pb,
        "eps":             eps,
        "revenue_b":       rev_b,
        "profit_margin":   margin,
        "debt_equity":     de,
        "roe":             roe,
        "dividend_yield":  div,
        "beta":            beta,
        "52w_high":        round(price * rng.uniform(1.06, 1.42), 2),
        "52w_low":         round(price * rng.uniform(0.58, 0.92), 2),
        "analyst_rating":  rng.choice(["Strong Buy", "Buy", "Buy", "Hold", "Hold", "Sell"]),
        "price_target":    round(price * rng.uniform(0.92, 1.48), 2),
        "data_source":     "synthetic",
    }


# ── News sentiment ────────────────────────────────────────────────────────────
_POSITIVE_TEMPLATES = [
    "{name} beats Q{q} earnings expectations with {pct}% revenue growth",
    "{name} announces ${val}B share buyback programme",
    "{name} expands into {region} market amid strong demand",
    "Analysts upgrade {ticker} to Buy on AI-driven growth outlook",
    "{name} reports record-high operating margin in latest quarter",
    "Wall Street raises {ticker} price target by {pct}%",
    "{name} announces strategic partnership to accelerate {area} initiatives",
    "{ticker} soars after delivering stronger-than-expected free cash flow",
]

_NEGATIVE_TEMPLATES = [
    "{name} faces regulatory scrutiny over {area} practices",
    "{name} misses revenue estimates as {area} segment weakens",
    "{ticker} stock falls after disappointing {area} guidance",
    "{name} announces restructuring amid {area} headwinds",
    "Analysts cut {ticker} price target citing valuation concerns",
    "{name} warns of slowing growth in {region} market",
]

_NEUTRAL_TEMPLATES = [
    "{name} CEO outlines five-year growth strategy at investor day",
    "{name} to present at upcoming financial services conference",
    "{ticker} trading volume remains elevated amid market uncertainty",
    "{name} files annual report with consistent dividend outlook",
]

_REGIONS  = ["Asia-Pacific", "European", "Latin American", "Middle Eastern", "North American"]
_AREAS    = ["data privacy", "antitrust", "supply chain", "cloud", "AI", "retail", "payments", "cybersecurity"]
_SOURCES  = ["Reuters", "Bloomberg", "CNBC", "Financial Times", "The Wall Street Journal",
             "MarketWatch", "Seeking Alpha", "Barron's", "The Economist"]


def generate_news(ticker: str, n: int = 8) -> list[dict[str, Any]]:
    """Return *n* synthetic news headlines with sentiment scores."""
    rng  = _rng(ticker, seed_offset=99)
    name = DEMO_STOCKS.get(ticker, ticker)
    articles = []
    now = datetime.now()

    # Mix positive/negative/neutral headlines
    templates = (
        _POSITIVE_TEMPLATES * 3 +
        _NEGATIVE_TEMPLATES * 2 +
        _NEUTRAL_TEMPLATES   * 1
    )

    for i in range(n):
        tpl = rng.choice(templates)
        headline = tpl.format(
            ticker=ticker, name=name,
            q=rng.randint(1, 4),
            pct=rng.randint(5, 38),
            val=round(rng.uniform(1, 25), 1),
            region=rng.choice(_REGIONS),
            area=rng.choice(_AREAS),
        )

        # Sentiment aligned with template type
        if tpl in _POSITIVE_TEMPLATES:
            score = round(rng.uniform(0.3, 0.95), 3)
        elif tpl in _NEGATIVE_TEMPLATES:
            score = round(rng.uniform(-0.95, -0.25), 3)
        else:
            score = round(rng.uniform(-0.15, 0.15), 3)

        label = (
            "Positive" if score > 0.2 else
            "Negative" if score < -0.2 else
            "Neutral"
        )
        pub_date = (now - timedelta(days=rng.randint(0, 28))).strftime("%Y-%m-%d")
        articles.append({
            "headline":        headline,
            "source":          rng.choice(_SOURCES),
            "published_date":  pub_date,
            "sentiment_score": score,
            "sentiment_label": label,
            "url":             f"https://example.com/news/{ticker.lower()}-{i+1}",
        })
    return articles


# ── Market overview ───────────────────────────────────────────────────────────
def generate_market_overview() -> dict[str, Any]:
    """Return synthetic index levels and macro indicators."""
    rng = random.Random(20241029)
    return {
        "indices": {
            "S&P 500":   {"level": round(rng.uniform(4900, 5500), 0), "change_pct": round(rng.uniform(-1.5,  1.8), 2)},
            "NASDAQ":    {"level": round(rng.uniform(15500, 18000), 0),"change_pct": round(rng.uniform(-2.0,  2.5), 2)},
            "Dow Jones": {"level": round(rng.uniform(38500, 43000), 0),"change_pct": round(rng.uniform(-1.2,  1.5), 2)},
            "VIX":       {"level": round(rng.uniform(13, 26), 1),      "change_pct": round(rng.uniform(-6.0,  6.0), 2)},
        },
        "macro": {
            "Fed Funds Rate":    round(rng.uniform(4.25, 5.50), 2),
            "10Y Treasury Yield": round(rng.uniform(3.8,  4.8),  2),
            "CPI YoY (%)":       round(rng.uniform(2.5,  4.0),   1),
            "GDP Growth (%)":    round(rng.uniform(1.5,  3.2),   1),
            "Unemployment (%)":  round(rng.uniform(3.5,  4.5),   1),
        },
    }


# ── Portfolio simulation ──────────────────────────────────────────────────────
def generate_portfolio_performance(
    tickers: list[str],
    weights: list[float],
    days: int = 365,
) -> list[dict[str, Any]]:
    """
    Return daily portfolio value timeseries.
    *weights* must sum to 1.0.
    """
    histories = {t: generate_price_history(t, days=days) for t in tickers}
    n_days = min(len(h) for h in histories.values())
    if n_days == 0:
        return []

    # Normalise each ticker to base 1.0 at start
    records = []
    for i in range(n_days):
        total = 0.0
        for t, w in zip(tickers, weights):
            h = histories[t]
            if i < len(h):
                total += w * (h[i]["close"] / h[0]["close"])
        records.append({
            "date":  histories[tickers[0]][i]["date"],
            "value": round(total * 10000, 2),  # assume $10k invested
        })
    return records
