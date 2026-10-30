"""
Sequential Agent Workflow Orchestrator — v2.
Implements six specialised agents that run in sequence to produce
a comprehensive investment research report.

Agent Pipeline:
  1. DataAgent       — fetch price history, fundamentals, news
  2. RatioAgent      — compute financial ratios & quality/value/momentum scores
  3. SentimentAgent  — analyse news sentiment with IBM Granite
  4. RiskAgent       — assess multi-dimensional risk with IBM Granite
  5. ReportAgent     — synthesise full investment research report with IBM Granite
  6. PortfolioAgent  — generate portfolio-level summary and recommendations
"""

from __future__ import annotations
from typing import Any, Callable
from datetime import datetime
import math

from financial_data import (
    get_price_history,
    get_fundamentals,
    get_news,
    get_market_overview,
    calculate_financial_ratios,
    compare_stocks,
    calculate_portfolio_metrics,
)
from llm_client import generate, generate_safe
from guardrails import build_system_prompt, sanitize_output, build_citations


# ── Progress callback type ────────────────────────────────────────────────────
ProgressFn = Callable[[str, int], None]   # (message, pct_complete)


def _noop(msg: str, pct: int) -> None:
    pass


# ── Agent 1: Data Fetching ────────────────────────────────────────────────────
class DataAgent:
    """Collects all raw data needed for downstream agents."""

    def run(
        self,
        tickers: list[str],
        days: int = 365,
        progress: ProgressFn = _noop,
    ) -> dict[str, Any]:
        progress("📡 DataAgent: fetching market overview …", 5)
        market = get_market_overview()

        progress("📡 DataAgent: fetching price histories …", 15)
        price_histories: dict[str, list] = {}
        for t in tickers:
            price_histories[t] = get_price_history(t, days=days)

        progress("📡 DataAgent: fetching fundamentals …", 28)
        fundamentals: dict[str, dict] = {}
        for t in tickers:
            fundamentals[t] = get_fundamentals(t)

        progress("📡 DataAgent: fetching news …", 42)
        news: dict[str, list] = {}
        for t in tickers:
            news[t] = get_news(t, n=10)

        # Collect data source identifiers for citations
        sources: set[str] = set()
        for t in tickers:
            src = fundamentals[t].get("data_source", "synthetic")
            sources.add(src)

        return {
            "tickers":         tickers,
            "market":          market,
            "price_histories": price_histories,
            "fundamentals":    fundamentals,
            "news":            news,
            "data_sources":    list(sources),
            "fetched_at":      datetime.now().isoformat(),
        }


# ── Agent 2: Ratio Calculation ────────────────────────────────────────────────
class RatioAgent:
    """Derives financial ratios and composite scores from fundamentals."""

    def run(
        self,
        data: dict[str, Any],
        progress: ProgressFn = _noop,
    ) -> dict[str, Any]:
        progress("🔢 RatioAgent: computing financial ratios …", 52)
        ratios: dict[str, dict] = {}
        for t in data["tickers"]:
            f = data["fundamentals"][t]
            ratios[t] = calculate_financial_ratios(f)

        # Rank tickers by quality score within this peer group
        ranked = sorted(
            data["tickers"],
            key=lambda t: ratios[t].get("quality_score", 0),
            reverse=True,
        )

        data["ratios"]       = ratios
        data["quality_rank"] = ranked
        return data


# ── Agent 3: Sentiment Analysis ───────────────────────────────────────────────
class SentimentAgent:
    """Uses IBM Granite to analyse news sentiment and macro conditions."""

    def run(
        self,
        data: dict[str, Any],
        progress: ProgressFn = _noop,
    ) -> dict[str, Any]:
        progress("📰 SentimentAgent: analysing news sentiment with IBM Granite …", 62)
        sentiment_summaries: dict[str, str] = {}
        sentiment_scores:    dict[str, float] = {}

        system = build_system_prompt("news")

        for t in data["tickers"]:
            articles = data["news"][t]
            headlines = "\n".join(
                f"- [{a['sentiment_label']} {a['sentiment_score']:+.2f}] {a['headline']}"
                f" — {a.get('source', '')} ({a.get('published_date', '')})"
                for a in articles
            )
            avg_score = (
                sum(a["sentiment_score"] for a in articles) / len(articles)
                if articles else 0.0
            )
            sentiment_scores[t] = round(avg_score, 3)

            prompt = (
                f"{system}\n\n"
                f"Company: {data['fundamentals'][t].get('name', t)} ({t})\n"
                f"Sector: {data['fundamentals'][t].get('sector', 'N/A')}\n"
                f"Recent news headlines:\n{headlines}\n\n"
                "Provide a concise 3-paragraph sentiment summary:\n"
                "Paragraph 1: Overall market sentiment tone and score interpretation.\n"
                "Paragraph 2: Key positive catalysts and opportunities identified.\n"
                "Paragraph 3: Key negative risks and short-term sentiment outlook.\n"
                "Sentiment Summary:"
            )
            try:
                raw = generate(prompt)
                sentiment_summaries[t] = sanitize_output(raw)
            except Exception as e:
                sentiment_summaries[t] = (
                    f"Sentiment analysis temporarily unavailable. "
                    f"Average headline sentiment score: {avg_score:+.3f} "
                    f"({'Positive' if avg_score > 0.2 else 'Negative' if avg_score < -0.2 else 'Neutral'})."
                )

        data["sentiment_summaries"] = sentiment_summaries
        data["sentiment_scores"]    = sentiment_scores
        return data


# ── Agent 4: Risk Assessment ──────────────────────────────────────────────────
class RiskAgent:
    """Uses IBM Granite to produce a structured multi-dimensional risk report."""

    _RISK_CATEGORIES = [
        "Market Risk (beta, volatility, macro exposure, interest rate sensitivity)",
        "Financial Risk (leverage, liquidity, debt service coverage, working capital)",
        "Business Risk (competitive moat, pricing power, margin sustainability)",
        "Regulatory & ESG Risk (compliance, litigation, environmental/social factors)",
        "Valuation Risk (overvaluation, multiple compression, growth expectations)",
    ]

    def run(
        self,
        data: dict[str, Any],
        progress: ProgressFn = _noop,
    ) -> dict[str, Any]:
        progress("⚠️  RiskAgent: assessing investment risks with IBM Granite …", 74)
        risk_reports:  dict[str, str] = {}
        overall_risk:  dict[str, int] = {}
        risk_breakdown: dict[str, dict] = {}

        system = build_system_prompt("risk")
        cats   = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(self._RISK_CATEGORIES))

        for t in data["tickers"]:
            f = data["fundamentals"][t]
            r = data["ratios"][t]
            ss = data["sentiment_scores"].get(t, 0)
            prompt = (
                f"{system}\n\n"
                f"Company: {f.get('name', t)} ({t}) — Sector: {f.get('sector', 'N/A')}\n"
                f"Key Quantitative Metrics:\n"
                f"  P/E: {r['pe_ratio']:.1f} | P/B: {r['pb_ratio']:.1f} | "
                f"P/S: {r.get('ps_ratio', 0):.1f} | EPS: ${f.get('eps', 0):.2f}\n"
                f"  Debt/Equity: {r['debt_equity']:.2f} | Beta: {r['beta']:.2f}\n"
                f"  ROE: {r['roe']:.1f}% | Profit Margin: {r['profit_margin']:.1f}%\n"
                f"  Quality Score: {r['quality_score']}/100 | "
                f"Value Score: {r['value_score']}/100 | "
                f"Momentum Score: {r.get('momentum_score', 0)}/100\n"
                f"  News Sentiment Score: {ss:+.3f}\n\n"
                f"Assess the following risk categories:\n{cats}\n\n"
                "For each category provide:\n"
                "  - Risk Level: Low / Medium / High\n"
                "  - 1–2 sentence evidence-based rationale.\n\n"
                "End with:\n"
                "Overall Risk Rating: [1=Very Low | 2=Low | 3=Medium | 4=High | 5=Very High]\n"
                "Executive Risk Summary: (2–3 sentences)\n"
                "Risk Assessment:"
            )
            try:
                raw = generate(prompt)
                risk_reports[t] = sanitize_output(raw)
            except Exception as e:
                risk_reports[t] = (
                    f"Risk assessment temporarily unavailable ({type(e).__name__}). "
                    f"Quantitative risk rating: {r.get('risk_rating', 3)}/5 "
                    f"(Beta: {r['beta']:.2f}, D/E: {r['debt_equity']:.2f})"
                )
            overall_risk[t] = r.get("risk_rating", 3)

        data["risk_reports"]   = risk_reports
        data["overall_risk"]   = overall_risk
        return data


# ── Agent 5: Report Generation ────────────────────────────────────────────────
class ReportAgent:
    """Synthesises all agent outputs into a final investment research report."""

    def run(
        self,
        data: dict[str, Any],
        user_prefs: dict[str, Any],
        progress: ProgressFn = _noop,
    ) -> dict[str, Any]:
        progress("📝 ReportAgent: generating investment research reports with IBM Granite …", 86)
        reports: dict[str, str] = {}
        system  = build_system_prompt("analyst")
        citations = build_citations(data.get("data_sources", ["synthetic"]))

        for t in data["tickers"]:
            f   = data["fundamentals"][t]
            r   = data["ratios"][t]
            ss  = data["sentiment_summaries"].get(t, "N/A")
            rsk = data["risk_reports"].get(t, "N/A")

            prices = data["price_histories"].get(t, [])
            if len(prices) >= 2:
                first_close = prices[0]["close"]
                last_close  = prices[-1]["close"]
                ytd_ret     = round((last_close - first_close) / first_close * 100, 1)
                high_52w    = max(r["close"] for r in prices) if prices else 0
                low_52w     = min(r["close"] for r in prices) if prices else 0
            else:
                ytd_ret = 0.0
                high_52w = f.get("52w_high", 0)
                low_52w  = f.get("52w_low", 0)

            # Peer ranking context
            rank_pos = data.get("quality_rank", []).index(t) + 1 if t in data.get("quality_rank", []) else "N/A"
            peer_count = len(data["tickers"])

            prompt = (
                f"{system}\n\n"
                f"=== EQUITY RESEARCH REPORT: {f.get('name', t)} ({t}) ===\n\n"
                f"INVESTOR PROFILE:\n"
                f"  Investment Horizon: {user_prefs.get('horizon', 'Medium-term (1–3 years)')}\n"
                f"  Risk Tolerance: {user_prefs.get('risk_tolerance', 'Moderate')}\n"
                f"  Sectors of Interest: {', '.join(user_prefs.get('sectors', ['Technology']))}\n"
                f"  Portfolio Budget: ${user_prefs.get('amount', 50000):,}\n\n"
                f"FINANCIAL SNAPSHOT:\n"
                f"  Current Price: ${f.get('price', 0):.2f} | "
                f"Market Cap: ${f.get('market_cap_b', 0):.1f}B | "
                f"Revenue: ${f.get('revenue_b', 0):.1f}B\n"
                f"  YTD Return: {ytd_ret:+.1f}% | "
                f"52W Range: ${low_52w:.2f}–${high_52w:.2f}\n"
                f"  P/E: {r['pe_ratio']:.1f} | P/B: {r['pb_ratio']:.1f} | "
                f"P/S: {r.get('ps_ratio', 0):.1f} | "
                f"EPS: ${f.get('eps', 0):.2f} | Div Yield: {r['dividend_yield']:.2f}%\n"
                f"  Quality Score: {r['quality_score']}/100 | "
                f"Value Score: {r['value_score']}/100 | "
                f"Momentum Score: {r.get('momentum_score', 0)}/100 | "
                f"Risk Rating: {r['risk_rating']}/5\n"
                f"  Peer Quality Rank: #{rank_pos} of {peer_count}\n"
                f"  Analyst Rating: {f.get('analyst_rating', 'N/A')} | "
                f"Price Target: ${f.get('price_target', 0):.2f} | "
                f"Upside: {r.get('upside_pct', 0):+.1f}%\n\n"
                f"SENTIMENT ANALYSIS:\n{ss}\n\n"
                f"RISK ASSESSMENT:\n{rsk}\n\n"
                "Generate a professional equity research report with these exact sections:\n"
                "**1. Executive Summary** — 3–4 sentences with the investment thesis and key verdict.\n"
                "**2. Business Overview** — sector, competitive positioning, key products/services, moat.\n"
                "**3. Financial Analysis** — valuation multiples, profitability, balance sheet, growth.\n"
                "**4. Sentiment & News Analysis** — market perception, recent catalysts, media tone.\n"
                "**5. Risk Factors** — top 3 risks, each with a mitigation or monitoring comment.\n"
                "**6. Investment Scenarios**\n"
                "   - Bull Case: (key upside drivers, price target rationale)\n"
                "   - Base Case: (most likely outcome, return estimate)\n"
                "   - Bear Case: (downside risks, floor price estimate)\n"
                "**7. Key Metrics Scorecard** — brief bullet-point summary of all major ratios.\n"
                "**8. Investment Recommendation** — fit for the investor's horizon and risk profile.\n\n"
                "Be analytical, balanced, and professional. Avoid generic filler. Be specific.\n"
                "Report:"
            )
            try:
                raw = generate(prompt)
                report_text = sanitize_output(raw)
                # Append citations
                reports[t] = report_text + "\n\n---\n" + citations
            except Exception as e:
                reports[t] = (
                    f"⚠️ Report generation encountered an issue ({type(e).__name__}: {e}). "
                    "Please verify IBM Granite credentials and network connectivity.\n\n"
                    f"---\n{citations}"
                )

        data["reports"]      = reports
        data["user_prefs"]   = user_prefs
        data["completed_at"] = datetime.now().isoformat()
        return data


# ── Agent 6: Portfolio Analysis ───────────────────────────────────────────────
class PortfolioAgent:
    """Generates a portfolio-level summary, weighting, and allocation advice."""

    def run(
        self,
        data: dict[str, Any],
        progress: ProgressFn = _noop,
    ) -> dict[str, Any]:
        progress("💼 PortfolioAgent: generating portfolio analysis …", 95)
        tickers = data["tickers"]
        ratios  = data["ratios"]
        funds   = data["fundamentals"]
        prefs   = data.get("user_prefs", {})

        # Equal-weight baseline
        n = len(tickers)
        weights = [1.0 / n] * n if n > 0 else []

        # Portfolio-level metrics
        portfolio_metrics = calculate_portfolio_metrics(tickers, weights, ratios)

        # Quality-weighted allocation suggestion
        quality_scores = [ratios[t].get("quality_score", 50) for t in tickers]
        q_total = sum(quality_scores)
        quality_weights = (
            [q / q_total for q in quality_scores] if q_total > 0
            else [1.0 / n] * n
        )

        # LLM portfolio summary
        system  = build_system_prompt("portfolio")
        summary_lines = []
        for t in tickers:
            r = ratios[t]
            f = funds[t]
            summary_lines.append(
                f"- {f.get('name', t)} ({t}): "
                f"Price ${f.get('price', 0):.2f}, "
                f"P/E {r['pe_ratio']:.1f}, "
                f"Quality {r['quality_score']}/100, "
                f"Risk {r['risk_rating']}/5, "
                f"Sentiment {data['sentiment_scores'].get(t, 0):+.2f}"
            )

        prompt = (
            f"{system}\n\n"
            f"PORTFOLIO HOLDINGS ({n} stocks):\n"
            + "\n".join(summary_lines) + "\n\n"
            f"INVESTOR PROFILE:\n"
            f"  Horizon: {prefs.get('horizon', 'Medium-term')}\n"
            f"  Risk Tolerance: {prefs.get('risk_tolerance', 'Moderate')}\n"
            f"  Budget: ${prefs.get('amount', 50000):,}\n\n"
            f"PORTFOLIO METRICS (equal-weight):\n"
            f"  Weighted P/E: {portfolio_metrics.get('portfolio_pe', 0):.1f} | "
            f"  Beta: {portfolio_metrics.get('portfolio_beta', 0):.2f} | "
            f"  Quality: {portfolio_metrics.get('portfolio_quality', 0)}/100 | "
            f"  Risk: {portfolio_metrics.get('portfolio_risk', 0):.1f}/5\n\n"
            "Generate a portfolio-level analysis with these sections:\n"
            "**1. Portfolio Overview** — composition, diversification, sector balance.\n"
            "**2. Risk Profile Assessment** — overall risk level, correlation concerns, hedging.\n"
            "**3. Allocation Recommendation** — suggested weighting for each stock with rationale.\n"
            "**4. Portfolio Strengths** — 3 key positive attributes.\n"
            "**5. Portfolio Risks** — 3 key concerns or gaps.\n"
            "**6. Rebalancing Suggestions** — any adjustments to better match the investor profile.\n"
            "Portfolio Analysis:"
        )
        try:
            raw = generate(prompt)
            portfolio_summary = sanitize_output(raw)
        except Exception as e:
            portfolio_summary = (
                f"Portfolio-level summary temporarily unavailable ({type(e).__name__}).\n\n"
                f"Equal-weight metrics: "
                f"P/E {portfolio_metrics.get('portfolio_pe', 0):.1f} | "
                f"Beta {portfolio_metrics.get('portfolio_beta', 0):.2f} | "
                f"Quality {portfolio_metrics.get('portfolio_quality', 0)}/100"
            )

        data["portfolio_metrics"]     = portfolio_metrics
        data["portfolio_weights"]     = dict(zip(tickers, quality_weights))
        data["equal_weights"]         = dict(zip(tickers, weights))
        data["portfolio_summary"]     = portfolio_summary
        return data


# ── Conversational Advisor Agent ──────────────────────────────────────────────
class AdvisorAgent:
    """Powers the conversational AI assistant in the dashboard sidebar."""

    def __init__(self):
        self._system = build_system_prompt("advisor")

    def answer(
        self,
        question: str,
        context: dict[str, Any],
    ) -> str:
        from guardrails import check_input
        block = check_input(question)
        if block:
            return block

        tickers = context.get("tickers", [])
        snapshot = ""
        for t in tickers[:4]:
            f = context.get("fundamentals", {}).get(t, {})
            r = context.get("ratios", {}).get(t, {})
            if f:
                snapshot += (
                    f"\n• {f.get('name', t)} ({t}): "
                    f"${f.get('price', 0):.2f}, "
                    f"P/E {r.get('pe_ratio', 0):.1f}, "
                    f"Quality {r.get('quality_score', 0)}/100, "
                    f"Risk {r.get('risk_rating', 3)}/5, "
                    f"Sentiment {context.get('sentiment_scores', {}).get(t, 0):+.2f}"
                )

        prompt = (
            f"{self._system}\n\n"
            f"Current analysis context:{snapshot or ' No analysis loaded yet — answer generally.'}\n\n"
            f"User Question: {question}\n\n"
            "Provide a helpful, concise answer (2–5 sentences). "
            "Reference the specific data above where relevant. "
            "If you cannot answer from the data, say so and suggest what information would help.\n"
            "Answer:"
        )
        try:
            raw = generate(prompt)
            return sanitize_output(raw)
        except Exception as e:
            return (
                f"I encountered an issue reaching IBM Granite ({type(e).__name__}). "
                "Please try again in a moment, or check the connection to IBM watsonx.ai."
            )


# ── Orchestrator: runs the full pipeline ──────────────────────────────────────
class InvestmentOrchestrator:
    """
    Runs DataAgent → RatioAgent → SentimentAgent → RiskAgent →
    ReportAgent → PortfolioAgent sequentially, threading the output
    of each agent into the next.
    """

    def __init__(self):
        self.data_agent      = DataAgent()
        self.ratio_agent     = RatioAgent()
        self.sentiment_agent = SentimentAgent()
        self.risk_agent      = RiskAgent()
        self.report_agent    = ReportAgent()
        self.portfolio_agent = PortfolioAgent()

    def run(
        self,
        tickers: list[str],
        user_prefs: dict[str, Any],
        days: int = 365,
        progress: ProgressFn = _noop,
    ) -> dict[str, Any]:
        """Execute the full sequential agent pipeline and return the final context."""
        ctx = self.data_agent.run(tickers, days=days, progress=progress)
        ctx = self.ratio_agent.run(ctx, progress=progress)
        ctx = self.sentiment_agent.run(ctx, progress=progress)
        ctx = self.risk_agent.run(ctx, progress=progress)
        ctx = self.report_agent.run(ctx, user_prefs=user_prefs, progress=progress)
        ctx = self.portfolio_agent.run(ctx, progress=progress)
        progress("✅ All agents complete!", 100)
        return ctx
