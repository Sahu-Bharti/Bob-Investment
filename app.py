"""
Investment Agentic AI — Streamlit Web Application v2.0
Powered by IBM Granite LLM on IBM watsonx.ai

Run:
    cd investment_ai
    python -m streamlit run app.py
"""

from __future__ import annotations

import sys
import os
import io
import time
from datetime import datetime
from typing import Any

# ── Fix UTF-8 on Windows ─────────────────────────────────────────────────────
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

from config import (
    APP_TITLE, APP_ICON, APP_VERSION,
    DEMO_STOCKS, SECTORS, RISK_LEVELS,
    FINANCIAL_DISCLAIMER, ANALYST_COLORS,
)
from agents import InvestmentOrchestrator, AdvisorAgent
import synthetic_data as syn


# ═══════════════════════════════════════════════════════════════════════════════
# Page configuration
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help":     "https://www.ibm.com/watsonx",
        "Report a bug": None,
        "About":        f"**{APP_TITLE}** v{APP_VERSION}\nPowered by IBM Granite on watsonx.ai",
    },
)


# ═══════════════════════════════════════════════════════════════════════════════
# Custom CSS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Global ── */
[data-testid="stAppViewContainer"] { background: #0d1117; color: #e6edf3; }
[data-testid="stSidebar"]          { background: #161b22; border-right: 1px solid #30363d; }
[data-testid="stHeader"]           { background: transparent; }
html, body, [class*="css"]         { font-family: -apple-system,"Segoe UI",system-ui,sans-serif; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 12px 16px;
}

/* ── Tabs ── */
button[data-baseweb="tab"]          { color: #8b949e !important; font-size: 13px; font-weight: 500; }
button[data-baseweb="tab"][aria-selected="true"] {
    color: #58a6ff !important;
    border-bottom: 2px solid #58a6ff !important;
    font-weight: 600 !important;
}

/* ── Section headers ── */
.section-header {
    font-size: 1.05rem;
    font-weight: 600;
    color: #58a6ff;
    border-bottom: 1px solid #30363d;
    padding-bottom: 6px;
    margin: 18px 0 12px 0;
}

/* ── Cards ── */
.card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 12px;
}
.card-accent {
    background: #0d2137;
    border: 1px solid #1f6feb;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 12px;
}

/* ── Report box ── */
.report-box {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 22px 26px;
    line-height: 1.75;
    font-size: 0.91rem;
    white-space: pre-wrap;
}

/* ── Risk badge ── */
.risk-1 { color: #3fb950; font-weight: 700; }
.risk-2 { color: #56d364; font-weight: 700; }
.risk-3 { color: #d29922; font-weight: 700; }
.risk-4 { color: #f85149; font-weight: 700; }
.risk-5 { color: #ff7b72; font-weight: 700; }

/* ── Disclaimer box ── */
.disclaimer {
    background: #161b22;
    border-left: 4px solid #d29922;
    border-radius: 4px;
    padding: 12px 16px;
    font-size: 0.82rem;
    color: #8b949e;
    margin-top: 24px;
}

/* ── Chat message ── */
.chat-user {
    background: #1c2128;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 0.9rem;
}
.chat-bot {
    background: #0d2137;
    border: 1px solid #1f6feb;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 0.9rem;
}

/* ── Sentiment pill ── */
.pill-pos { background:#1a472a; color:#3fb950; padding:2px 8px; border-radius:12px; font-size:0.8rem; font-weight:600; }
.pill-neg { background:#4a1414; color:#f85149; padding:2px 8px; border-radius:12px; font-size:0.8rem; font-weight:600; }
.pill-neu { background:#2d2d2d; color:#8b949e; padding:2px 8px; border-radius:12px; font-size:0.8rem; font-weight:600; }

/* ── Agent pipeline card ── */
.agent-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 12px 10px;
    text-align: center;
    height: 100%;
}

/* ── Score bar ── */
.score-bar-wrap { background:#21262d; border-radius:6px; height:8px; width:100%; margin-top:4px; }
.score-bar-fill { border-radius:6px; height:8px; }

/* ── IBM brand footer ── */
.ibm-footer {
    text-align: center;
    font-size: 0.74rem;
    color: #484f58;
    padding: 24px 0 10px 0;
    border-top: 1px solid #21262d;
    margin-top: 32px;
}

/* ── Landing feature card ── */
.feature-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 22px 20px;
    text-align: center;
    height: 100%;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Session state initialisation
# ═══════════════════════════════════════════════════════════════════════════════
def _init_state():
    defaults = {
        "analysis_done":  False,
        "context":        {},
        "chat_history":   [],
        "advisor":        None,   # lazy init
        "orchestrator":   None,   # lazy init
        "active_ticker":  None,
        "preferences":    {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Lazy-init heavy objects only when first needed
    if st.session_state["advisor"] is None:
        st.session_state["advisor"] = AdvisorAgent()
    if st.session_state["orchestrator"] is None:
        st.session_state["orchestrator"] = InvestmentOrchestrator()


_init_state()


# ═══════════════════════════════════════════════════════════════════════════════
# Plotly dark theme base
# ═══════════════════════════════════════════════════════════════════════════════
_DARK = dict(
    paper_bgcolor="#0d1117",
    plot_bgcolor="#0d1117",
    font=dict(color="#e6edf3", size=12),
    xaxis=dict(gridcolor="#21262d", linecolor="#30363d", zerolinecolor="#30363d"),
    yaxis=dict(gridcolor="#21262d", linecolor="#30363d", zerolinecolor="#30363d"),
)


# ═══════════════════════════════════════════════════════════════════════════════
# Chart helpers
# ═══════════════════════════════════════════════════════════════════════════════
def chart_price_history(ticker: str, records: list[dict]) -> go.Figure:
    if not records:
        return go.Figure()
    dates  = [r["date"]  for r in records]
    opens  = [r["open"]  for r in records]
    highs  = [r["high"]  for r in records]
    lows   = [r["low"]   for r in records]
    closes = [r["close"] for r in records]
    vols   = [r["volume"] for r in records]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.03,
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=dates, open=opens, high=highs, low=lows, close=closes,
        name=ticker,
        increasing_line_color="#3fb950",
        decreasing_line_color="#f85149",
        increasing_fillcolor="#1a472a",
        decreasing_fillcolor="#4a1414",
    ), row=1, col=1)

    # Moving averages
    s = pd.Series(closes)
    for window, colour, dash in [(20, "#58a6ff", "dot"), (50, "#d29922", "dash")]:
        if len(closes) >= window:
            ma = s.rolling(window).mean().tolist()
            fig.add_trace(go.Scatter(
                x=dates, y=ma, name=f"MA{window}",
                line=dict(color=colour, width=1.5, dash=dash),
            ), row=1, col=1)

    # Bollinger Bands (20-day)
    if len(closes) >= 20:
        ma20  = s.rolling(20).mean()
        std20 = s.rolling(20).std()
        upper = (ma20 + 2 * std20).tolist()
        lower = (ma20 - 2 * std20).tolist()
        fig.add_trace(go.Scatter(
            x=dates, y=upper, name="BB Upper",
            line=dict(color="rgba(88,166,255,0.3)", width=1),
            showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=dates, y=lower, name="BB Lower",
            fill="tonexty", fillcolor="rgba(88,166,255,0.05)",
            line=dict(color="rgba(88,166,255,0.3)", width=1),
            showlegend=False,
        ), row=1, col=1)

    # Volume bars
    vol_colors = [
        "#3fb950" if i == 0 or closes[i] >= closes[i - 1] else "#f85149"
        for i in range(len(closes))
    ]
    fig.add_trace(go.Bar(
        x=dates, y=vols, name="Volume",
        marker_color=vol_colors, opacity=0.65,
    ), row=2, col=1)

    fig.update_layout(
        **_DARK,
        title=dict(text=f"{ticker} — Price History & Volume", font=dict(size=14)),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        height=520,
        margin=dict(l=0, r=0, t=50, b=0),
    )
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Volume",      row=2, col=1)
    return fig


def chart_radar(ticker: str, ratios: dict) -> go.Figure:
    categories = ["Quality", "Value", "ROE", "Margin", "Low Risk", "Momentum"]
    roe_n      = min(ratios.get("roe", 0) / 60 * 100, 100)
    mar_n      = min(ratios.get("profit_margin", 0) / 40 * 100, 100)
    risk_n     = max(0, 100 - ratios.get("risk_rating", 3) / 5 * 100)
    mom_n      = ratios.get("momentum_score", 50)

    values = [
        ratios.get("quality_score", 50),
        ratios.get("value_score",   50),
        roe_n, mar_n, risk_n, mom_n,
    ]

    fig = go.Figure(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor="rgba(88, 166, 255, 0.15)",
        line=dict(color="#58a6ff", width=2),
        name=ticker,
    ))
    fig.update_layout(
        **_DARK,
        polar=dict(
            bgcolor="#161b22",
            radialaxis=dict(visible=True, range=[0, 100], color="#484f58", tickfont=dict(size=9)),
            angularaxis=dict(color="#8b949e", tickfont=dict(size=10)),
        ),
        title=dict(text=f"{ticker} — Scorecard", font=dict(size=13)),
        height=320,
        margin=dict(l=40, r=40, t=50, b=20),
    )
    return fig


def chart_comparison_bar(comp_data: list[dict], metric: str, label: str) -> go.Figure:
    tickers = [d["ticker"] for d in comp_data]
    values  = [d.get(metric, 0) or 0 for d in comp_data]
    colors  = px.colors.qualitative.Plotly[:len(tickers)]

    fig = go.Figure(go.Bar(
        x=tickers, y=values,
        marker_color=colors,
        text=[f"{v:.1f}" for v in values],
        textposition="outside",
        textfont=dict(size=11),
    ))
    fig.update_layout(
        **_DARK,
        title=dict(text=label, font=dict(size=13)),
        height=300,
        margin=dict(l=0, r=0, t=42, b=0),
        showlegend=False,
        yaxis=dict(gridcolor="#21262d"),
    )
    return fig


def chart_sentiment_gauge(score: float, ticker: str) -> go.Figure:
    color = "#3fb950" if score > 0.2 else "#f85149" if score < -0.2 else "#d29922"
    label = "Positive" if score > 0.2 else "Negative" if score < -0.2 else "Neutral"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        delta=dict(reference=0, valueformat=".3f"),
        number=dict(font=dict(color=color, size=26), valueformat=".3f"),
        gauge=dict(
            axis=dict(range=[-1, 1], tickcolor="#484f58", tickwidth=1,
                      tickvals=[-1, -0.5, 0, 0.5, 1],
                      ticktext=["-1.0", "-0.5", "0", "+0.5", "+1.0"]),
            bar=dict(color=color, thickness=0.28),
            bgcolor="#161b22",
            borderwidth=1,
            bordercolor="#30363d",
            steps=[
                dict(range=[-1.0, -0.2], color="#4a1414"),
                dict(range=[-0.2,  0.2], color="#2d2d2d"),
                dict(range=[ 0.2,  1.0], color="#1a472a"),
            ],
        ),
        title=dict(text=f"{ticker} Sentiment — {label}", font=dict(color="#8b949e", size=12)),
    ))
    fig.update_layout(
        **_DARK,
        height=240,
        margin=dict(l=20, r=20, t=40, b=10),
    )
    return fig


def chart_risk_heatmap(tickers: list[str], risks: dict[str, int]) -> go.Figure:
    vals   = [[risks.get(t, 3) for t in tickers]]
    labels = [[f"{risks.get(t,3)}/5" for t in tickers]]
    fig = go.Figure(go.Heatmap(
        z=vals,
        x=tickers,
        y=["Risk"],
        text=labels,
        texttemplate="%{text}",
        colorscale=[[0,"#1a472a"],[0.25,"#56d364"],[0.5,"#d29922"],[0.75,"#f85149"],[1.0,"#ff7b72"]],
        zmin=1, zmax=5,
        showscale=True,
        colorbar=dict(title="Risk 1-5", tickcolor="#484f58",
                      titlefont=dict(color="#8b949e", size=11)),
        textfont=dict(size=13, color="#e6edf3"),
    ))
    fig.update_layout(
        **_DARK,
        title=dict(text="Risk Heatmap — All Holdings", font=dict(size=13)),
        height=160,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def chart_allocation_donut(risk_profile: str) -> go.Figure:
    cfg    = RISK_LEVELS.get(risk_profile, RISK_LEVELS["Moderate"])
    labels = ["Equities", "Bonds", "Cash"]
    values = [cfg["equity_pct"], cfg["bond_pct"], cfg["cash_pct"]]
    colors = ["#58a6ff", "#3fb950", "#d29922"]

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.58,
        marker=dict(colors=colors, line=dict(color="#0d1117", width=2)),
        textinfo="label+percent",
        textfont=dict(color="#e6edf3", size=12),
        hovertemplate="%{label}: %{value}%<extra></extra>",
    ))
    fig.update_layout(
        **_DARK,
        title=dict(text=f"{risk_profile} Allocation", font=dict(size=13)),
        height=290,
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
    )
    return fig


def chart_portfolio_weights(tickers: list[str], weights: dict) -> go.Figure:
    w_vals  = [round(weights.get(t, 0) * 100, 1) for t in tickers]
    colors  = px.colors.qualitative.Plotly[:len(tickers)]
    fig = go.Figure(go.Pie(
        labels=tickers, values=w_vals,
        hole=0.5,
        marker=dict(colors=colors, line=dict(color="#0d1117", width=2)),
        textinfo="label+percent",
        textfont=dict(color="#e6edf3", size=12),
        hovertemplate="%{label}: %{value}%<extra></extra>",
    ))
    fig.update_layout(
        **_DARK,
        title=dict(text="Quality-Weighted Portfolio Allocation", font=dict(size=13)),
        height=300,
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
    )
    return fig


def chart_portfolio_performance(tickers: list[str], weights: dict, days: int) -> go.Figure:
    w_list = [weights.get(t, 1 / len(tickers)) for t in tickers]
    perf   = syn.generate_portfolio_performance(tickers, w_list, days=days)
    if not perf:
        return go.Figure()

    dates  = [p["date"]  for p in perf]
    values = [p["value"] for p in perf]
    first  = values[0] if values else 10000
    pct_change = [(v - first) / first * 100 for v in values]
    color  = "#3fb950" if pct_change[-1] >= 0 else "#f85149"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=pct_change,
        fill="tozeroy",
        fillcolor=f"rgba({'63,185,80' if pct_change[-1] >= 0 else '248,81,73'},0.12)",
        line=dict(color=color, width=2),
        name="Portfolio Return (%)",
        hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#484f58", line_width=1)
    fig.update_layout(
        **_DARK,
        title=dict(text="Simulated Portfolio Return (Quality-Weighted)", font=dict(size=13)),
        yaxis_title="Return (%)",
        height=320,
        margin=dict(l=0, r=0, t=45, b=0),
    )
    return fig


def chart_scatter_quality_risk(tickers: list[str], ratios: dict, funds: dict) -> go.Figure:
    """Quality score vs Risk rating bubble chart."""
    x_vals    = [ratios[t].get("quality_score", 50) for t in tickers]
    y_vals    = [ratios[t].get("risk_rating", 3)     for t in tickers]
    sizes     = [max(funds[t].get("market_cap_b", 10), 5) for t in tickers]
    max_size  = max(sizes)
    norm_size = [20 + (s / max_size) * 50 for s in sizes]
    colors    = px.colors.qualitative.Plotly[:len(tickers)]

    fig = go.Figure()
    for i, t in enumerate(tickers):
        fig.add_trace(go.Scatter(
            x=[x_vals[i]], y=[y_vals[i]],
            mode="markers+text",
            marker=dict(size=norm_size[i], color=colors[i], opacity=0.85,
                        line=dict(color="#0d1117", width=1)),
            text=[t],
            textposition="top center",
            textfont=dict(size=11, color="#e6edf3"),
            name=t,
            hovertemplate=(
                f"<b>{t}</b><br>Quality: {x_vals[i]}<br>"
                f"Risk: {y_vals[i]}/5<br>"
                f"Mkt Cap: ${funds[t].get('market_cap_b', 0):.0f}B<extra></extra>"
            ),
        ))

    fig.update_layout(
        **_DARK,
        title=dict(text="Quality vs Risk (bubble = market cap)", font=dict(size=13)),
        xaxis_title="Quality Score (0–100)",
        yaxis_title="Risk Rating (1–5)",
        yaxis=dict(range=[0.5, 5.5], dtick=1, gridcolor="#21262d"),
        xaxis=dict(range=[0, 105], gridcolor="#21262d"),
        height=350,
        showlegend=False,
        margin=dict(l=0, r=0, t=45, b=0),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════════
def render_sidebar() -> dict[str, Any] | None:
    with st.sidebar:
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:6px'>"
            f"<span style='font-size:1.8rem'>{APP_ICON}</span>"
            f"<div><div style='font-weight:700;font-size:1.05rem'>{APP_TITLE}</div>"
            f"<div style='color:#8b949e;font-size:0.75rem'>v{APP_VERSION} · IBM Granite</div></div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown("---")

        # ── Stock selection
        st.markdown("#### 🎯 Stock Selection")
        all_tickers = list(DEMO_STOCKS.keys())
        selected_tickers = st.multiselect(
            "Select Stocks (1–6)",
            options=all_tickers,
            default=["AAPL", "MSFT", "NVDA"],
            format_func=lambda t: f"{t} — {DEMO_STOCKS.get(t, t)}",
            max_selections=6,
        )

        # ── Investment preferences
        st.markdown("#### 📅 Investment Horizon")
        horizon = st.selectbox(
            "Time Horizon",
            ["Short-term (< 1 year)", "Medium-term (1–3 years)", "Long-term (3+ years)"],
            index=1,
            label_visibility="collapsed",
        )

        st.markdown("#### ⚖️ Risk Tolerance")
        risk_tolerance = st.select_slider(
            "Risk Tolerance",
            options=["Conservative", "Moderate", "Aggressive"],
            value="Moderate",
            label_visibility="collapsed",
        )
        cfg = RISK_LEVELS[risk_tolerance]
        color = cfg.get("color", "#d29922")
        st.markdown(
            f"<div style='font-size:0.78rem;color:{color};padding:4px 8px;"
            f"background:#161b22;border-radius:6px;border:1px solid #30363d'>"
            f"{cfg['description']}</div>",
            unsafe_allow_html=True,
        )

        st.markdown("#### 🏭 Sectors of Interest")
        sectors = st.multiselect(
            "Sectors",
            options=SECTORS,
            default=["Technology", "Healthcare"],
            label_visibility="collapsed",
        )

        st.markdown("#### 💵 Portfolio Size (USD)")
        amount = st.number_input(
            "Portfolio Size",
            min_value=1_000,
            max_value=10_000_000,
            value=50_000,
            step=5_000,
            format="%d",
            label_visibility="collapsed",
        )

        st.markdown("#### 📊 Historical Period")
        days = st.selectbox(
            "Period",
            options=[90, 180, 365, 730],
            format_func=lambda d: {90: "3 Months", 180: "6 Months",
                                   365: "1 Year", 730: "2 Years"}[d],
            index=2,
            label_visibility="collapsed",
        )

        st.markdown("---")

        run_btn = st.button(
            "🚀 Run Investment Analysis",
            type="primary",
            use_container_width=True,
            disabled=not selected_tickers,
        )
        if not selected_tickers:
            st.warning("Select at least one stock to continue.")

        st.markdown("---")

        # ── AI Chat
        st.markdown("#### 🤖 AI Chat Assistant")
        with st.expander("Chat with IBM Granite", expanded=False):
            _render_chat()

        st.markdown(
            '<div class="ibm-footer">IBM Granite 3 8B · watsonx.ai<br>'
            'IBM Cloud · au-syd · Project dc89f050</div>',
            unsafe_allow_html=True,
        )

        if run_btn and selected_tickers:
            return {
                "tickers":        selected_tickers,
                "horizon":        horizon,
                "risk_tolerance": risk_tolerance,
                "sectors":        sectors,
                "amount":         amount,
                "days":           days,
            }
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Chat widget
# ═══════════════════════════════════════════════════════════════════════════════
def _render_chat():
    history = st.session_state.chat_history
    for msg in history[-8:]:
        role_class = "chat-user" if msg["role"] == "user" else "chat-bot"
        icon = "👤" if msg["role"] == "user" else "🤖"
        st.markdown(
            f'<div class="{role_class}">{icon} {msg["content"]}</div>',
            unsafe_allow_html=True,
        )

    user_q = st.text_input(
        "Ask about the analysis…",
        key="chat_input",
        placeholder="e.g. Which stock has the best value score?",
        label_visibility="collapsed",
    )
    col1, col2 = st.columns([3, 1])
    with col1:
        send = st.button("Send ➤", key="send_btn", use_container_width=True)
    with col2:
        if st.button("🗑", key="clear_chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    if send and user_q.strip():
        history.append({"role": "user", "content": user_q.strip()})
        with st.spinner("IBM Granite is thinking…"):
            reply = st.session_state.advisor.answer(
                user_q.strip(),
                st.session_state.context,
            )
        history.append({"role": "assistant", "content": reply})
        st.session_state.chat_history = history
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# Header
# ═══════════════════════════════════════════════════════════════════════════════
def render_header():
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.title(f"{APP_ICON} Investment Agentic AI")
        st.caption(
            "Sequential AI Agents · IBM Granite LLM · IBM watsonx.ai · "
            "Financial Research · Risk Assessment · Portfolio Analytics"
        )
    with col2:
        st.metric("Platform", "IBM watsonx.ai")
    with col3:
        now = datetime.now().strftime("%d %b %Y %H:%M")
        st.metric("Last Updated", now)


# ═══════════════════════════════════════════════════════════════════════════════
# Market overview bar
# ═══════════════════════════════════════════════════════════════════════════════
def render_market_overview(market: dict):
    st.markdown('<div class="section-header">🌍 Live Market Overview</div>', unsafe_allow_html=True)
    indices = market.get("indices", {})
    macro   = market.get("macro", {})

    cols = st.columns(len(indices))
    for i, (name, data) in enumerate(indices.items()):
        delta_str   = f"{data['change_pct']:+.2f}%"
        delta_color = "normal" if data["change_pct"] >= 0 else "inverse"
        if name == "VIX":
            delta_color = "inverse" if data["change_pct"] >= 0 else "normal"
        cols[i].metric(
            label=name,
            value=f"{data['level']:,.0f}" if name != "VIX" else f"{data['level']:.1f}",
            delta=delta_str,
            delta_color=delta_color,
        )

    with st.expander("📊 Macro Indicators", expanded=False):
        m_cols = st.columns(len(macro))
        for i, (k, v) in enumerate(macro.items()):
            m_cols[i].metric(k, f"{v}%")


# ═══════════════════════════════════════════════════════════════════════════════
# Agent pipeline status
# ═══════════════════════════════════════════════════════════════════════════════
def render_agent_pipeline_status(done: bool = False):
    agents = [
        ("📡", "Data Agent",      "Fetch prices · fundamentals · news"),
        ("🔢", "Ratio Agent",     "Compute ratios · quality · value · momentum"),
        ("📰", "Sentiment Agent", "IBM Granite · news sentiment analysis"),
        ("⚠️",  "Risk Agent",     "IBM Granite · multi-dimensional risk"),
        ("📝", "Report Agent",    "IBM Granite · full research report"),
        ("💼", "Portfolio Agent", "IBM Granite · portfolio analysis"),
    ]
    st.markdown('<div class="section-header">🤖 Agent Pipeline</div>', unsafe_allow_html=True)
    cols = st.columns(6)
    for i, (icon, name, desc) in enumerate(agents):
        with cols[i]:
            color  = "#3fb950" if done else "#58a6ff"
            status = "✅ Done" if done else "⏳ Ready"
            st.markdown(
                f"<div class='agent-card'>"
                f"<div style='font-size:1.5rem'>{icon}</div>"
                f"<div style='color:{color};font-weight:600;font-size:0.82rem;margin:4px 0'>{name}</div>"
                f"<div style='color:#484f58;font-size:0.68rem;line-height:1.4'>{desc}</div>"
                f"<div style='color:{color};font-size:0.72rem;margin-top:6px'>{status}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Tab: Overview
# ═══════════════════════════════════════════════════════════════════════════════
def render_overview_tab(ctx: dict):
    tickers  = ctx["tickers"]
    funds    = ctx["fundamentals"]
    ratios   = ctx["ratios"]
    risks    = ctx["overall_risk"]
    rank     = ctx["quality_rank"]
    sent_sc  = ctx["sentiment_scores"]

    # ── Snapshot cards
    st.markdown('<div class="section-header">📈 Holdings Snapshot</div>', unsafe_allow_html=True)
    for t in tickers:
        f     = funds[t]
        r     = ratios[t]
        risk_n = risks.get(t, 3)
        sent   = sent_sc.get(t, 0)
        price  = f.get("price", 0)
        hist   = ctx["price_histories"].get(t, [])
        ytd = 0.0
        if len(hist) >= 2:
            ytd = round((hist[-1]["close"] - hist[0]["close"]) / hist[0]["close"] * 100, 1)

        c1, c2, c3, c4, c5, c6, c7, c8 = st.columns([1.8, 1, 1, 1, 1, 1, 1, 1])
        with c1:
            analyst = f.get("analyst_rating", "N/A")
            a_color = ANALYST_COLORS.get(analyst, "#8b949e")
            st.markdown(
                f"**{f.get('name', t)}**  \n"
                f"`{t}` &nbsp;·&nbsp; {f.get('sector', '')}  \n"
                f"<span style='color:{a_color};font-size:0.78rem'>{analyst}</span>",
                unsafe_allow_html=True,
            )
        ytd_delta = f"{ytd:+.1f}%"
        c2.metric("Price",    f"${price:.2f}",          ytd_delta)
        c3.metric("P/E",      f"{r['pe_ratio']:.1f}",   f"P/B {r['pb_ratio']:.1f}")
        c4.metric("Quality",  f"{r['quality_score']}/100")
        c5.metric("Value",    f"{r['value_score']}/100")
        c6.metric("Momentum", f"{r.get('momentum_score', 0)}/100")
        risk_emj = ["🟢", "🟢", "🟡", "🟠", "🔴"][risk_n - 1]
        c7.metric("Risk", f"{risk_emj} {risk_n}/5")
        sent_emj = "📈" if sent > 0.2 else ("📉" if sent < -0.2 else "➡️")
        c8.metric("Sentiment", f"{sent_emj} {sent:+.2f}")
        st.markdown("---")

    # ── Risk heatmap
    if len(tickers) > 1:
        st.plotly_chart(chart_risk_heatmap(tickers, risks), use_container_width=True)

    # ── Allocation + ranking
    prefs = ctx.get("user_prefs", {})
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown('<div class="section-header">💼 Suggested Asset Allocation</div>', unsafe_allow_html=True)
        st.plotly_chart(
            chart_allocation_donut(prefs.get("risk_tolerance", "Moderate")),
            use_container_width=True,
        )
    with c2:
        st.markdown('<div class="section-header">🏆 Quality Ranking (Peer Group)</div>', unsafe_allow_html=True)
        rank_rows = []
        for pos, t in enumerate(rank, 1):
            f = funds[t]
            r = ratios[t]
            rank_rows.append({
                "Rank":     pos,
                "Ticker":   t,
                "Company":  f.get("name", t),
                "Quality":  r["quality_score"],
                "Value":    r["value_score"],
                "Momentum": r.get("momentum_score", 0),
                "Risk":     f"{risks.get(t, 3)}/5",
                "Beta":     r["beta"],
                "ROE %":    r["roe"],
                "P/E":      r["pe_ratio"],
                "Analyst":  f.get("analyst_rating", "N/A"),
            })
        df_rank = pd.DataFrame(rank_rows).set_index("Rank")
        st.dataframe(df_rank, use_container_width=True, height=350)

    # ── Quality vs Risk bubble
    if len(tickers) > 1:
        st.markdown('<div class="section-header">🔵 Quality vs Risk Analysis</div>', unsafe_allow_html=True)
        st.plotly_chart(
            chart_scatter_quality_risk(tickers, ratios, funds),
            use_container_width=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Tab: Charts
# ═══════════════════════════════════════════════════════════════════════════════
def render_charts_tab(ctx: dict):
    tickers = ctx["tickers"]

    active = st.selectbox(
        "Select Stock",
        tickers,
        format_func=lambda t: f"{t} — {ctx['fundamentals'].get(t, {}).get('name', t)}",
        key="chart_ticker_select",
    )
    st.session_state.active_ticker = active

    if active:
        records = ctx["price_histories"].get(active, [])
        col1, col2 = st.columns([3, 1])
        with col1:
            st.plotly_chart(chart_price_history(active, records), use_container_width=True)
        with col2:
            st.plotly_chart(chart_radar(active, ctx["ratios"].get(active, {})), use_container_width=True)
            sc = ctx["sentiment_scores"].get(active, 0.0)
            st.plotly_chart(chart_sentiment_gauge(sc, active), use_container_width=True)

        # Price stats
        if records:
            closes  = [r["close"] for r in records]
            volumes = [r["volume"] for r in records]
            s_col1, s_col2, s_col3, s_col4, s_col5 = st.columns(5)
            s_col1.metric("Current",   f"${closes[-1]:.2f}")
            s_col2.metric("52W High",  f"${max(closes):.2f}")
            s_col3.metric("52W Low",   f"${min(closes):.2f}")
            s_col4.metric("Avg Volume", f"{sum(volumes)//len(volumes):,.0f}")
            ytd = round((closes[-1] - closes[0]) / closes[0] * 100, 2)
            s_col5.metric("Period Return", f"{ytd:+.2f}%")

    # ── Peer comparison charts
    if len(tickers) > 1:
        st.markdown("---")
        st.markdown('<div class="section-header">📊 Peer Comparison Charts</div>', unsafe_allow_html=True)
        comp_data = [
            {"ticker": t, **ctx["fundamentals"][t], **ctx["ratios"][t]}
            for t in tickers
        ]
        metric_pairs = [
            ("quality_score",  "Quality Score (0–100)"),
            ("value_score",    "Value Score (0–100)"),
            ("momentum_score", "Momentum Score (0–100)"),
            ("pe_ratio",       "P/E Ratio"),
            ("roe",            "Return on Equity (%)"),
            ("profit_margin",  "Profit Margin (%)"),
            ("debt_equity",    "Debt / Equity Ratio"),
            ("dividend_yield", "Dividend Yield (%)"),
        ]
        cols = st.columns(2)
        for i, (metric, label) in enumerate(metric_pairs):
            with cols[i % 2]:
                st.plotly_chart(chart_comparison_bar(comp_data, metric, label), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Tab: Fundamentals
# ═══════════════════════════════════════════════════════════════════════════════
def render_fundamentals_tab(ctx: dict):
    tickers = ctx["tickers"]
    rows = []
    for t in tickers:
        f = ctx["fundamentals"][t]
        r = ctx["ratios"][t]
        rows.append({
            "Ticker":         t,
            "Company":        f.get("name", t),
            "Sector":         f.get("sector", ""),
            "Price ($)":      f.get("price", 0),
            "Mkt Cap ($B)":   f.get("market_cap_b", 0),
            "Revenue ($B)":   f.get("revenue_b", 0),
            "P/E":            r["pe_ratio"],
            "P/B":            r["pb_ratio"],
            "P/S":            r.get("ps_ratio", 0),
            "EPS ($)":        f.get("eps", 0),
            "ROE (%)":        r["roe"],
            "Margin (%)":     r["profit_margin"],
            "D/E":            r["debt_equity"],
            "Beta":           r["beta"],
            "Div Yield (%)":  r["dividend_yield"],
            "Graham ($)":     r["graham_number"],
            "Quality":        r["quality_score"],
            "Value":          r["value_score"],
            "Momentum":       r.get("momentum_score", 0),
            "Risk (1-5)":     r["risk_rating"],
            "Upside (%)":     r.get("upside_pct", 0),
            "Analyst":        f.get("analyst_rating", "N/A"),
            "Target ($)":     f.get("price_target", 0),
        })

    df = pd.DataFrame(rows).set_index("Ticker")
    st.dataframe(df.style.format(precision=2), use_container_width=True, height=420)

    col1, col2 = st.columns(2)
    with col1:
        csv = df.to_csv()
        st.download_button(
            "⬇️ Download Fundamentals CSV",
            data=csv,
            file_name="investment_ai_fundamentals.csv",
            mime="text/csv",
        )
    with col2:
        import json
        json_data = df.reset_index().to_json(orient="records", indent=2)
        st.download_button(
            "⬇️ Download Fundamentals JSON",
            data=json_data,
            file_name="investment_ai_fundamentals.json",
            mime="application/json",
        )

    # Graham Number analysis
    st.markdown('<div class="section-header">📐 Graham Number Analysis</div>', unsafe_allow_html=True)
    graham_rows = []
    for t in tickers:
        f = ctx["fundamentals"][t]
        r = ctx["ratios"][t]
        graham = r.get("graham_number", 0)
        price  = f.get("price", 0)
        margin_of_safety = round((graham - price) / price * 100, 1) if price > 0 and graham > 0 else 0
        graham_rows.append({
            "Ticker":     t,
            "Price ($)":  price,
            "Graham ($)": graham,
            "MoS (%)":    margin_of_safety,
            "Signal":     "Undervalued" if margin_of_safety > 15 else
                          "Fairly Valued" if margin_of_safety > -15 else "Overvalued",
        })
    df_g = pd.DataFrame(graham_rows).set_index("Ticker")
    st.dataframe(df_g.style.format(precision=2), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Tab: News & Sentiment
# ═══════════════════════════════════════════════════════════════════════════════
def render_news_tab(ctx: dict):
    tickers = ctx["tickers"]
    active  = st.selectbox(
        "Select Company",
        tickers,
        format_func=lambda t: f"{t} — {ctx['fundamentals'].get(t, {}).get('name', t)}",
        key="news_ticker_select",
    )
    if not active:
        return

    articles = ctx["news"].get(active, [])
    ss       = ctx["sentiment_summaries"].get(active, "")
    sc       = ctx["sentiment_scores"].get(active, 0.0)

    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown(
            '<div class="section-header">📰 Sentiment Summary — IBM Granite Analysis</div>',
            unsafe_allow_html=True,
        )
        st.markdown(f'<div class="report-box">{ss}</div>', unsafe_allow_html=True)
    with c2:
        st.plotly_chart(chart_sentiment_gauge(sc, active), use_container_width=True)
        # Sentiment distribution
        pos_count = sum(1 for a in articles if a["sentiment_label"] == "Positive")
        neu_count = sum(1 for a in articles if a["sentiment_label"] == "Neutral")
        neg_count = sum(1 for a in articles if a["sentiment_label"] == "Negative")
        st.markdown(
            f"<div class='card'>"
            f"<div style='font-size:0.8rem;color:#8b949e;margin-bottom:8px'>Headline Breakdown</div>"
            f"<div style='display:flex;gap:8px;flex-wrap:wrap'>"
            f"<span class='pill-pos'>✓ {pos_count} Positive</span>"
            f"<span class='pill-neu'>≈ {neu_count} Neutral</span>"
            f"<span class='pill-neg'>✗ {neg_count} Negative</span>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-header">📋 Recent Headlines</div>', unsafe_allow_html=True)
    for art in articles:
        pill_class = {
            "Positive": "pill-pos",
            "Negative": "pill-neg",
            "Neutral":  "pill-neu",
        }.get(art["sentiment_label"], "pill-neu")
        pill     = f'<span class="{pill_class}">{art["sentiment_label"]} ({art["sentiment_score"]:+.2f})</span>'
        src      = art.get("source", "")
        date     = art.get("published_date", "")
        url      = art.get("url", "#")
        headline = art.get("headline", "")
        st.markdown(
            f"{pill} &nbsp;<a href='{url}' style='color:#c9d1d9'>{headline}</a>"
            f" &nbsp;<span style='color:#484f58;font-size:0.78rem'>— {src} · {date}</span>",
            unsafe_allow_html=True,
        )
        st.markdown("<hr style='border-color:#21262d;margin:6px 0'>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Tab: Risk Assessment
# ═══════════════════════════════════════════════════════════════════════════════
def render_risk_tab(ctx: dict):
    tickers = ctx["tickers"]
    active  = st.selectbox(
        "Select Company",
        tickers,
        format_func=lambda t: f"{t} — {ctx['fundamentals'].get(t, {}).get('name', t)}",
        key="risk_ticker_select",
    )
    if not active:
        return

    risk_text = ctx["risk_reports"].get(active, "")
    risk_n    = ctx["overall_risk"].get(active, 3)
    r         = ctx["ratios"].get(active, {})
    f         = ctx["fundamentals"].get(active, {})

    risk_labels = ["Very Low", "Low", "Medium", "High", "Very High"]
    risk_colors = ["#3fb950", "#56d364", "#d29922", "#f85149", "#ff7b72"]
    rl     = risk_labels[risk_n - 1]
    rc     = risk_colors[risk_n - 1]

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown('<div class="section-header">🎯 Risk Scorecard</div>', unsafe_allow_html=True)
        st.markdown(
            f"<div class='card' style='text-align:center'>"
            f"<div style='font-size:3rem;font-weight:800;color:{rc}'>{risk_n}/5</div>"
            f"<div style='color:{rc};font-weight:600;font-size:1.1rem;margin-top:4px'>{rl} Risk</div>"
            f"<div style='color:#8b949e;font-size:0.78rem;margin-top:6px'>"
            f"{f.get('name', active)} ({active})</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.metric("Beta (Market Sensitivity)",  f"{r.get('beta', 0):.2f}")
        st.metric("Debt / Equity",              f"{r.get('debt_equity', 0):.2f}")
        st.metric("Profit Margin",              f"{r.get('profit_margin', 0):.1f}%")
        st.metric("Dividend Yield",             f"{r.get('dividend_yield', 0):.2f}%")
        st.metric("52W High",                   f"${f.get('52w_high', 0):.2f}")
        st.metric("52W Low",                    f"${f.get('52w_low', 0):.2f}")

    with c2:
        st.markdown(
            '<div class="section-header">📋 AI Risk Report — IBM Granite</div>',
            unsafe_allow_html=True,
        )
        st.markdown(f'<div class="report-box">{risk_text}</div>', unsafe_allow_html=True)

    # Risk comparison if multiple tickers
    if len(tickers) > 1:
        st.markdown("---")
        st.plotly_chart(chart_risk_heatmap(tickers, ctx["overall_risk"]), use_container_width=True)

        # Risk metrics comparison table
        st.markdown('<div class="section-header">📊 Risk Metrics Comparison</div>', unsafe_allow_html=True)
        risk_rows = []
        for t in tickers:
            rr = ctx["ratios"][t]
            ff = ctx["fundamentals"][t]
            risk_rows.append({
                "Ticker":  t,
                "Risk":    f"{ctx['overall_risk'].get(t, 3)}/5",
                "Label":   risk_labels[ctx["overall_risk"].get(t, 3) - 1],
                "Beta":    rr["beta"],
                "D/E":     rr["debt_equity"],
                "Margin%": rr["profit_margin"],
                "ROE%":    rr["roe"],
                "Div%":    rr["dividend_yield"],
            })
        df_risk = pd.DataFrame(risk_rows).set_index("Ticker")
        st.dataframe(df_risk.style.format(precision=2), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Tab: Research Reports
# ═══════════════════════════════════════════════════════════════════════════════
def render_reports_tab(ctx: dict):
    tickers = ctx["tickers"]
    active  = st.selectbox(
        "Select Company",
        tickers,
        format_func=lambda t: f"{t} — {ctx['fundamentals'].get(t, {}).get('name', t)}",
        key="report_ticker_select",
    )
    if not active:
        return

    report = ctx["reports"].get(active, "No report generated.")
    f = ctx["fundamentals"].get(active, {})
    r = ctx["ratios"].get(active, {})

    # Header strip
    analyst = f.get("analyst_rating", "N/A")
    a_color = ANALYST_COLORS.get(analyst, "#8b949e")
    st.markdown(
        f"<div class='card'>"
        f"<div style='display:flex;align-items:center;gap:24px;flex-wrap:wrap'>"
        f"<div><span style='font-size:1.4rem;font-weight:700'>{active}</span> "
        f"<span style='color:#8b949e'> — {f.get('name', active)}</span></div>"
        f"<div style='display:flex;gap:20px;flex-wrap:wrap;font-size:0.84rem;color:#8b949e'>"
        f"<span>Sector: <b style='color:#e6edf3'>{f.get('sector', 'N/A')}</b></span>"
        f"<span>Price: <b style='color:#e6edf3'>${f.get('price', 0):.2f}</b></span>"
        f"<span>Market Cap: <b style='color:#e6edf3'>${f.get('market_cap_b', 0):.1f}B</b></span>"
        f"<span>Analyst: <b style='color:{a_color}'>{analyst}</b></span>"
        f"<span>Target: <b style='color:#e6edf3'>${f.get('price_target', 0):.2f}</b></span>"
        f"<span>Upside: <b style='color:#3fb950'>{r.get('upside_pct', 0):+.1f}%</b></span>"
        f"</div></div></div>",
        unsafe_allow_html=True,
    )

    # Key metrics row
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Quality Score", f"{r['quality_score']}/100")
    m2.metric("Value Score",   f"{r['value_score']}/100")
    m3.metric("Momentum",      f"{r.get('momentum_score', 0)}/100")
    m4.metric("Risk Rating",   f"{r['risk_rating']}/5")
    m5.metric("P/E Ratio",     f"{r['pe_ratio']:.1f}x")
    m6.metric("Div Yield",     f"{r['dividend_yield']:.2f}%")

    st.markdown(f'<div class="report-box">{report}</div>', unsafe_allow_html=True)

    # Download buttons
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            f"⬇️ Download {active} Report (.txt)",
            data=report,
            file_name=f"investment_report_{active}.txt",
            mime="text/plain",
        )
    with col2:
        import json
        report_json = json.dumps({
            "ticker": active,
            "company": f.get("name", active),
            "generated_at": datetime.now().isoformat(),
            "report": report,
            "metrics": r,
        }, indent=2)
        st.download_button(
            f"⬇️ Download {active} Report (.json)",
            data=report_json,
            file_name=f"investment_report_{active}.json",
            mime="application/json",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Tab: Portfolio Analysis
# ═══════════════════════════════════════════════════════════════════════════════
def render_portfolio_tab(ctx: dict):
    tickers  = ctx["tickers"]
    pm       = ctx.get("portfolio_metrics", {})
    weights  = ctx.get("portfolio_weights", {})
    summary  = ctx.get("portfolio_summary", "")
    prefs    = ctx.get("user_prefs", {})

    st.markdown('<div class="section-header">💼 Portfolio-Level Metrics</div>', unsafe_allow_html=True)

    # Aggregate metrics
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Weighted P/E",      f"{pm.get('portfolio_pe', 0):.1f}x")
    col2.metric("Portfolio Beta",    f"{pm.get('portfolio_beta', 0):.2f}")
    col3.metric("Avg Div Yield",     f"{pm.get('portfolio_div_yield', 0):.2f}%")
    col4.metric("Portfolio Quality", f"{pm.get('portfolio_quality', 0)}/100")
    col5.metric("Portfolio Value",   f"{pm.get('portfolio_value', 0)}/100")
    risk_val = pm.get("portfolio_risk", 3.0)
    risk_emj = ["🟢","🟢","🟡","🟠","🔴"][min(int(risk_val) - 1, 4)]
    col6.metric("Portfolio Risk",    f"{risk_emj} {risk_val:.1f}/5")

    # Charts row
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        st.plotly_chart(chart_portfolio_weights(tickers, weights), use_container_width=True)
    with c2:
        st.plotly_chart(chart_allocation_donut(prefs.get("risk_tolerance", "Moderate")), use_container_width=True)
    with c3:
        st.plotly_chart(
            chart_portfolio_performance(tickers, weights, prefs.get("days", 365)),
            use_container_width=True,
        )

    # Allocation table
    st.markdown('<div class="section-header">📊 Quality-Weighted Allocation</div>', unsafe_allow_html=True)
    alloc_rows = []
    for t in tickers:
        f = ctx["fundamentals"][t]
        r = ctx["ratios"][t]
        w = weights.get(t, 0)
        budget = prefs.get("amount", 50000)
        alloc_rows.append({
            "Ticker":       t,
            "Company":      f.get("name", t),
            "Sector":       f.get("sector", ""),
            "Weight (%)":   round(w * 100, 1),
            "$ Allocation": round(w * budget, 2),
            "Quality":      r["quality_score"],
            "Risk":         r["risk_rating"],
            "Analyst":      f.get("analyst_rating", "N/A"),
        })
    df_alloc = pd.DataFrame(alloc_rows).set_index("Ticker")
    st.dataframe(df_alloc.style.format(precision=2), use_container_width=True)

    # Portfolio AI summary
    st.markdown(
        '<div class="section-header">🤖 Portfolio Analysis — IBM Granite</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="report-box">{summary}</div>', unsafe_allow_html=True)

    # Download
    st.download_button(
        "⬇️ Download Portfolio Analysis (.txt)",
        data=summary,
        file_name="portfolio_analysis.txt",
        mime="text/plain",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Landing page
# ═══════════════════════════════════════════════════════════════════════════════
def render_landing():
    st.markdown("")
    features = [
        ("🤖", "#58a6ff", "6 Sequential AI Agents",
         "Data → Ratio → Sentiment → Risk → Report → Portfolio agents "
         "run in pipeline to deliver comprehensive investment research."),
        ("💡", "#3fb950", "IBM Granite LLM",
         "Sentiment analysis, multi-dimensional risk assessment, and full "
         "equity research reports powered by IBM Granite 3 8B on watsonx.ai."),
        ("📊", "#d29922", "Interactive Dashboard",
         "Candlestick charts with Bollinger Bands, radar scorecards, risk heatmaps, "
         "sentiment gauges, peer comparisons, and portfolio analytics."),
        ("🛡️", "#f85149", "Ethical AI Guardrails",
         "Input validation, output sanitization, blocked topic detection, "
         "and mandatory financial disclaimers on every AI response."),
        ("💼", "#a371f7", "Portfolio Analytics",
         "Quality-weighted allocation recommendations, portfolio-level metrics, "
         "simulated performance, and IBM Granite portfolio insights."),
        ("🔗", "#58a6ff", "Source Citations",
         "All AI-generated reports include timestamped data source citations, "
         "IBM Granite model details, and project attribution."),
    ]
    cols = st.columns(3)
    for i, (icon, color, title, desc) in enumerate(features):
        with cols[i % 3]:
            st.markdown(
                f"<div class='feature-card'>"
                f"<div style='font-size:2rem'>{icon}</div>"
                f"<div style='color:{color};font-weight:600;margin:10px 0 6px'>{title}</div>"
                f"<div style='color:#8b949e;font-size:0.85rem;line-height:1.5'>{desc}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("")
    st.info(
        "👈 **Select stocks and investment preferences in the sidebar**, then click "
        "**🚀 Run Investment Analysis** to start the 6-agent IBM Granite pipeline."
    )
    st.markdown(
        f'<div class="disclaimer">{FINANCIAL_DISCLAIMER}</div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Main application
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    render_header()

    # Market overview (always visible)
    if st.session_state.analysis_done:
        overview_data = st.session_state.context.get("market", {})
    else:
        overview_data = syn.generate_market_overview()
    render_market_overview(overview_data)

    st.markdown("---")
    render_agent_pipeline_status(done=st.session_state.analysis_done)
    st.markdown("---")

    # Sidebar returns prefs only when Run button clicked
    prefs = render_sidebar()

    if prefs:
        st.session_state.preferences = prefs
        status_ph  = st.empty()
        prog_bar   = st.progress(0)

        def _update_progress(msg: str, pct: int):
            status_ph.info(msg)
            prog_bar.progress(pct)

        with st.spinner("🤖 Running 6-agent IBM Granite pipeline…"):
            ctx = st.session_state.orchestrator.run(
                tickers    = prefs["tickers"],
                user_prefs = {
                    "horizon":        prefs["horizon"],
                    "risk_tolerance": prefs["risk_tolerance"],
                    "sectors":        prefs["sectors"],
                    "amount":         prefs["amount"],
                },
                days     = prefs["days"],
                progress = _update_progress,
            )
        st.session_state.context       = ctx
        st.session_state.analysis_done = True
        status_ph.success("✅ All 6 agents complete! Explore the dashboard below.")
        prog_bar.progress(100)
        time.sleep(0.8)
        status_ph.empty()
        prog_bar.empty()
        st.rerun()

    # Dashboard tabs
    if st.session_state.analysis_done:
        ctx  = st.session_state.context
        tabs = st.tabs([
            "📊 Overview",
            "📈 Charts",
            "📋 Fundamentals",
            "📰 News & Sentiment",
            "⚠️ Risk Assessment",
            "📝 Research Reports",
            "💼 Portfolio",
        ])
        with tabs[0]: render_overview_tab(ctx)
        with tabs[1]: render_charts_tab(ctx)
        with tabs[2]: render_fundamentals_tab(ctx)
        with tabs[3]: render_news_tab(ctx)
        with tabs[4]: render_risk_tab(ctx)
        with tabs[5]: render_reports_tab(ctx)
        with tabs[6]: render_portfolio_tab(ctx)

        st.markdown(
            f'<div class="disclaimer">{FINANCIAL_DISCLAIMER}</div>',
            unsafe_allow_html=True,
        )
    else:
        render_landing()

    st.markdown(
        f'<div class="ibm-footer">'
        f'Made with IBM Bob &nbsp;·&nbsp; IBM Granite 3 8B Instruct &nbsp;·&nbsp; '
        f'IBM watsonx.ai (au-syd) &nbsp;·&nbsp; '
        f'Investment Agentic AI v{APP_VERSION}'
        f'</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
