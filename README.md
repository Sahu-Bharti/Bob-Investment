# 📈 Investment Agentic AI

### Powered by IBM Granite LLM · IBM watsonx.ai · Streamlit

> A complete, responsive investment research platform that uses a **6-agent sequential AI architecture** to deliver AI-powered stock analysis, risk assessment, portfolio analytics, and professional investment research reports.

---

## 🚀 Live Demo

```
http://localhost:8501          ← Local
http://192.168.1.52:8501       ← Network
```

To generate a public URL, run in a second terminal:
```bash
ssh -o StrictHostKeyChecking=no -R 80:localhost:8501 serveo.net
```

---

## 🏗️ Architecture

```
User Preferences (Sidebar)
        │
        ▼
┌─────────────┐    ┌─────────────┐    ┌──────────────────┐    ┌────────────┐    ┌─────────────┐    ┌──────────────┐
│  DataAgent  │───▶│ RatioAgent  │───▶│ SentimentAgent   │───▶│ RiskAgent  │───▶│ ReportAgent │───▶│PortfolioAgent│
│             │    │             │    │ (IBM Granite)    │    │(IBM Granite│    │(IBM Granite)│    │(IBM Granite) │
│ • Prices    │    │ • P/E, P/B  │    │ • News summary   │    │• Risk cats │    │• Full report│    │• Allocation  │
│ • Fundament.│    │ • P/S, ROE  │    │ • Sentiment score│    │• Risk rating    │• Bull/Bear  │    │• Performance │
│ • News      │    │ • Quality   │    │                  │    │            │    │• Citations  │    │• Rebalancing │
│ • Market    │    │ • Value     │    │                  │    │            │    │             │    │              │
│             │    │ • Momentum  │    │                  │    │            │    │             │    │              │
└─────────────┘    └─────────────┘    └──────────────────┘    └────────────┘    └─────────────┘    └──────────────┘
                                                                                                          │
                                                                                                          ▼
                                                                                               Interactive Dashboard
                                                                                           (Streamlit · Plotly · 7 Tabs)
```

---

## ✨ Features

| Feature | Description |
|---|---|
| **6 Sequential Agents** | Data → Ratio → Sentiment → Risk → Report → Portfolio pipeline |
| **IBM Granite LLM** | Sentiment, risk, reports and portfolio analysis via watsonx.ai |
| **7 Dashboard Tabs** | Overview · Charts · Fundamentals · News · Risk · Reports · Portfolio |
| **Candlestick Charts** | OHLCV with MA20, MA50, Bollinger Bands, and volume bars |
| **Financial Ratios** | P/E, P/B, P/S, ROE, D/E, EPS, Graham Number, Momentum Score |
| **Quality vs Risk Bubble** | Interactive scatter chart (bubble = market cap) |
| **Company Comparison** | Side-by-side peer analysis across 8 metrics |
| **News Sentiment** | AI-powered headline analysis with sentiment gauges |
| **Risk Assessment** | 5-category risk framework with IBM Granite narrative |
| **Research Reports** | Full equity research: Executive Summary, Scenarios, Scorecard |
| **Portfolio Analytics** | Quality-weighted allocation, simulated performance, rebalancing |
| **Source Citations** | Timestamped data source citations in every AI report |
| **Conversational AI** | Sidebar chat assistant powered by IBM Granite |
| **Ethical Guardrails** | Input blocking, output sanitization, mandatory disclaimers |
| **Synthetic + Real Data** | Works fully offline; connects to Alpha Vantage / Finnhub / NewsAPI |
| **Export** | Download reports as .txt and .json; fundamentals as CSV |

---

## ⚡ Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/Sahu-Bharti/Bob-Investment.git
cd Bob-Investment
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the application
```bash
python -m streamlit run app.py
```

The app opens at **http://localhost:8501**

### 4. Use the app
1. Select **1–6 stocks** in the left sidebar
2. Set **investment horizon**, **risk tolerance**, and **sectors**
3. Enter your **portfolio size (USD)**
4. Click **🚀 Run Investment Analysis**
5. Explore the **7 dashboard tabs**

---

## 🤖 IBM Granite Configuration

The IBM watsonx.ai credentials are pre-configured in [`config.py`](config.py):

| Setting | Value |
|---|---|
| **Project ID** | `dc89f050-b0c0-46ea-8ed1-5b02a864a73e` |
| **Endpoint URL** | `https://au-syd.ml.cloud.ibm.com` |
| **Model** | `ibm/granite-3-8b-instruct` |
| **Region** | IBM Cloud · au-syd |

### Generation Parameters
| Parameter | Value |
|---|---|
| max_new_tokens | 1500 |
| temperature | 0.35 |
| top_k | 40 |
| top_p | 0.90 |
| repetition_penalty | 1.15 |
| max_retries | 3 |

---

## 🔌 Real Data APIs (Optional)

Add API keys to [`config.py`](config.py) to switch from synthetic to live data:

| Provider | Constant | Data | Sign Up |
|---|---|---|---|
| Alpha Vantage | `ALPHA_VANTAGE_API_KEY` | Stock prices (OHLCV) | [alphavantage.co](https://www.alphavantage.co/support/#api-key) |
| Finnhub | `FINNHUB_API_KEY` | Fundamentals, metrics | [finnhub.io](https://finnhub.io/register) |
| NewsAPI | `NEWSAPI_KEY` | News headlines | [newsapi.org](https://newsapi.org/register) |

> If any key is left blank, that data source falls back to **synthetic data** automatically.

---

## 📁 File Structure

```
investment_ai/
├── app.py              # Streamlit application — main entry point
├── agents.py           # 6-agent sequential workflow orchestrator
├── financial_data.py   # Data fetching layer (real + synthetic fallback)
├── synthetic_data.py   # Synthetic OHLCV, fundamentals, news, portfolio sim
├── llm_client.py       # IBM Granite LLM client with retry logic
├── guardrails.py       # Ethical AI guardrails, input/output validation
├── config.py           # Configuration, credentials, constants
├── requirements.txt    # Python dependencies
├── .streamlit/
│   └── config.toml     # Dark theme & server settings
└── README.md
```

---

## 🤖 Agent Details

### Agent 1 — DataAgent
Fetches all raw data: price history (OHLCV), company fundamentals, news headlines, and market overview indices (S&P 500, NASDAQ, Dow, VIX).

### Agent 2 — RatioAgent
Computes financial ratios and composite scores:
- **Quality Score** (0–100): ROE + margin + leverage + EPS
- **Value Score** (0–100): P/E + P/B relative benchmarks
- **Momentum Score** (0–100): dividend yield + profitability + low leverage
- **Graham Number**: intrinsic value proxy
- **Risk Rating** (1–5): beta + debt/equity + margin

### Agent 3 — SentimentAgent *(IBM Granite)*
Analyses recent news headlines with IBM Granite to produce a 3-paragraph sentiment summary covering tone, positive catalysts, and negative risks.

### Agent 4 — RiskAgent *(IBM Granite)*
Assesses 5 risk dimensions with IBM Granite:
1. Market Risk (beta, volatility, macro)
2. Financial Risk (leverage, liquidity)
3. Business Risk (competitive moat, margins)
4. Regulatory & ESG Risk
5. Valuation Risk (multiple compression)

### Agent 5 — ReportAgent *(IBM Granite)*
Generates a professional 8-section equity research report:
1. Executive Summary
2. Business Overview
3. Financial Analysis
4. Sentiment & News Analysis
5. Risk Factors
6. Investment Scenarios (Bull / Base / Bear)
7. Key Metrics Scorecard
8. Investment Recommendation + Citations

### Agent 6 — PortfolioAgent *(IBM Granite)*
Portfolio-level analysis:
- Quality-weighted allocation recommendation
- Aggregate P/E, Beta, Dividend Yield, Quality, Risk
- Simulated portfolio performance chart
- Diversification, rebalancing, and risk profile assessment

---

## 🛡️ Ethical AI & Guardrails

- All LLM inputs are validated against a blocked-topics list (insider trading, market manipulation, fraud, etc.)
- All LLM outputs pass through sanitization (harmful content removal, system-prompt leak detection)
- Every page and AI response includes a **mandatory financial disclaimer**
- Reports include **timestamped source citations**
- The application explicitly states it does **not** constitute financial advice
- Synthetic data is clearly labelled; real data sources are cited

---

## 📊 Dashboard Tabs

| Tab | Contents |
|---|---|
| **📊 Overview** | Holdings snapshot, risk heatmap, quality ranking, allocation donut, quality-vs-risk bubble |
| **📈 Charts** | Candlestick + Bollinger Bands + MA20/50, volume bars, radar scorecard, sentiment gauge, peer comparison bars |
| **📋 Fundamentals** | Full ratio table, Graham Number analysis, CSV/JSON download |
| **📰 News & Sentiment** | IBM Granite sentiment summary, sentiment gauge, headline breakdown, recent articles |
| **⚠️ Risk Assessment** | IBM Granite risk report, risk scorecard, risk metrics comparison table |
| **📝 Research Reports** | Full AI equity research report, key metrics row, TXT/JSON download |
| **💼 Portfolio** | Quality-weighted allocation pie, simulated performance, allocation table, IBM Granite portfolio analysis |

---

## 🔧 Development

### Running locally
```bash
cd investment_ai
python -m streamlit run app.py
```

### Sharing publicly (no install needed)
```bash
# In a second terminal while the app is running:
ssh -o StrictHostKeyChecking=no -R 80:localhost:8501 serveo.net
```

### Environment
- Python 3.10+
- Streamlit 1.35+
- Plotly 5.22+
- IBM watsonx-ai 1.1+

---

## 📋 Requirements

```
streamlit>=1.35.0
plotly>=5.22.0
pandas>=2.2.0
requests>=2.31.0
ibm-watsonx-ai>=1.1.0
```

---

## ⚠️ Financial Disclaimer

> This application is for **informational and educational purposes only**.
> It does **not** constitute financial advice, a solicitation, or a recommendation to buy or sell any security.
> Past performance is not indicative of future results.
> Always consult a **licensed financial advisor** before making investment decisions.
> AI-generated analysis may contain errors or omissions and should not be the sole basis for any investment decision.

---

## 🏷️ Credits

| Component | Provider |
|---|---|
| **LLM** | IBM Granite 3 8B Instruct |
| **AI Platform** | IBM watsonx.ai (au-syd) |
| **Framework** | Streamlit |
| **Charts** | Plotly |
| **Built with** | IBM Bob |

---

*Made with ❤️ using IBM Bob · IBM Granite · IBM watsonx.ai*
