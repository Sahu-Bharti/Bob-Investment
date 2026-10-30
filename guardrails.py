"""
Ethical AI guardrails for the Investment Agentic AI.
Blocks harmful/illegal financial topics and sanitizes LLM outputs.
"""

from __future__ import annotations
import re
from config import BLOCKED_TOPICS, FINANCIAL_DISCLAIMER


# ── Input guard ───────────────────────────────────────────────────────────────
def _contains_blocked(text: str) -> bool:
    lower = text.lower()
    return any(topic in lower for topic in BLOCKED_TOPICS)


def check_input(user_text: str) -> str | None:
    """
    Returns an error message if the input should be blocked, else None.
    Applied only during free-form Q&A, not during structured data collection.
    """
    if not user_text or not user_text.strip():
        return "Please enter a question or message."

    if len(user_text.strip()) > 2000:
        return "Your message is too long. Please keep questions under 2000 characters."

    if _contains_blocked(user_text):
        return (
            "⚠️ I'm unable to assist with that request. "
            "This platform is designed for legal, ethical investment research only. "
            "Please consult a licensed financial advisor for legitimate investment guidance."
        )
    return None


# ── Output sanitizer ─────────────────────────────────────────────────────────
def sanitize_output(text: str) -> str:
    """
    Clean and validate LLM output before display:
    1. Remove accidentally generated harmful content.
    2. Strip leaked system-prompt fragments.
    3. Normalise whitespace and punctuation.
    """
    if not text or not text.strip():
        return "No response generated. Please try again."

    if _contains_blocked(text):
        return (
            "I was unable to generate a safe response for that query. "
            "Please rephrase your question or consult a licensed financial professional."
        )

    # Remove common system-prompt leak patterns
    text = re.sub(r"(?i)^(system:|assistant:|user:|<\|system\|>|<\|user\|>|<\|assistant\|>).*$",
                  "", text, flags=re.MULTILINE)

    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip leading/trailing whitespace
    return text.strip()


# ── System prompt builder ────────────────────────────────────────────────────
_BASE_PROMPT = (
    "You are an expert AI investment research assistant powered by IBM Granite 3 8B Instruct "
    "running on IBM watsonx.ai (au-syd region).\n"
    "Your analysis MUST be:\n"
    "- Objective, evidence-based, and data-driven.\n"
    "- Clearly structured with numbered or bolded section headers.\n"
    "- Honest about the limitations of AI-generated financial analysis.\n"
    "- Free of guaranteed return claims, speculative hype, or unfounded projections.\n"
    "- Compliant with ethical investment research standards (CFA Institute guidelines).\n"
    "- Always concluding with a recommendation to consult a licensed financial advisor.\n"
    f"MANDATORY DISCLAIMER: {FINANCIAL_DISCLAIMER}\n\n"
)


def build_system_prompt(role: str = "analyst") -> str:
    """
    Build the system prompt for different investment agent roles.
    role: 'analyst' | 'news' | 'risk' | 'advisor' | 'portfolio'
    """
    role_additions = {
        "analyst": (
            "You are a fundamental equity analyst with expertise in DCF valuation, "
            "ratio analysis, and competitive positioning. Analyse financial ratios, "
            "valuation multiples, competitive moat, and sector dynamics with precision."
        ),
        "news": (
            "You are a financial news analyst specialising in quantitative and qualitative "
            "sentiment analysis. Assess market-moving news, identify key themes, "
            "quantify sentiment impact, and flag material risks or catalysts."
        ),
        "risk": (
            "You are a risk management specialist with expertise in enterprise risk frameworks. "
            "Identify, quantify, and explain financial, operational, regulatory, "
            "geopolitical, and macroeconomic risks. "
            "Assign clear risk ratings (Low/Medium/High) with quantitative justification where possible."
        ),
        "advisor": (
            "You are a conversational investment research assistant helping retail investors "
            "understand complex financial concepts. "
            "Answer questions clearly and concisely, always citing the data provided. "
            "Use plain English where possible. "
            "Never make specific buy/sell recommendations without extensive, explicit caveats."
        ),
        "portfolio": (
            "You are a portfolio construction specialist with expertise in modern portfolio theory, "
            "diversification, factor investing, and asset allocation. "
            "Analyse portfolio-level metrics, correlation, and risk-adjusted returns."
        ),
    }
    addition = role_additions.get(role, "")
    return _BASE_PROMPT + addition


# ── Citation builder ─────────────────────────────────────────────────────────
def build_citations(data_sources: list[str]) -> str:
    """Build a formatted citations block from a list of source identifiers."""
    from config import DATA_SOURCES
    lines = ["**Data Sources & Citations:**"]
    for src in set(data_sources):
        if src in DATA_SOURCES:
            lines.append(f"- {DATA_SOURCES[src]}")
    lines.append("- IBM Granite 3 8B Instruct · IBM watsonx.ai · Project: dc89f050")
    lines.append("- Analysis timestamp: " + __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M UTC"))
    return "\n".join(lines)
