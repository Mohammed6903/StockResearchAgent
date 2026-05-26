"""Static, authoritative glossary of the metrics the tool uses.

Kept as plain data (no LLM) so definitions and formulas are always correct. The explainer
uses these definitions and pairs them with a stock's real values; the `explain` CLI command
prints them directly. Beginner-oriented.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Term:
    key: str               # matches a Fundamentals/QuantMetrics field where possible
    name: str
    aliases: list[str]
    definition: str
    formula: str
    derived_from: str
    how_to_read: str
    caveat: str
    india_note: str = ""
    higher_is_better: bool | None = None  # used for templated interpretation; None = nuanced


_TERMS: list[Term] = [
    Term(
        "pe_ratio", "P/E ratio (trailing)",
        ["pe", "p/e", "price to earnings", "price earnings"],
        "How much you pay for ₹1 of the company's past-year profit.",
        "share price ÷ earnings per share (EPS)",
        "Income statement: net income → EPS; price from the market.",
        "Lower can mean cheaper, higher can mean expensive OR high growth expected. Compare "
        "to the company's own history and its sector, not in isolation.",
        "A high P/E on falling earnings is a red flag; a low P/E on a declining business is a "
        "'value trap', not a bargain.",
        higher_is_better=False,
    ),
    Term(
        "forward_pe", "Forward P/E",
        ["forward pe", "forward p/e"],
        "Like P/E but using forecast next-year earnings instead of past earnings.",
        "share price ÷ expected forward EPS",
        "Analyst earnings estimates + current price.",
        "If forward P/E is much lower than trailing P/E, the market expects profit to grow.",
        "Forecasts can be wrong; treat as expectation, not fact.",
        higher_is_better=False,
    ),
    Term(
        "peg_ratio", "PEG ratio",
        ["peg"],
        "P/E adjusted for growth — tries to say whether a high P/E is justified by growth.",
        "P/E ÷ expected earnings growth rate (%)",
        "P/E and a growth estimate.",
        "Around 1 is often considered 'fair'; well above ~2 suggests you're paying a lot even "
        "after accounting for growth.",
        "Very sensitive to the growth estimate used.",
        higher_is_better=False,
    ),
    Term(
        "pb_ratio", "P/B ratio (price-to-book)",
        ["pb", "p/b", "price to book"],
        "Price versus the company's net asset value on its books.",
        "share price ÷ book value per share",
        "Balance sheet: total equity (assets − liabilities).",
        "Below 1 means the market values it below book assets; useful for banks/heavy-asset "
        "firms, less so for asset-light tech.",
        "Book value can understate or overstate real worth (intangibles, old asset costs).",
        higher_is_better=False,
    ),
    Term(
        "ps_ratio", "P/S ratio (price-to-sales)",
        ["ps", "p/s", "price to sales"],
        "Price versus annual revenue — useful when profits are thin or negative.",
        "market cap ÷ total revenue",
        "Income statement: total revenue.",
        "Lower is cheaper per ₹ of sales; only compare within the same industry.",
        "Ignores profitability entirely — high sales with no profit is still risky.",
        higher_is_better=False,
    ),
    Term(
        "debt_to_equity", "Debt-to-equity (D/E)",
        ["de", "d/e", "debt to equity", "leverage"],
        "How much the company borrows relative to owners' capital.",
        "total debt ÷ shareholders' equity",
        "Balance sheet: total debt and total equity.",
        "Higher = more leverage = more risk if earnings fall or rates rise. Reasonable levels "
        "vary by industry (capital-heavy sectors run higher).",
        "yfinance often reports this as a percentage (e.g. 89 = 0.89×). Capital-intensive "
        "Indian firms (steel, infra, telecom) naturally carry high D/E.",
        higher_is_better=False,
    ),
    Term(
        "current_ratio", "Current ratio",
        ["current ratio", "liquidity"],
        "Can the company pay its bills due within a year?",
        "current assets ÷ current liabilities",
        "Balance sheet: current assets and current liabilities.",
        "Above 1 means short-term assets cover short-term dues; below 1 is a liquidity "
        "warning (though some efficient businesses run below 1).",
        "A very high ratio can mean idle cash not being put to work.",
        higher_is_better=True,
    ),
    Term(
        "quick_ratio", "Quick ratio (acid-test)",
        ["quick ratio", "acid test"],
        "Like current ratio but excludes inventory (which may be hard to sell fast).",
        "(current assets − inventory) ÷ current liabilities",
        "Balance sheet: current assets, inventory, current liabilities.",
        "A stricter liquidity test; below ~0.5 signals reliance on selling inventory to pay "
        "near-term bills.",
        "Industries with fast-moving inventory are less affected by a low quick ratio.",
        higher_is_better=True,
    ),
    Term(
        "roe", "Return on equity (ROE)",
        ["roe", "return on equity"],
        "How much profit the company generates per ₹1 of owners' money.",
        "net income ÷ shareholders' equity",
        "Income statement (net income) and balance sheet (equity).",
        "Higher is generally better; consistently high ROE signals a quality business.",
        "Can be inflated by high debt — always read ROE alongside D/E.",
        higher_is_better=True,
    ),
    Term(
        "roa", "Return on assets (ROA)",
        ["roa", "return on assets"],
        "How efficiently the company turns all its assets into profit.",
        "net income ÷ total assets",
        "Income statement (net income) and balance sheet (total assets).",
        "Higher = more efficient asset use. Unlike ROE it isn't flattered by leverage.",
        "Capital-heavy industries naturally show lower ROA.",
        higher_is_better=True,
    ),
    Term(
        "gross_margin", "Gross margin",
        ["gross margin"],
        "Share of revenue left after the direct cost of making the product.",
        "(revenue − cost of goods sold) ÷ revenue",
        "Income statement: revenue and cost of revenue.",
        "Higher means stronger pricing power / lower input cost burden.",
        "Comparable only within an industry — software is ~80%, steel is much lower.",
        higher_is_better=True,
    ),
    Term(
        "operating_margin", "Operating margin",
        ["operating margin"],
        "Profit from core operations as a share of revenue (before interest and tax).",
        "operating income ÷ revenue",
        "Income statement: operating income and revenue.",
        "Higher = more profitable core business; rising margin is a good operational sign.",
        "Excludes financing costs, which matter a lot for high-debt firms.",
        higher_is_better=True,
    ),
    Term(
        "profit_margin", "Net profit margin",
        ["profit margin", "net margin"],
        "Share of revenue that ends up as bottom-line profit.",
        "net income ÷ revenue",
        "Income statement: net income and revenue.",
        "Higher is better; track the trend over time.",
        "One-off gains/losses can distort a single period.",
        higher_is_better=True,
    ),
    Term(
        "beta", "Beta",
        ["beta"],
        "How much the stock tends to move relative to the market index.",
        "slope of the stock's returns regressed on the benchmark's returns",
        "Daily price history of the stock and the benchmark (NIFTY 50 / SPY).",
        "1 = moves with the market; >1 = more volatile (amplifies up AND down); <1 = steadier.",
        "Backward-looking and can change over time; says nothing about value.",
        higher_is_better=None,
    ),
    Term(
        "alpha", "Alpha",
        ["alpha"],
        "Return earned beyond what its market risk (beta) alone would predict.",
        "annualised excess return vs the beta-implied return (CAPM)",
        "Stock and benchmark return history + risk-free rate.",
        "Positive = outperformed its risk-adjusted expectation; negative = underperformed.",
        "Past alpha does not predict future alpha.",
        higher_is_better=True,
    ),
    Term(
        "sharpe", "Sharpe ratio",
        ["sharpe"],
        "Return earned per unit of risk (volatility) taken.",
        "(annual return − risk-free rate) ÷ annual volatility",
        "Daily returns + the configured risk-free rate.",
        "Higher = better risk-adjusted return. >1 is good; negative means you weren't "
        "rewarded for the risk over the period.",
        "Period-dependent; a single bad stretch can dominate.",
        higher_is_better=True,
    ),
    Term(
        "annual_volatility", "Volatility",
        ["volatility", "vol", "std"],
        "How much the price swings around — a measure of risk.",
        "standard deviation of daily returns, annualised",
        "Daily price history.",
        "Higher = bigger swings = more risk (and bigger potential gains/losses).",
        "Volatility is not the same as a permanent loss of capital.",
        higher_is_better=False,
    ),
    Term(
        "max_drawdown", "Maximum drawdown",
        ["max drawdown", "drawdown", "mdd"],
        "The worst peak-to-trough fall over the period.",
        "minimum of (price ÷ running peak − 1)",
        "Daily price history.",
        "Closer to 0 is better; a −50% drawdown means it once fell by half from a high.",
        "Tells you the historical worst case, not the future one.",
        higher_is_better=False,
    ),
    Term(
        "dividend_yield", "Dividend yield",
        ["dividend yield", "dividend", "yield"],
        "Annual dividend income as a percentage of the share price.",
        "annual dividend per share ÷ share price",
        "Company dividend declarations + price.",
        "Higher gives more income; very high yields can signal a falling price or an "
        "unsustainable payout.",
        "A dividend can be cut; yield alone isn't safety.",
        higher_is_better=None,
    ),
    Term(
        "free_cashflow", "Free cash flow (FCF)",
        ["fcf", "free cash flow", "free cashflow"],
        "Cash left after running the business and funding capital spending.",
        "operating cash flow − capital expenditure",
        "Cash-flow statement: operating cash flow and capex.",
        "Positive and growing FCF means the business self-funds and can pay debt/dividends.",
        "Lumpy capex (big projects) can make a single year look weak or strong.",
        higher_is_better=True,
    ),
    Term(
        "market_cap", "Market capitalisation",
        ["market cap", "mcap", "marketcap"],
        "Total market value of all the company's shares — its 'size'.",
        "share price × shares outstanding",
        "Price and share count.",
        "Bigger caps tend to be more stable; small caps swing more.",
        "Size is not quality or value on its own.",
        higher_is_better=None,
    ),
]

_BY_KEY: dict[str, Term] = {t.key: t for t in _TERMS}


def all_terms() -> list[Term]:
    return list(_TERMS)


def get(key: str) -> Term | None:
    return _BY_KEY.get(key)


def lookup(query: str) -> Term | None:
    """Match by key, name, or alias (case-insensitive substring)."""
    q = query.strip().lower()
    if not q:
        return None
    for t in _TERMS:
        if q == t.key or q == t.name.lower() or q in {a.lower() for a in t.aliases}:
            return t
    for t in _TERMS:
        hay = " ".join([t.key, t.name, *t.aliases]).lower()
        if q in hay:
            return t
    return None
