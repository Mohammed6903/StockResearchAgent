# StockResearchAgent

An agentic, daily stock-research CLI built on **Google Vertex AI Gemini**. It scans a
universe of stocks (your watchlist ∪ an index), pulls forensic fundamentals, computes
risk/return metrics, reads the political–macro news landscape, reasons over all of it per
company, self-critiques, and emits a **ranked analytical report** with pros/cons and a
bullish/neutral/bearish lean. Every run is persisted to SQLite so you build a track record.

> **Stance:** this is *analytical research, not investment advice.* It surfaces reasoning
> and evidence; the judgment is yours.

---

## Guiding principle: the LLM interprets, it never calculates

This is the design decision everything else flows from. All quantitative metrics
(beta, alpha, Sharpe, volatility, drawdown, returns) are computed **deterministically in
Python** from real price series. Fundamentals (P/E, FCF, margins, debt ratios) are fetched
from a data source and passed **verbatim** into the prompt. Gemini is only ever asked to
*interpret* verified numbers and grounded news — never to produce a number it could
hallucinate. Missing data is flagged as a warning, never invented.

This gives three tiers of provenance, strongest first:

1. **Quant metrics** — computed by our own code (`quant/metrics.py`); independently
   re-computable and unit-tested against known inputs.
2. **Fundamentals** — live from yfinance (Yahoo) at fetch time, fed verbatim to the model.
   Fidelity source→model is guaranteed; the source's own correctness is not — so we
   optionally **cross-check against a second free source** (see below).
3. **News/macro** — retrieved via Gemini Google Search grounding with real source URLs;
   a summary of live results as of search time (the softest link). Citations are
   **surfaced and persisted** so every claim is auditable (see below).

### Grounding citations (tier-3 auditability)

News and macro signals come from Gemini Google Search grounding, which returns real source
URLs. To keep these auditable rather than taking the model's word:

- The macro pipeline grounds a search, then a structured pass attributes each `MacroSignal`
  to sources **by id** (indexing the numbered source list). We resolve those ids back to the
  real URLs deterministically, so the model never writes a URL itself. Signals it fails to
  attribute are flagged `Sources: unattributed`.
- Per-ticker news sources are carried onto each `TickerAnalysis`.
- Citations render in the terminal, in the saved Markdown report (as clickable links), and
  are persisted to SQLite (source URLs on the `macro_signals` table; full set in the report
  JSON).

Source URLs are Vertex grounding redirect links (they bounce to the publisher); the link
*title* carries the publisher domain (e.g. `federalreserve.gov`, `reuters.com`).

**Recency guards.** Grounded search sometimes surfaces old articles and labels them
"recent." To prevent stale narratives, every news/macro query is anchored to the current
date and asked to ignore anything older than ~1 month, and the analyst is told the
fundamentals are live, to disregard prior-year events, and to prefer the live numbers when
recent news contradicts them.

**Cross-source agreement.** A macro signal is only marked *corroborated* when it is backed
by ≥ 2 **distinct publisher domains** (deduped on the title-domain, since two links from the
same outlet are not independent confirmation). Uncorroborated signals keep their evidence
but have their confidence **capped at 50%** and are flagged `⚠ single-source` in the report
and `(single-source)` in the CLI — so a single-outlet claim can never present as
high-confidence.

### Fundamentals cross-check (tier-2 hardening)

To avoid trusting Yahoo alone, fundamentals can be cross-checked against an independent
free source (**Alpha Vantage** `OVERVIEW`). Comparable ratios (P/E, PEG, P/B, P/S, margins,
ROA, ROE, market cap, dividend yield) are compared field-by-field; any disagreement beyond
a relative tolerance (default 10%) is surfaced as a **data warning** on that ticker, e.g.:

```
Data warnings: P/E mismatch: Yahoo 37.65 vs 2nd source 25 (34% diff)
```

The comparison core (`data/verify.py::compare_fundamentals`) is a pure, unit-tested function
independent of any data source. Set a free key to enable it:

```bash
export ALPHAVANTAGE_API_KEY=your_key   # https://www.alphavantage.co/support/#api-key
```

Without a key, verification is a no-op that records an explicit
`"fundamentals not cross-checked (no 2nd source / API key)"` note, so the report is honest
about whether verification actually happened. Because Alpha Vantage's free tier is rate
limited (~25 req/day), verification is **on by default for `analyze`** (single ticker) and
**opt-in via `--verify` for `scan`** (to protect the daily quota).

---

## Architecture

Follows the recommended shape for production research agents:
**Orchestrator → parallel Workers → Reflection → tool-grounded Synthesis.**

```
CLI (Typer)
  │
  └─ Orchestrator (daily pipeline, thread-pool fan-out)
       1. Macro/Political Analyst        (runs once per day)
            ├─ Gemini Google Search grounding → grounded headlines + source URLs
            └─ structured pass → typed MacroSignals (sector/region/impact/confidence)
       2. Universe Resolver               (pure Python) → watchlist ∪ index, capped
       3. Per-ticker Workers (parallel), each ticker:
            ├─ get_fundamentals(ticker)   (yfinance .info)         → forensic data
            ├─ get_closes(ticker)         (yfinance history)       → quant computed locally
            ├─ get_company_news(ticker)   (Gemini grounding)       → grounded headlines
            ├─ Analyst → structured verdict (lean/score/pros/cons/reasoning)
            └─ Reflection → skeptical critic checks claims vs facts, revises once
       4. Synthesizer/Reporter → ranked DailyReport → Markdown + SQLite
```

Per-ticker failures are isolated: one bad ticker is logged and skipped, never aborting the
run.

### Module layout

| Path | Responsibility |
|------|----------------|
| `src/stockresearch/config.py` | Loads `settings.yaml` / `watchlist.yaml` into typed Pydantic models. |
| `src/stockresearch/models.py` | Shared schemas: `Fundamentals`, `QuantMetrics`, `TickerSnapshot`, `MacroSignal`, `TickerAnalysis`, `DailyReport`. |
| `src/stockresearch/gemini.py` | Vertex env setup + cached `google-genai` client. |
| `src/stockresearch/data/` | Data adapters: `yfinance_adapter`, `news_adapter` (grounding), `index_membership` (S&P 500), `verify` (2nd-source fundamentals cross-check), `cache` (day-level disk cache). |
| `src/stockresearch/quant/metrics.py` | Deterministic beta/alpha/Sharpe/volatility/drawdown/returns. No LLM, no network. |
| `src/stockresearch/agents/` | `universe`, `macro_agent`, `ticker_analyst` (+reflection), `explainer` (beginner teaching), `orchestrator`, `reason` (structured-output helper), `schemas`. |
| `src/stockresearch/glossary.py` | Static, authoritative metric definitions/formulas (no LLM). |
| `src/stockresearch/evaluate.py` | Outcome scoring, calibration, training-data export. Deterministic. |
| `src/stockresearch/portfolio.py` | Holdings review: value/P&L/weights + diversification checks. |
| `src/stockresearch/store/db.py` | SQLite persistence: runs, analyses, macro signals, outcomes + queries. |
| `src/stockresearch/report.py` | Markdown + `rich` terminal rendering. |
| `src/stockresearch/cli.py` | Typer CLI: scan/analyze/macro/explain/score/calibration/export/watch/config/portfolio. |
| `tests/` | Quant, store, agents, verify, citations, india, evaluate, portfolio — all no-network. |

---

## Key design decisions

- **CLI, not a web app.** Fastest path to daily value; the report is also written to
  `reports/<date>.md` for sharing.
- **Universe = watchlist ∪ index.** Your core names are always covered; the index gives
  breadth. Capped by `universe.max_tickers` for cost safety.
- **Free data sources.** yfinance for fundamentals/prices, Gemini Google Search grounding
  for news — no paid feeds required.
- **Structured outputs over ADK runtime.** Google ADK is installed, but the LLM reasoning
  runs through `google-genai` structured outputs (`response_schema`) — the layer ADK sits
  on. Because all data is gathered deterministically *before* reasoning, the work is a
  batch transform: structured generation is more reliable and trivially unit-testable
  (mockable, no async Runner/session ceremony) while keeping the exact
  orchestrator→workers→reflection→synthesis pattern. Swappable to the ADK runtime if a
  conversational/agentic UI is later needed.
- **Reflection pass.** A second skeptical model call checks the analyst's claims against the
  facts and revises once — risk reduction, not extra intelligence.
- **Deterministic facts win.** Identity (ticker/name/sector) and metrics on the final
  `TickerAnalysis` are always taken from the snapshot, never from the model's output.
- **Persistence by default.** Each run is stored whole (JSON) plus flat rows for trend and
  track-record queries.
- **Day-level caching.** Fetches are cached (TTL configurable) so re-runs and multi-command
  sessions don't re-hit data sources.

---

## Requirements

- Python 3.11+
- Google Cloud project with **Vertex AI enabled** and Application Default Credentials
  (ADC) configured (`~/.config/gcloud/application_default_credentials.json`).
- [`uv`](https://github.com/astral-sh/uv) (used here) or pip.

---

## Setup

```bash
cd StockResearchAgent

# create venv + install (editable, with dev extras)
uv venv --python 3.11
uv pip install -e ".[dev]"
```

Configure `config/settings.yaml`:

```yaml
vertex:
  project: YOUR_GCP_PROJECT      # defaults to the ADC quota project
  location: us-central1
  model: gemini-2.5-pro          # reasoning/synthesis
  fast_model: gemini-2.5-flash   # macro scan / news triage
universe:
  index: SP500                   # SP500 | NIFTY50 | NONE
  benchmark: SPY                 # used for beta/alpha (e.g. ^NSEI, ^BSESN for India)
  max_tickers: 50                # cost safety cap
  region: global                 # macro/news focus: global | India | US
quant:
  lookback_days: 365
  risk_free_rate: 0.045
runtime:
  concurrency: 4
  cache_ttl_hours: 12
  reports_dir: reports
  db_path: .cache/stockresearch.db
```

Set your watchlist in `config/watchlist.yaml`:

```yaml
tickers: [AAPL, MSFT, NVDA, GOOGL, AMZN, META, TSLA]
```

---

## Usage

The installed entrypoint is `stockresearch` (or `python -m stockresearch.cli`).
Add `-v` / `--verbose` before a subcommand for INFO logging.

### `scan` — full daily run

Analyzes the whole universe, ranks results, writes `reports/<date>.md`, and persists the run.

```bash
stockresearch scan
stockresearch scan --limit 10        # only first 10 tickers
stockresearch scan --no-news         # skip per-ticker news (faster/cheaper)
stockresearch scan --no-reflect      # skip the critique pass
stockresearch scan --no-macro        # skip the political/macro scan
stockresearch scan --verify          # cross-check fundamentals (uses AV quota)
```

### `analyze TICKER` — single-stock deep dive

```bash
stockresearch analyze NVDA
stockresearch analyze AAPL --no-reflect
stockresearch analyze AAPL --no-verify   # skip the fundamentals cross-check
stockresearch analyze TATASTEEL.NS --explain   # add a beginner walkthrough
```

Prints lean/score/confidence, the quant metrics, reasoning, pros/cons, and any data
warnings (including fundamentals mismatches). Verification runs by default here.

With `--explain`, it adds a beginner "How to read this" section: each ratio defined in plain
language with the stock's real value and how it's derived from the actual statement line
items (e.g. `current ratio = current assets ₹X ÷ current liabilities ₹Y`), a plain-English
walkthrough of how the numbers + recent news produced the verdict, and the news→stock
linkage. One extra LLM call; on-demand only.

### `macro` — political/macro scan only

```bash
stockresearch macro
```

Grounded headlines distilled into typed market-moving signals (impact, sectors, confidence).

### `report [DATE]` — re-render a stored run

```bash
stockresearch report              # latest
stockresearch report 2026-05-26   # specific date
```

### `runs` — list stored runs

```bash
stockresearch runs
```

### `watch` — manage tracked tickers

Edit the watchlist without touching YAML:

```bash
stockresearch watch list
stockresearch watch add AMD AVGO     # add one or more (deduped, upper-cased)
stockresearch watch remove TSLA
stockresearch watch clear            # prompts; -y to skip
```

### `config` — view and update settings

All settings are viewable; only daily-use keys are editable via `set` (changes are
validated against the schema and your YAML comments are preserved). Editable keys:
`universe.index`, `universe.benchmark`, `universe.max_tickers`, `vertex.model`,
`vertex.fast_model`.

```bash
stockresearch config show                       # all settings; editable ones marked
stockresearch config get universe.benchmark
stockresearch config set universe.benchmark QQQ
stockresearch config set universe.index NONE    # watchlist-only scans
stockresearch config set vertex.model gemini-2.5-flash
```

Other settings (quant params, runtime) stay YAML-only to avoid foot-guns — edit
`config/settings.yaml` directly for those.

### Markets other than the US (e.g. India / BSE & NSE)

The tool is market-agnostic via yfinance ticker suffixes and a configurable index/benchmark:

- **Tickers:** use the yfinance suffix — `.NS` for NSE, `.BO` for BSE (e.g. `RELIANCE.NS`,
  `INFY.BO`). Add them with `stockresearch watch add RELIANCE.NS TCS.NS`.
- **Index:** `stockresearch config set universe.index NIFTY50` (NIFTY 50 constituents,
  fetched from Wikipedia) — or `NONE` for watchlist-only.
- **Benchmark:** `stockresearch config set universe.benchmark '^NSEI'` (NIFTY 50) or
  `'^BSESN'` (SENSEX) so beta/alpha are measured against the right market.
- **Macro focus:** `stockresearch config set universe.region India` biases the macro and
  news scans toward India-specific drivers (RBI, SEBI, Union Budget, monsoon, FII/FPI flows,
  USD/INR) **plus** global events that move Indian markets (US Fed, crude oil).
- **Risk-free rate:** set `quant.risk_free_rate` in `settings.yaml` to your local level
  (e.g. `0.065` for the Indian 10Y G-Sec) so Sharpe is meaningful.

Fundamentals, prices, and ratios come back in the local currency (e.g. ₹). Note: the
optional Alpha Vantage cross-check uses `.BSE` symbols rather than yfinance's `.NS`/`.BO`,
so verification coverage for Indian tickers is partial.

### `track TICKER` — historical track record

Every past call for a ticker (lean, score, price, beta, alpha, Sharpe) across runs —
the basis for later scoring prediction accuracy against realized moves.

```bash
stockresearch track GOOGL
```

### `explain [TERM]` — learn the metrics

Plain-language glossary (no LLM, no cost). With no argument it lists every metric; with a
term it explains what it is, the formula, which statement it comes from, how to read it, and
caveats (including India notes).

```bash
stockresearch explain                 # list all terms
stockresearch explain pe              # P/E ratio
stockresearch explain debt_to_equity
```

### `score` / `calibration` / `export` — outcomes & training data

Turn your run history into a feedback loop and a labeled dataset for a future custom model.

```bash
stockresearch score                   # grade past calls vs realized 7/30/90d returns
stockresearch calibration             # hit-rate by confidence bucket and by lean
stockresearch export data/train.jsonl # dump {inputs, verdict, outcomes} as JSONL
```

`score` only grades calls old enough for each horizon (run daily and revisit). `calibration`
tells you whether the tool's confidence is meaningful and whether bullish/bearish actually
predict direction. `export` joins each run's stored snapshot (inputs), the verdict, and the
realized forward returns — the dataset to train/evaluate a finance model on.

### `portfolio` — track and review your holdings

```bash
stockresearch portfolio add RELIANCE.NS --qty 10 --price 1200
stockresearch portfolio remove TCS.NS
stockresearch portfolio list
stockresearch portfolio review        # live value, P&L, weights, diversification flags
```

`review` prices your holdings live, shows P&L and each position's weight and sector, pulls
the latest stored lean per holding, and flags concentration / heavy sector exposure / too few
names — framed as beginner education, **not advice**.

---

## Development

```bash
pytest            # unit tests (quant, store, mocked agents — no network)
ruff check src tests
```

Tests never hit the network: data adapters and the LLM are monkeypatched, so the suite is
fast and deterministic. Live behavior is verified manually via `analyze` / `scan`.

---

## Output example

`scan` produces a ranked table plus a saved Markdown report:

```
Ranked Analysis (4 scanned)
 # | Ticker | Lean    | Score | Conf | Beta | Alpha  | Sharpe
 1 | GOOGL  | bullish | 0.85  | 80%  | 1.10 | 41.3%  | 1.73
 2 | NVDA   | bullish | 0.85  | 75%  | 1.83 | 11.6%  | 0.78
 3 | MSFT   | neutral | 0.65  | 60%  | 0.88 | -16.8% | -0.20
 4 | AAPL   | neutral | 0.45  | 70%  | 1.19 | 0.6%   | 0.54
```

---

## Roadmap ideas

- Add SEC EDGAR as an additional (no-key, authoritative) verification source alongside
  Alpha Vantage.
- Train a custom finance model on the exported dataset and A/B it against Gemini.
- Scheduled daily runs (cron) with a pushed digest.
- Optional ADK `Agent`/`Runner` runtime for conversational follow-ups.

---

## Disclaimer

For research and educational purposes only. Not financial advice. Data may be delayed,
incomplete, or incorrect. Verify independently before making any investment decision.
