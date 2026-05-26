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
| `src/stockresearch/agents/` | `universe`, `macro_agent`, `ticker_analyst` (+reflection), `orchestrator`, `reason` (structured-output helper), `schemas` (internal LLM output schemas). |
| `src/stockresearch/store/db.py` | SQLite persistence + track-record queries. |
| `src/stockresearch/report.py` | Markdown + `rich` terminal rendering. |
| `src/stockresearch/cli.py` | Typer CLI entrypoint. |
| `tests/` | Quant, store, and mocked-agent tests (no network). |

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
  index: SP500                   # SP500 | NONE
  benchmark: SPY                 # used for beta/alpha
  max_tickers: 50                # cost safety cap
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
```

Prints lean/score/confidence, the quant metrics, reasoning, pros/cons, and any data
warnings (including fundamentals mismatches). Verification runs by default here.

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

### `track TICKER` — historical track record

Every past call for a ticker (lean, score, price, beta, alpha, Sharpe) across runs —
the basis for later scoring prediction accuracy against realized moves.

```bash
stockresearch track GOOGL
```

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
- Prediction-accuracy scorer: grade past calls against realized forward returns.
- Scheduled daily runs (cron) with a pushed digest.
- Optional ADK `Agent`/`Runner` runtime for conversational follow-ups.

---

## Disclaimer

For research and educational purposes only. Not financial advice. Data may be delayed,
incomplete, or incorrect. Verify independently before making any investment decision.
