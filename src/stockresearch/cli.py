"""CLI entrypoint. Commands: scan, analyze, macro, report, track, runs, watch, config."""

from __future__ import annotations

import logging
from datetime import date as Date

import typer
from rich.console import Console
from rich.table import Table

from . import manage
from . import report as report_mod
from .gemini import configure_vertex_env

app = typer.Typer(help="Agentic daily stock research (Vertex AI Gemini).", no_args_is_help=True)
watch_app = typer.Typer(help="Manage tracked tickers (watchlist).", no_args_is_help=True)
config_app = typer.Typer(help="View and update settings.", no_args_is_help=True)
portfolio_app = typer.Typer(help="Track your holdings and review them.", no_args_is_help=True)
app.add_typer(watch_app, name="watch")
app.add_typer(config_app, name="config")
app.add_typer(portfolio_app, name="portfolio")
console = Console()


@app.callback()
def _setup(verbose: bool = typer.Option(False, "--verbose", "-v")):
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)
    configure_vertex_env()


@app.command()
def scan(
    no_news: bool = typer.Option(False, help="Skip per-ticker news (faster/cheaper)."),
    no_reflect: bool = typer.Option(False, help="Skip the reflection/critique pass."),
    no_macro: bool = typer.Option(False, help="Skip the macro/political scan."),
    verify: bool = typer.Option(
        False, help="Cross-check fundamentals vs 2nd source (uses Alpha Vantage quota)."
    ),
    limit: int = typer.Option(0, help="Cap number of tickers (0 = config max)."),
):
    """Run the full daily pipeline over the universe; save report + persist to DB."""
    from .agents.orchestrator import run_daily
    from .agents.universe import resolve_universe
    from .store import persist_analyses

    universe = resolve_universe()
    if limit > 0:
        universe = universe[:limit]

    with console.status(f"Scanning {len(universe)} tickers…") as status:
        done = {"n": 0}

        def tick(t: str):
            done["n"] += 1
            status.update(f"Analyzed {done['n']}/{len(universe)} ({t})")

        rpt = run_daily(
            universe, reflect=not no_reflect, fetch_news=not no_news,
            skip_macro=no_macro, verify=verify, progress=tick,
        )

    persist_analyses(
        rpt.run_date, rpt.analyses, rpt.snapshots,
        macro_summary=rpt.macro_summary, macro_signals=rpt.macro_signals,
        universe_size=rpt.universe_size, errors=rpt.errors,
    )
    path = report_mod.save_markdown(rpt)
    report_mod.print_terminal(rpt, console)
    console.print(f"\n[green]Saved[/green] {path} and persisted run {rpt.run_date}.")


@app.command()
def analyze(
    ticker: str,
    no_reflect: bool = typer.Option(False),
    verify: bool = typer.Option(
        True, "--verify/--no-verify", help="Cross-check fundamentals vs 2nd source."
    ),
    explain: bool = typer.Option(
        False, "--explain", help="Add a beginner walkthrough of the ratios and news linkage."
    ),
    no_save: bool = typer.Option(False, "--no-save", help="Don't record this analysis."),
):
    """Deep-dive a single ticker (recorded to the day's report unless --no-save)."""
    from .agents.orchestrator import run_single

    with console.status(f"Analyzing {ticker.upper()}…"):
        a, exp = run_single(
            ticker, reflect=not no_reflect, verify=verify, explain=explain, save=not no_save
        )
    report_mod.print_single(a, console)
    if exp:
        report_mod.print_explanation(exp, console)
    if not no_save:
        console.print(f"\n[dim]Recorded to {Date.today().isoformat()}'s report.[/dim]")


@app.command()
def advise(
    ticker: str,
    no_reflect: bool = typer.Option(False),
    verify: bool = typer.Option(True, "--verify/--no-verify"),
    no_save: bool = typer.Option(False, "--no-save", help="Don't record this recommendation."),
):
    """Actionable, capital-sized BUY/SELL/HOLD call for a ticker (logged for win-rate)."""
    from .agents.orchestrator import run_advice
    from .config import load_settings

    if load_settings().account.capital <= 0:
        console.print("[red]No capital configured.[/red] Set it first, e.g.:")
        console.print("  stockresearch config set account.capital 100000")
        raise typer.Exit(1)

    with console.status(f"Advising on {ticker.upper()}…"):
        analysis, rec = run_advice(
            ticker, reflect=not no_reflect, verify=verify, save=not no_save
        )
    report_mod.print_single(analysis, console)
    report_mod.print_recommendation(rec, console)
    if not no_save:
        console.print(f"[dim]Recorded recommendation for {Date.today().isoformat()}.[/dim]")


@app.command()
def recommendations(ticker: str = typer.Argument(None, help="Filter to one ticker")):
    """List logged buy/sell/hold recommendations."""
    from .store import get_store

    rows = get_store().recommendations_rows(ticker)
    if not rows:
        console.print("[yellow]No recommendations logged yet. Use `advise`.[/yellow]")
        return
    table = Table(title="Recommendations")
    for col in ("Date", "Ticker", "Action", "Shares", "Amount", "Target%", "Conf", "Price"):
        table.add_column(col)
    for r in rows:
        act = (r.get("action") or "").upper()
        color = {"BUY": "green", "SELL": "red"}.get(act, "yellow")
        table.add_row(
            r["run_date"], r["ticker"], f"[{color}]{act}[/{color}]",
            f"{(r.get('shares') or 0):g}", f"{(r.get('amount') or 0):,.0f}",
            f"{(r.get('target_pct') or 0) * 100:.0f}%", f"{(r.get('confidence') or 0):.0%}",
            f"{r.get('price_at_rec'):,.2f}" if r.get("price_at_rec") else "—",
        )
    console.print(table)


@app.command()
def winrate():
    """Win-rate of logged buy/sell calls (run `score` first to grade matured ones)."""
    from . import evaluate

    w = evaluate.winrate()
    if not w:
        console.print("[yellow]No matured recommendations scored yet. Run `score`.[/yellow]")
        raise typer.Exit()
    console.print(f"[bold]Win rate:[/bold] {w['win_rate']:.0%} over {w['total']} "
                  "matured buy/sell calls\n")
    t1 = Table(title="By action")
    for col in ("Action", "N", "Win rate", "Avg fwd return"):
        t1.add_column(col)
    for action, d in w["by_action"].items():
        wr = f"{d['wins'] / d['n']:.0%}" if d["n"] else "—"
        avg = f"{d['ret_sum'] / d['n'] * 100:.1f}%" if d["n"] else "—"
        t1.add_row(action, str(d["n"]), wr, avg)
    console.print(t1)
    t2 = Table(title="By horizon")
    for col in ("Horizon", "N", "Win rate"):
        t2.add_column(col)
    for h, d in w["by_horizon"].items():
        wr = f"{d['wins'] / d['n']:.0%}" if d["n"] else "—"
        t2.add_row(f"{h}d", str(d["n"]), wr)
    console.print(t2)


@app.command()
def explain(term: str = typer.Argument(None, help="Metric to explain, e.g. 'pe', 'sharpe'")):
    """Explain a financial metric in plain language (no arg = list all terms)."""
    from . import glossary

    if not term:
        console.print("[bold]Metrics you can explain:[/bold]")
        for t in glossary.all_terms():
            console.print(f"  [cyan]{t.key}[/cyan] — {t.name}")
        console.print("\n[dim]Run `stockresearch explain <term>` for details.[/dim]")
        return
    t = glossary.lookup(term)
    if not t:
        console.print(f"[red]No metric matches '{term}'.[/red] Try `stockresearch explain`.")
        raise typer.Exit(1)
    console.rule(f"[bold]{t.name}")
    console.print(f"[bold]What it is:[/bold] {t.definition}")
    console.print(f"[bold]Formula:[/bold] {t.formula}")
    console.print(f"[bold]Comes from:[/bold] {t.derived_from}")
    console.print(f"[bold]How to read it:[/bold] {t.how_to_read}")
    if t.india_note:
        console.print(f"[bold]India note:[/bold] {t.india_note}")
    console.print(f"[yellow]Caveat:[/yellow] {t.caveat}")


@app.command()
def macro():
    """Run only the political/macro news scan."""
    from .agents.macro_agent import run_macro_scan

    with console.status("Scanning macro/political landscape…"):
        summary, signals = run_macro_scan()
    if summary:
        console.print(f"[bold]Summary[/bold]\n{summary}\n")
    table = Table(title="Macro Signals")
    for col in ("Impact", "Headline", "Sectors", "Conf", "Sources"):
        table.add_column(col)
    for s in signals:
        if not s.sources:
            srcs = "[red]0[/red]"
        elif s.corroborated:
            srcs = str(len(s.sources))
        else:
            srcs = f"[yellow]{len(s.sources)} (single-source)[/yellow]"
        table.add_row(s.impact.value, s.headline, ", ".join(s.sectors) or "broad",
                      f"{s.confidence:.0%}", srcs)
    console.print(table)
    for s in signals:
        if s.sources:
            console.print(f"\n[dim]{s.headline}[/dim]")
            for src in s.sources:
                console.print(f"  [dim]- {src.title or src.source}[/dim] {src.url or ''}")


@app.command()
def report(date_str: str = typer.Argument(None, help="YYYY-MM-DD; default latest")):
    """Re-render a stored run from the database."""
    from .store import get_store
    db = get_store()

    rpt = db.load_report(Date.fromisoformat(date_str)) if date_str else db.latest_report()
    if not rpt:
        console.print("[red]No stored run found.[/red]")
        raise typer.Exit(1)
    report_mod.print_terminal(rpt, console)


@app.command()
def runs():
    """List stored runs."""
    from .store import get_store
    db = get_store()

    table = Table(title="Runs")
    for col in ("Date", "Universe", "Created"):
        table.add_column(col)
    for r in db.list_runs():
        table.add_row(r["run_date"], str(r["universe_size"]), str(r["created_at"]))
    console.print(table)


@app.command()
def score():
    """Grade past calls against realized forward returns (7/30/90d where matured)."""
    from . import evaluate

    with console.status("Scoring matured calls…"):
        n = evaluate.score_matured()
        r = evaluate.score_recommendations()
    console.print(f"[green]Scored {n} analysis outcomes and {r} recommendation outcomes.[/green]")
    if n == 0 and r == 0:
        console.print("[dim]No calls are old enough yet — run daily and revisit in a week.[/dim]")


@app.command()
def calibration():
    """Show whether stated confidence and leans match realized outcomes."""
    from . import evaluate

    c = evaluate.calibration()
    if not c:
        console.print("[yellow]No scored outcomes yet. Run `stockresearch score` first.[/yellow]")
        raise typer.Exit()
    console.print(f"[bold]Overall accuracy:[/bold] {c['overall_accuracy']:.0%} "
                  f"over {c['total']} outcomes\n")
    t1 = Table(title="Accuracy by stated confidence")
    for col in ("Confidence", "N", "Hit rate"):
        t1.add_column(col)
    for label, d in c["by_confidence"].items():
        hit = f"{d['correct'] / d['n']:.0%}" if d["n"] else "—"
        t1.add_row(label, str(d["n"]), hit)
    console.print(t1)
    t2 = Table(title="By lean")
    for col in ("Lean", "N", "Hit rate", "Avg fwd return"):
        t2.add_column(col)
    for lean, d in c["by_lean"].items():
        hit = f"{d['correct'] / d['n']:.0%}" if d["n"] else "—"
        avg = f"{d['ret_sum'] / d['n'] * 100:.1f}%" if d["n"] else "—"
        t2.add_row(lean, str(d["n"]), hit, avg)
    console.print(t2)


@app.command()
def export(path: str = typer.Argument(..., help="Output .jsonl path for training data")):
    """Export labeled training data (inputs + verdict + realized outcomes) as JSONL."""
    from . import evaluate

    n = evaluate.export_jsonl(path)
    console.print(f"[green]Wrote {n} rows[/green] to {path}")


@app.command()
def track(ticker: str):
    """Show the historical track record of our calls for one ticker."""
    from .store import get_store
    db = get_store()

    hist = db.ticker_history(ticker)
    if not hist:
        console.print(f"[yellow]No history for {ticker.upper()}.[/yellow]")
        raise typer.Exit()
    table = Table(title=f"{ticker.upper()} — call history")
    for col in ("Date", "Lean", "Conf", "Score", "Price", "Beta", "Alpha", "Sharpe"):
        table.add_column(col)
    for r in hist:
        table.add_row(
            r["run_date"], r["lean"] or "—", f"{(r['confidence'] or 0):.0%}",
            f"{(r['score'] or 0):.2f}",
            f"{r['last_price']:.2f}" if r["last_price"] else "—",
            f"{r['beta']:.2f}" if r["beta"] else "—",
            f"{r['alpha'] * 100:.1f}%" if r["alpha"] else "—",
            f"{r['sharpe']:.2f}" if r["sharpe"] else "—",
        )
    console.print(table)


# ---- watch: manage tracked tickers -----------------------------------------------------

@watch_app.command("list")
def watch_list():
    """List the tracked tickers (watchlist)."""
    tickers = manage.list_watchlist()
    if not tickers:
        console.print("[yellow]Watchlist is empty.[/yellow]")
        return
    console.print(f"[bold]{len(tickers)} tracked:[/bold] " + ", ".join(tickers))


@watch_app.command("add")
def watch_add(tickers: list[str] = typer.Argument(..., help="Tickers to add, e.g. AAPL NVDA")):
    """Add one or more tickers to the watchlist."""
    added, full = manage.add_watchlist(tickers)
    if added:
        console.print(f"[green]Added:[/green] {', '.join(added)}")
    else:
        console.print("[yellow]Nothing new to add.[/yellow]")
    console.print(f"[dim]Watchlist ({len(full)}): {', '.join(full)}[/dim]")


@watch_app.command("remove")
def watch_remove(tickers: list[str] = typer.Argument(..., help="Tickers to remove")):
    """Remove one or more tickers from the watchlist."""
    removed, full = manage.remove_watchlist(tickers)
    if removed:
        console.print(f"[red]Removed:[/red] {', '.join(removed)}")
    else:
        console.print("[yellow]None of those were in the watchlist.[/yellow]")
    console.print(f"[dim]Watchlist ({len(full)}): {', '.join(full) or '(empty)'}[/dim]")


@watch_app.command("clear")
def watch_clear(yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation.")):
    """Remove all tickers from the watchlist."""
    if not yes:
        typer.confirm("Clear the entire watchlist?", abort=True)
    manage.clear_watchlist()
    console.print("[green]Watchlist cleared.[/green]")


# ---- config: view and update settings --------------------------------------------------

@config_app.command("show")
def config_show():
    """Show all settings; editable keys are marked."""
    settings = manage.show_settings()
    table = Table(title="Settings")
    for col in ("Key", "Value", "Editable"):
        table.add_column(col)

    def walk(prefix: str, node):
        for k, v in node.items():
            key = f"{prefix}.{k}" if prefix else k
            if hasattr(v, "items"):
                walk(key, v)
            else:
                editable = "[green]yes[/green]" if key in manage.SETTABLE else ""
                table.add_row(key, str(v), editable)

    walk("", settings)
    console.print(table)


@config_app.command("get")
def config_get(key: str = typer.Argument(..., help="Dotted key, e.g. universe.benchmark")):
    """Print the value of one setting."""
    try:
        console.print(f"{key} = {manage.get_setting(key)}")
    except KeyError:
        console.print(f"[red]No such setting: {key}[/red]")
        raise typer.Exit(1) from None


@config_app.command("set")
def config_set(
    key: str = typer.Argument(..., help="Dotted key (must be an editable key)"),
    value: str = typer.Argument(..., help="New value"),
):
    """Update a setting (validated; only daily-use keys are editable)."""
    try:
        new = manage.set_setting(key, value)
    except (ValueError, KeyError) as e:
        console.print(f"[red]{e}[/red]")
        console.print("[dim]Editable keys:[/dim] " + ", ".join(sorted(manage.SETTABLE)))
        raise typer.Exit(1) from None
    console.print(f"[green]Set[/green] {key} = {new}")


# ---- portfolio: holdings & review ------------------------------------------------------

@portfolio_app.command("add")
def portfolio_add(
    ticker: str,
    qty: float = typer.Option(..., "--qty", help="Number of shares held."),
    price: float = typer.Option(..., "--price", help="Your average buy price."),
):
    """Add or update a holding."""
    holdings = manage.add_holding(ticker, qty, price)
    console.print(f"[green]Saved[/green] {ticker.upper()}: {qty} @ {price}")
    console.print(f"[dim]{len(holdings)} holding(s) tracked.[/dim]")


@portfolio_app.command("remove")
def portfolio_remove(ticker: str):
    """Remove a holding."""
    removed, holdings = manage.remove_holding(ticker)
    if removed:
        console.print(f"[red]Removed[/red] {ticker.upper()}")
    else:
        console.print(f"[yellow]{ticker.upper()} not in portfolio.[/yellow]")
    console.print(f"[dim]{len(holdings)} holding(s) tracked.[/dim]")


@portfolio_app.command("list")
def portfolio_list():
    """List your tracked holdings (as entered)."""
    holdings = manage.list_holdings()
    if not holdings:
        console.print("[yellow]No holdings. Add one with `portfolio add`.[/yellow]")
        return
    table = Table(title="Holdings")
    for col in ("Ticker", "Qty", "Avg price"):
        table.add_column(col)
    for h in holdings:
        table.add_row(h["ticker"], f"{h['qty']:g}", f"{h['avg_price']:g}")
    console.print(table)


@portfolio_app.command("review")
def portfolio_review():
    """Live value, P&L, weights, and beginner diversification checks."""
    from . import portfolio as pf

    with console.status("Pricing your holdings…"):
        r = pf.review()
    if not r:
        console.print("[yellow]No holdings. Add one with `portfolio add`.[/yellow]")
        return

    table = Table(title="Portfolio")
    for col in ("Ticker", "Qty", "Value", "P&L", "P&L %", "Weight", "Sector", "Lean"):
        table.add_column(col)
    for h in r["rows"]:
        pl = h["pl"]
        pl_color = "green" if (pl or 0) >= 0 else "red"
        table.add_row(
            h["ticker"], f"{h['qty']:g}",
            f"{h['value']:,.0f}" if h["value"] is not None else "—",
            f"[{pl_color}]{pl:,.0f}[/{pl_color}]" if pl is not None else "—",
            f"[{pl_color}]{h['pl_pct'] * 100:.1f}%[/{pl_color}]" if h["pl_pct"] is not None else "—",
            f"{h['weight'] * 100:.0f}%" if h["weight"] is not None else "—",
            h["sector"], h["latest_lean"] or "—",
        )
    console.print(table)
    pl_color = "green" if r["total_pl"] >= 0 else "red"
    console.print(
        f"\nTotal value: {r['total_value']:,.0f}  |  Cost: {r['total_cost']:,.0f}  |  "
        f"P&L: [{pl_color}]{r['total_pl']:,.0f}[/{pl_color}]"
    )
    if r["flags"]:
        console.print("\n[bold yellow]Things to think about:[/bold yellow]")
        for f in r["flags"]:
            console.print(f"  • {f}")
    console.print("\n[dim]Educational checks, not investment advice.[/dim]")


if __name__ == "__main__":
    app()
