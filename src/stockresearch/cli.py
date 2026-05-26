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
app.add_typer(watch_app, name="watch")
app.add_typer(config_app, name="config")
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
    from .store import db

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

    db.save_report(rpt)
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
):
    """Deep-dive a single ticker."""
    from .agents.orchestrator import run_single

    with console.status(f"Analyzing {ticker.upper()}…"):
        a = run_single(ticker, reflect=not no_reflect, verify=verify)
    report_mod.print_single(a, console)


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
    from .store import db

    rpt = db.load_report(Date.fromisoformat(date_str)) if date_str else db.latest_report()
    if not rpt:
        console.print("[red]No stored run found.[/red]")
        raise typer.Exit(1)
    report_mod.print_terminal(rpt, console)


@app.command()
def runs():
    """List stored runs."""
    from .store import db

    table = Table(title="Runs")
    for col in ("Date", "Universe", "Created"):
        table.add_column(col)
    for r in db.list_runs():
        table.add_row(r["run_date"], str(r["universe_size"]), str(r["created_at"]))
    console.print(table)


@app.command()
def track(ticker: str):
    """Show the historical track record of our calls for one ticker."""
    from .store import db

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


if __name__ == "__main__":
    app()
