"""Render a DailyReport to Markdown (saved) and a rich terminal view."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.table import Table

from .config import load_settings
from .models import (  # noqa: F401
    Action, DailyReport, Explanation, Lean, NewsItem, Recommendation, TickerAnalysis,
)

_ACTION_COLOR = {Action.BUY: "green", Action.HOLD: "yellow", Action.SELL: "red"}

_LEAN_COLOR = {Lean.BULLISH: "green", Lean.NEUTRAL: "yellow", Lean.BEARISH: "red"}


def _md_link(src: NewsItem) -> str:
    title = src.title or src.source or "source"
    return f"[{title}]({src.url})" if src.url else title


def _fmt(v: float | None, pct: bool = False, nd: int = 2) -> str:
    if v is None:
        return "—"
    return f"{v * 100:.1f}%" if pct else f"{v:.{nd}f}"


def to_markdown(report: DailyReport) -> str:
    lines: list[str] = []
    lines.append(f"# Stock Research — {report.run_date.isoformat()}")
    lines.append("")
    lines.append(f"*Universe scanned: {report.universe_size} tickers. "
                 f"Analytical research, not investment advice.*")
    lines.append("")

    if report.macro_summary or report.macro_signals:
        lines.append("## Political / Macro Landscape")
        if report.macro_summary:
            lines.append(report.macro_summary)
            lines.append("")
        for s in report.macro_signals:
            tags = ", ".join(s.sectors) or "broad"
            flag = "" if s.corroborated else " ⚠ single-source"
            lines.append(f"- **[{s.impact.value}]** {s.headline} "
                         f"_(sectors: {tags}; conf {s.confidence:.0%}{flag})_")
            if s.sources:
                cites = ", ".join(_md_link(src) for src in s.sources)
                lines.append(f"  - Sources: {cites}")
            else:
                lines.append("  - _Sources: unattributed_")
        lines.append("")

    ranked = sorted(report.analyses, key=lambda a: a.score, reverse=True)
    lines.append("## Ranked Analysis")
    lines.append("")
    lines.append("| # | Ticker | Lean | Score | Conf | Beta | Alpha | Sharpe |")
    lines.append("|---|--------|------|-------|------|------|-------|--------|")
    for i, a in enumerate(ranked, 1):
        m = a.metrics
        lines.append(
            f"| {i} | {a.ticker} | {a.lean.value} | {a.score:.2f} | {a.confidence:.0%} | "
            f"{_fmt(m.beta)} | {_fmt(m.alpha, pct=True)} | {_fmt(m.sharpe)} |"
        )
    lines.append("")

    lines.append("## Detail")
    for a in ranked:
        lines.append("")
        lines.append(f"### {a.ticker} — {a.name or ''} ({a.lean.value}, score {a.score:.2f})")
        if a.reasoning:
            lines.append(a.reasoning)
        if a.pros:
            lines.append("\n**Pros:**")
            lines.extend(f"- {p}" for p in a.pros)
        if a.cons:
            lines.append("\n**Cons:**")
            lines.extend(f"- {c}" for c in a.cons)
        if a.sources:
            lines.append("\n**Sources:**")
            lines.extend(f"- {_md_link(src)}" for src in a.sources)
        if a.data_warnings:
            lines.append(f"\n_Data warnings: {', '.join(a.data_warnings)}_")

    if report.errors:
        lines.append("\n## Errors")
        lines.extend(f"- {e}" for e in report.errors)

    return "\n".join(lines) + "\n"


def save_markdown(report: DailyReport) -> Path:
    out_dir = load_settings().reports_path
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{report.run_date.isoformat()}.md"
    path.write_text(to_markdown(report))
    return path


def print_terminal(report: DailyReport, console: Console | None = None) -> None:
    console = console or Console()
    console.rule(f"[bold]Stock Research — {report.run_date.isoformat()}")
    if report.macro_summary:
        console.print(f"[dim]{report.macro_summary}[/dim]\n")

    table = Table(title=f"Ranked Analysis ({report.universe_size} scanned)")
    for col in ("#", "Ticker", "Lean", "Score", "Conf", "Beta", "Alpha", "Sharpe"):
        table.add_column(col)
    for i, a in enumerate(sorted(report.analyses, key=lambda x: x.score, reverse=True), 1):
        color = _LEAN_COLOR.get(a.lean, "white")
        m = a.metrics
        table.add_row(
            str(i), a.ticker, f"[{color}]{a.lean.value}[/{color}]", f"{a.score:.2f}",
            f"{a.confidence:.0%}", _fmt(m.beta), _fmt(m.alpha, pct=True), _fmt(m.sharpe),
        )
    console.print(table)
    if report.errors:
        console.print(f"[red]{len(report.errors)} ticker(s) failed[/red]")


def print_single(a: TickerAnalysis, console: Console | None = None) -> None:
    console = console or Console()
    color = _LEAN_COLOR.get(a.lean, "white")
    m = a.metrics
    console.rule(f"[bold]{a.ticker} — {a.name or ''}")
    console.print(f"Lean: [{color}]{a.lean.value}[/{color}]  "
                  f"Score: {a.score:.2f}  Confidence: {a.confidence:.0%}\n")
    console.print(
        f"Price: {_fmt(m.last_price)}  Beta: {_fmt(m.beta)}  Alpha: {_fmt(m.alpha, pct=True)}  "
        f"Sharpe: {_fmt(m.sharpe)}  Vol: {_fmt(m.annual_volatility, pct=True)}  "
        f"1Y: {_fmt(m.return_1y, pct=True)}  MaxDD: {_fmt(m.max_drawdown, pct=True)}\n"
    )
    if a.reasoning:
        console.print(a.reasoning + "\n")
    if a.pros:
        console.print("[green]Pros:[/green]")
        for p in a.pros:
            console.print(f"  + {p}")
    if a.cons:
        console.print("[red]Cons:[/red]")
        for c in a.cons:
            console.print(f"  - {c}")
    if a.sources:
        console.print("\n[dim]Sources:[/dim]")
        for src in a.sources:
            link = src.url or ""
            console.print(f"  [dim]- {src.title or src.source}[/dim] {link}")
    if a.data_warnings:
        console.print(f"\n[yellow]Data warnings: {', '.join(a.data_warnings)}[/yellow]")


def print_recommendation(rec: Recommendation, console: Console | None = None) -> None:
    console = console or Console()
    color = _ACTION_COLOR.get(rec.action, "white")
    console.rule(f"[bold]Recommendation — {rec.ticker}")
    console.print(f"Action: [{color}]{rec.action.value.upper()}[/{color}]  "
                  f"Confidence: {rec.confidence:.0%}")
    if rec.action != Action.HOLD and rec.shares:
        console.print(
            f"Size: {rec.shares:g} shares  (~{rec.amount:,.0f} @ {rec.price_at_rec:,.2f})  "
            f"target weight {rec.target_pct:.0%} of capital"
        )
    if rec.rationale:
        console.print(f"\n{rec.rationale}")
    for n in rec.notes:
        console.print(f"[yellow]Note:[/yellow] {n}")
    console.print("\n[dim]Sized suggestion for research/tracking — not financial advice.[/dim]")


def print_explanation(exp: Explanation, console: Console | None = None) -> None:
    console = console or Console()
    console.rule("[bold cyan]How to read this (beginner)")
    if exp.metric_lines:
        console.print("[bold]Metrics explained:[/bold]")
        for line in exp.metric_lines:
            console.print(f"  • {line}")
    if exp.walkthrough:
        console.print("\n[bold]Why this verdict:[/bold]")
        console.print(exp.walkthrough)
    if exp.news_links:
        console.print("\n[bold]How the news connects:[/bold]")
        for link in exp.news_links:
            console.print(f"  • {link}")
    console.print("\n[dim]Educational explanation — not investment advice.[/dim]")
