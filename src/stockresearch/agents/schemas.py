"""Internal structured-output schemas for the reasoning steps.

These isolate what the LLM is allowed to produce (judgments) from the deterministic
facts (metrics, ticker identity) that the pipeline fills in itself.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..models import Lean


class MacroSignalDraft(BaseModel):
    """What the structured pass emits. `source_ids` index into the numbered source list
    we provide, so URLs are resolved deterministically rather than written by the model."""

    headline: str
    summary: str = ""
    regions: list[str] = Field(default_factory=list)
    sectors: list[str] = Field(default_factory=list)
    affected_tickers: list[str] = Field(default_factory=list)
    impact: Lean = Lean.NEUTRAL
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    source_ids: list[int] = Field(
        default_factory=list,
        description="IDs of the supporting sources from the provided numbered list",
    )


class MacroDraftList(BaseModel):
    summary: str = Field(default="", description="2-3 sentence overall market-mood summary")
    signals: list[MacroSignalDraft] = Field(default_factory=list)


class AnalystVerdict(BaseModel):
    lean: Lean = Lean.NEUTRAL
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    score: float = Field(
        0.0, ge=0.0, le=1.0,
        description="composite attractiveness 0-1, higher = more attractive target",
    )
    pros: list[str] = Field(default_factory=list)
    cons: list[str] = Field(default_factory=list)
    reasoning: str = ""


class CritiqueResult(BaseModel):
    needs_revision: bool = False
    issues: list[str] = Field(default_factory=list)
    revised: AnalystVerdict
