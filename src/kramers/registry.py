"""The handbook's registry: one published result per entry.

An entry is one short module in ``kramers/handbook``. It holds a function (the result,
implemented), the sources that print it (with the equation and where it was read), the
conditions under which it holds, and the checks that compare it with an independent
computation. Importing ``kramers`` discovers every entry module; documentation pages,
the handbook data file and the nightly verification are generated from the registry.

Entry numbers (``K0001``...) are permanent: once given, never reused or changed.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = [
    "AREAS",
    "KINDS",
    "Check",
    "CheckResult",
    "Entry",
    "Source",
    "entries",
    "get",
    "register",
    "search",
    "to_json",
]

AREAS: Mapping[str, str] = {
    "escape": "Escape over a barrier",
    "driven": "Driven escape and switching under a ramp",
    "targets": "Diffusion to a target",
    "resetting": "Stochastic resetting",
    "active": "Active and anomalous motion",
    "transport": "Transport in periodic potentials",
    "neurons": "Neuron models",
}

KINDS: Mapping[str, str] = {
    "exact": "Exact within the stated model",
    "asymptotic": "Leading term of an expansion; its error is mapped against an exact reference",
    "estimator": "Turns measured data into a quantity of the model",
    "computed-reference": "Computed here to stated precision; no closed form is claimed",
}

_ID = re.compile(r"^K\d{4}$")


@dataclass(frozen=True)
class Source:
    """A place where the result is printed.

    ``locator`` names the equation and page as printed in ``version_read``; ``printed``
    transcribes it (LaTeX) as it appears there, in the paper's own symbols; ``notation``
    says how those symbols map onto the entry's.
    """

    citation: str
    doi: str | None
    locator: str
    version_read: str
    printed: str = ""
    notation: str = ""
    role: str = "primary"  # primary | restatement | original
    evidence: str = "opened"  # opened | citation-only

    def __post_init__(self) -> None:
        if self.role not in {"primary", "restatement", "original"}:
            raise ValueError(f"unknown source role {self.role!r}")
        if self.evidence not in {"opened", "citation-only"}:
            raise ValueError(f"unknown evidence label {self.evidence!r}")


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one check: the worst discrepancy found and the tolerance it had."""

    passed: bool
    worst: float
    tolerance: float
    detail: str

    def __str__(self) -> str:
        mark = "pass" if self.passed else "FAIL"
        return f"{mark}: worst {self.worst:.2e} (tolerance {self.tolerance:.0e})\n{self.detail}"


@dataclass(frozen=True)
class Check:
    """A comparison of the entry with something independent of its derivation."""

    name: str
    reference: str
    run: Callable[[], CheckResult]
    slow: bool = False


@dataclass(frozen=True)
class Entry:
    """One result of the handbook."""

    id: str
    slug: str
    title: str
    area: str
    kind: str
    statement: str
    model: str
    formula: str
    function: Callable[..., Any]
    parameters: Mapping[str, str]
    sources: tuple[Source, ...]
    validity: str
    checks: tuple[Check, ...]
    aliases: tuple[str, ...] = ()
    related: tuple[str, ...] = ()
    notes: str = ""
    written_by: tuple[str, ...] = ("A. Sokol (AI-assisted)",)
    checked_by: tuple[str, ...] = ()
    added: str = ""
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _ID.match(self.id):
            raise ValueError(f"entry id {self.id!r} must look like K0001")
        if self.area not in AREAS:
            raise ValueError(f"{self.id}: unknown area {self.area!r}")
        if self.kind not in KINDS:
            raise ValueError(f"{self.id}: unknown kind {self.kind!r}")
        if not any(s.role == "primary" for s in self.sources) and self.kind != "computed-reference":
            raise ValueError(f"{self.id}: needs at least one primary source")
        if not self.checks:
            raise ValueError(f"{self.id}: needs at least one check")

    def cite(self) -> str:
        """The sources to cite when using this entry, primary first."""
        order = {"primary": 0, "original": 1, "restatement": 2}
        lines = [s.citation + (f", doi:{s.doi}" if s.doi else "")
                 for s in sorted(self.sources, key=lambda s: order[s.role])]
        lines.append(f"Implemented and checked in kramers, entry {self.id}.")
        return "\n".join(lines)

    def verify(self, include_slow: bool = False) -> dict[str, CheckResult]:
        """Run the checks; returns their outcomes by name."""
        return {c.name: c.run() for c in self.checks if include_slow or not c.slow}

    def summary(self) -> dict[str, Any]:
        d = {k: v for k, v in asdict(self).items() if k not in {"function", "checks", "extra"}}
        d["function"] = f"{self.function.__module__}.{self.function.__name__}"
        d["checks"] = [{"name": c.name, "reference": c.reference, "slow": c.slow} for c in self.checks]
        return d


_REGISTRY: dict[str, Entry] = {}


def register(entry: Entry) -> Entry:
    """Add an entry. Ids, slugs and function names must be new."""
    if entry.id in _REGISTRY:
        raise ValueError(f"entry {entry.id} is already registered")
    for other in _REGISTRY.values():
        if other.slug == entry.slug:
            raise ValueError(f"slug {entry.slug!r} is taken by {other.id}")
        if other.function.__name__ == entry.function.__name__:
            raise ValueError(f"function name {entry.function.__name__!r} is taken by {other.id}")
    _REGISTRY[entry.id] = entry
    return entry


def _load() -> None:
    import importlib

    importlib.import_module("kramers.handbook")  # registers every entry on import


def get(key: str) -> Entry:
    """An entry by id (``"K0004"``), slug or function name."""
    _load()
    if key in _REGISTRY:
        return _REGISTRY[key]
    for e in _REGISTRY.values():
        if key in {e.slug, e.function.__name__}:
            return e
    raise KeyError(f"no entry {key!r}")


def entries(area: str | None = None, kind: str | None = None) -> list[Entry]:
    """All entries, in order of their numbers, optionally filtered."""
    _load()
    out = sorted(_REGISTRY.values(), key=lambda e: e.id)
    if area is not None:
        out = [e for e in out if e.area == area]
    if kind is not None:
        out = [e for e in out if e.kind == kind]
    return out


def search(text: str) -> list[Entry]:
    """Entries whose title, statement, aliases or sources mention every word of ``text``."""
    words = [w.lower() for w in text.split()]
    hits = []
    for e in entries():
        hay = " ".join([e.title, e.statement, e.model, " ".join(e.aliases),
                        " ".join(s.citation for s in e.sources)]).lower()
        if all(w in hay for w in words):
            hits.append(e)
    return hits


def to_json(selection: Iterable[Entry] | None = None, indent: int = 2) -> str:
    """The handbook (or a selection) as JSON, for other tools and other languages."""
    items = list(selection) if selection is not None else entries()
    return json.dumps({"format": "kramers-handbook/1", "entries": [e.summary() for e in items]},
                      indent=indent, ensure_ascii=False)
