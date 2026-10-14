#!/usr/bin/env python3
"""Generate the handbook's pages and data file from the registry.

    python tools/build_handbook.py

Writes docs/handbook/index.md, one page per entry (docs/handbook/K0001.md ...),
docs/errata.md, handbook.json, and the entry table in README.md between the markers
<!-- entries:start --> and <!-- entries:end -->. Latest check outcomes are taken from
reports/verification.json when it exists.
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import kramers  # noqa: E402
from kramers.registry import AREAS, KINDS  # noqa: E402


def _verification() -> dict[tuple[str, str], dict]:
    path = ROOT / "reports" / "verification.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {(r["entry"], r["check"]): r | {"generated": data["meta"]["generated"]} for r in data["results"]}


def _doi_link(doi: str | None) -> str:
    return f" [doi:{doi}](https://doi.org/{doi})" if doi else ""


def entry_page(e: kramers.Entry, ver: dict) -> str:
    sig = f"{e.function.__name__}{inspect.signature(e.function)}"
    lines = [f"# {e.id}. {e.title}", "",
             f"**Area:** {AREAS[e.area]}. **Kind:** {e.kind} ({KINDS[e.kind].lower()}).", "",
             e.statement, "", "## Model", "", f"$$ {e.model} $$", "", "## Result", "", f"$$ {e.formula} $$", "",
             "## Use it", "", "```python", f"from kramers.formulas import {e.function.__name__}",
             f"{sig}", "```", ""]
    doc = inspect.getdoc(e.function) or ""
    lines += ["```text", doc, "```", "", "| Parameter | Meaning |", "|---|---|"]
    lines += [f"| `{k}` | {v} |" for k, v in e.parameters.items()]
    lines += ["", "## Where it is printed", ""]
    if not e.sources:
        lines += ["This entry is a computed reference: no closed form is claimed. Its checks below are its evidence.", ""]
    for s in e.sources:
        role = {"primary": "Primary source", "restatement": "Also printed in", "original": "Original work"}[s.role]
        lines += [f"- **{role}:** {s.citation}.{_doi_link(s.doi)}", f"  - Where: {s.locator}.", f"  - Read: {s.version_read}."]
        if s.printed:
            lines += [f"  - As printed: $ {s.printed} $"]
        if s.notation:
            lines += [f"  - Notation: {s.notation}."]
    lines += ["", "## When it holds", "", e.validity, "", "## How it is checked", "",
              "| Check | Against | Latest result |", "|---|---|---|"]
    for c in e.checks:
        r = ver.get((e.id, c.name))
        if r is None:
            res = "not yet run"
        else:
            res = (f"{'pass' if r['passed'] else 'FAIL'}: worst {r['worst']:.1e} (tolerance {r['tolerance']:.0e}), "
                   f"{r['generated'][:10]}")
        lines.append(f"| {c.name}{' (slow, nightly)' if c.slow else ''} | {c.reference} | {res} |")
    if e.notes:
        lines += ["", "## Notes", "", e.notes]
    if e.aliases:
        lines += ["", "**Also known as:** " + "; ".join(e.aliases) + "."]
    if e.related:
        lines += ["", "**Related entries:** " + ", ".join(f"[{r}]({r}.md)" for r in e.related) + "."]
    lines += ["", "## Cite", "", "```text", e.cite(), "```", "",
              f"*Written by:* {', '.join(e.written_by)}. *Checked by people:* {', '.join(e.checked_by) or 'nobody yet; see CONTRIBUTING.md'}. "
              f"*Added:* {e.added}.", ""]
    return "\n".join(lines)


def main() -> None:
    ver = _verification()
    out = ROOT / "docs" / "handbook"
    out.mkdir(parents=True, exist_ok=True)
    entries = kramers.entries()
    for e in entries:
        (out / f"{e.id}.md").write_text(entry_page(e, ver))
    index = ["# The handbook", "", f"{len(entries)} entries. Each page gives the result, where it is printed, when it holds, "
             "and how it is checked.", ""]
    table = ["| Entry | Result | Kind | Primary source |", "|---|---|---|---|"]
    for area, name in AREAS.items():
        group = [e for e in entries if e.area == area]
        if not group:
            continue
        index += [f"## {name}", "", "| Entry | Result | Kind | Source |", "|---|---|---|---|"]
        for e in group:
            prim = next((s.citation for s in e.sources if s.role == "primary"), "computed here")
            index.append(f"| [{e.id}]({e.id}.md) | {e.title} | {e.kind} | {prim} |")
            table.append(f"| [{e.id}](docs/handbook/{e.id}.md) | {e.title} | {e.kind} | {prim} |")
        index.append("")
    (out / "index.md").write_text("\n".join(index))
    errata = ["# Errata found while building the handbook", "",
              "Misprints in published equations, found when an entry was implemented and checked. Each was "
              "verified numerically here; none of them affects the papers' main results unless stated. We "
              "have not found published errata for them. If you are an author and we are wrong, please say "
              "so in an issue; we will correct this page and credit you.", ""]
    for e in entries:
        if any(w in e.notes.lower() for w in ("erratum", "misprint")):
            errata += [f"## {e.id}. {e.title}", "", e.notes, ""]
    (ROOT / "docs" / "errata.md").write_text("\n".join(errata))
    (ROOT / "handbook.json").write_text(kramers.to_json())
    readme = ROOT / "README.md"
    if readme.exists():
        text = readme.read_text()
        start, end = "<!-- entries:start -->", "<!-- entries:end -->"
        if start in text and end in text:
            head, rest = text.split(start, 1)
            _, tail = rest.split(end, 1)
            readme.write_text(head + start + "\n" + "\n".join(table) + "\n" + end + tail)
    print(f"wrote {len(entries)} pages, the index, the errata page, handbook.json")


if __name__ == "__main__":
    main()
