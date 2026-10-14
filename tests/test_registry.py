"""The registry's own rules: numbering, sources, documentation, export."""

import json
import re

import pytest

import kramers
from kramers.registry import AREAS, KINDS

DOI = re.compile(r"^10\.\d{4,9}/\S+$")


def test_ids_are_contiguous_and_permanent_format():
    ids = [e.id for e in kramers.entries()]
    assert ids == [f"K{i:04d}" for i in range(1, len(ids) + 1)]


@pytest.mark.parametrize("entry", kramers.entries(), ids=lambda e: e.id)
def test_entry_is_complete(entry):
    assert entry.area in AREAS and entry.kind in KINDS
    assert entry.function.__doc__ and len(entry.function.__doc__) > 40
    assert entry.statement and entry.validity and entry.formula
    for s in entry.sources:
        if s.doi is not None:
            assert DOI.match(s.doi), s.doi
        assert s.version_read, f"{entry.id}: say which version of the source was read"
        if s.role == "primary":
            assert s.evidence == "opened", f"{entry.id}: a primary source must have been opened"
            assert s.locator and s.printed, f"{entry.id}: give the equation and its printed form"
    if entry.kind != "computed-reference":
        assert any(s.role == "primary" for s in entry.sources)


def test_export_and_search():
    data = json.loads(kramers.to_json())
    assert data["format"] == "kramers-handbook/1"
    assert len(data["entries"]) == len(kramers.entries())
    found = {e.id for e in kramers.search("resetting")}
    assert {"K0015", "K0016"} <= found


def test_formulas_namespace():
    from kramers import formulas
    assert formulas.passage_time_saddle_node is kramers.get("K0004").function
    assert "kramers_rate_overdamped" in dir(formulas)
