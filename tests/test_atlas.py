"""The validity atlas: every map is stored, its claim follows from its data, and its data are current."""

import re
from pathlib import Path

import pytest

import kramers
from kramers import atlas

DOI = re.compile(r"^10\.\d{4,9}/\S+$")
MAPS = atlas.maps()


def test_map_ids_are_contiguous():
    assert [m.id for m in MAPS] == [f"M{i:02d}" for i in range(1, len(MAPS) + 1)]


@pytest.mark.parametrize("m", MAPS, ids=lambda m: m.id)
def test_map_is_complete_and_stored(m):
    known = {e.id for e in kramers.entries()}
    assert m.entries and set(m.entries) <= known
    assert m.status in {"draft", "approved"}
    for p in m.prior:
        assert DOI.match(p.doi), p.doi
        assert p.read, f"{m.id}: say what of {p.citation} was read"
    data = m.load()
    assert data["format"] == atlas.FORMAT and data["id"] == m.id
    assert data["axes"] and data["series"]


@pytest.mark.parametrize("m", MAPS, ids=lambda m: m.id)
def test_claim_follows_from_the_stored_data(m):
    result = m.checks()[0].run()
    assert result.passed, result.detail
    assert "nan" not in m.claim()


@pytest.mark.parametrize("m", MAPS, ids=lambda m: m.id)
def test_every_map_has_a_page_and_a_figure(m):
    docs = Path(__file__).parent.parent / "docs" / "atlas"
    assert (docs / f"{m.id}.png").exists(), "run tools/build_atlas.py --figures"
    page = (docs / f"{m.id}.md").read_text()
    assert m.claim() in page, "run tools/build_handbook.py"


def test_entries_link_to_their_maps():
    docs = Path(__file__).parent.parent / "docs" / "handbook"
    for m in MAPS:
        for e in m.entries:
            assert f"../atlas/{m.id}.md" in (docs / f"{e}.md").read_text(), f"{e} does not link to {m.id}"


@pytest.mark.slow
@pytest.mark.parametrize("m", MAPS, ids=lambda m: m.id)
def test_stored_map_is_current(m):
    result = m.checks()[1].run()
    assert result.passed, result.detail
