"""Every entry's fast checks. The slow ones run nightly (tools/verify_all.py --slow)."""

import pytest

import kramers

FAST = [(e, c) for e in kramers.entries() for c in e.checks if not c.slow]


@pytest.mark.parametrize("entry, check", FAST, ids=[f"{e.id}-{i}" for i, (e, c) in enumerate(FAST)])
def test_check(entry, check):
    outcome = check.run()
    assert outcome.passed, f"{entry.id} {check.name}\n{outcome.detail}"
