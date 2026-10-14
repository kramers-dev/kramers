# Working rules for this repository

These rules apply to every contributor, human or AI agent. `CLAUDE.md` points here.

## Set up and test

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest                 # fast checks
.venv/bin/python tools/verify_all.py --slow  # all checks, writes reports/
.venv/bin/python tools/build_handbook.py   # pages, handbook.json, README table
.venv/bin/ruff check .
```

## The rule on claims

Every scientific statement carries its source: a printed equation (paper, equation
number, page, version read), or a computed check (the script or check that produced
it). A statement that has neither is removed. In particular:

- An entry's primary source must have been opened; its equation is transcribed as
  printed. Sources not opened are labelled `evidence="citation-only"`.
- A number in a test or a check names its origin and states its tolerance with the
  reason.
- A formula outside its validity raises `OutsideDomain` or warns with
  `ValidityWarning`; it never returns a silent number.
- Use of the handbook by anyone is never invented; a person is never named as a
  checker or contributor without written consent.
- A misprint in a paper is reported only with a numerical test that shows it, and is
  worded as a question to the authors until they answer.

## History and authorship

- Commit in small steps with accurate messages. Never rewrite pushed history.
- `AUTHORSHIP.md` records who designed, wrote, reviewed and tested each part.
- Keep the `Co-Authored-By:` trailer that an AI tool adds to a commit.
- Nothing is pushed, released or posted anywhere without the maintainer's say-so.

## Parts reserved for the founding author

Listed in `RESERVED_FOR_AUTHOR.md`. An AI agent may specify and review these, and may
not write them.
