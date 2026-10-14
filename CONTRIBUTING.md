# Contributing

Thank you. The unit of contribution is small on purpose: one published result, one
file, one hour.

## The fastest ways to help

1. **Check an entry (ten minutes).** Open its page in `docs/handbook/`, compare the
   equation and its conditions with the paper, and open an issue "Check: K00NN" saying
   "right" or "wrong because ...". If you are an author of the paper, say so: with your
   consent, the entry will name you as its checker.
2. **Report a missing result (five minutes).** Open an issue "Propose: <result>" with
   the paper and the equation number.
3. **Add an entry (about an hour).** See below.

## Your first entry in an hour

1. Pick an item from [OPEN_ENTRIES.md](OPEN_ENTRIES.md) and open an issue "Claim: O0NN"
   so that nobody duplicates it. Claims lapse after two weeks without a pull request.
2. Copy `tools/entry_template.py` to `src/kramers/handbook/k0NNN_short_name.py`, with
   the next free number. Nothing else needs to be edited: the entry is discovered,
   tested and documented automatically.
3. Fill in the function and the metadata. The rules:
   - **Open the source.** Record the equation number, the page or anchor, and the
     version you read. Transcribe the equation as printed in `printed`, in the
     paper's own symbols, and explain the mapping to yours in `notation`.
   - **Write at least one independent check.** It must compare your function with
     something that does not share its derivation: one of the solvers in
     `kramers.solvers`, a simulation, or the same result printed in another paper.
     Give a tolerance you can justify, and say in a comment why.
   - **State when it holds.** Assumptions, limits, and for approximate results how
     large the error is where you checked.
4. Run `python -m pytest` and `python tools/build_handbook.py`, and open a pull
   request.

You will get a first answer within 48 hours, and either a merge or a clear list of
changes within a week.

## If you used AI tools

Say so in the pull request, with what they did. The same rules apply: the source must
have been opened by someone, and the check must pass. You answer for the contribution.

## Credit

- The entry names who wrote it and who checked it.
- Every release's author list (in `CITATION.cff` and the release DOI) includes everyone
  who has merged an entry or checked one, if they agree.
- Authorship of a software paper is offered to everyone with substantive merged work.
- The original paper is always cited first on the entry's page.

## Style

NumPy and SciPy only at run time. Type hints and NumPy-style docstrings. Line length
88 (`ruff check .`). Random numbers only from a `numpy.random.Generator` with a seed.
A function outside its validity raises `kramers.OutsideDomain` or warns with
`kramers.ValidityWarning`, never returns a silent number.

Be kind; see [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
