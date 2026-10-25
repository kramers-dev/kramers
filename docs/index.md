# kramers

An open handbook of first-passage and escape results in statistical physics. Each published
formula is a Python function tied to its paper, equation and page, and no entry is accepted
unless an independent numerical check passes.

- **[The handbook](handbook/index.md):** every entry, grouped by area, with its source, when it
  holds, and the checks it passed.
- **[Questions for authors](errata.md):** printed equations that disagree with our
  computations, listed as questions until the authors answer.
- **Install:** `pip install kramers` (from the first release).
- **Source and issues:** [github.com/kramers-dev/kramers](https://github.com/kramers-dev/kramers).

## How it is made

kramers is developed with AI-assisted tools. Nothing here asks to be trusted: every equation's
source was opened and its location recorded, and every number is recomputed by the checks each
night.

## Cite

Cite the original paper on the entry's page first, then the software (see `CITATION.cff`).
