# Authorship record

Who designed, wrote, reviewed and tested each part. Every statement about the use of
AI tools in this project is built from this file. "Pending" means the step has not
happened; it is not a claim that it will.

- **AS**: Alexander Sokol, founding author.
- **Claude**: Claude Code (Anthropic), model Claude Opus 5.5.
- **Claude agents**: research agents run by Claude, reading papers on the open web
  (no logins); their reports are kept in the author's private build folder.
- **Astra**: GPT-6 Astra (OpenAI), the second-model reviewer.

| Date | Part | Designed by | Written by | Reviewed by | Tested by |
|---|---|---|---|---|---|
| 2026-10-12 | The project's shape (a verified handbook of first-passage results; entries as the unit; registry; growth plan) | AS asked for a large, collaborative statistical-physics project; Claude proposed the design in AS's project plan | Claude | AS: pending | not applicable |
| 2026-10-14 | `src/kramers/solvers/` (quadrature, Fokker-Planck, simulation, Laplace) | Claude | Claude | Astra: pending. AS: pending | Claude, against closed forms and 65-digit reference values (`tests/`) |
| 2026-10-14 | `src/kramers/registry.py`, `_checking.py`, `formulas.py`, tools | Claude | Claude | Astra: pending. AS: pending | Claude (`tests/test_registry.py`) |
| 2026-10-14 | Entries K0001-K0030 (`src/kramers/handbook/`) | Choice of results: Claude, from the seed list in AS's project plan | Claude; every source located, opened and transcribed by Claude agents, with equation numbers and versions recorded per entry | Astra: pending. AS: pending | Claude: each entry's checks (`tools/verify_all.py --slow`) |
| 2026-10-14 | The 65-digit reference values in K0008 | Astra (method and code, 9 October 2026, for geoews) | Astra; reduced by Claude | not applicable | Claude reproduced them |
| 2026-10-14 | `docs/errata.md` findings | Found by Claude agents while verifying sources; each re-checked numerically by Claude, and the Haenggi-Talkner-Borkovec sign also on the journal scan | Claude | AS: pending, before any author is contacted | Claude |
| 2026-10-14 | README, CONTRIBUTING, GOVERNANCE, HOW_THIS_IS_MADE, CODE_OF_CONDUCT | Claude, from AS's project plan | Claude | AS: pending | not applicable |

Parts reserved for AS are listed in [RESERVED_FOR_AUTHOR.md](RESERVED_FOR_AUTHOR.md).
