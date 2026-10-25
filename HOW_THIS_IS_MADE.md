# How this is made

## Who does what

- **Alexander Sokol** designs and directs the project, decides what goes in, and signs
  every release.
- kramers is developed with AI-assisted tools; every number reruns.
- **Independent reviews** check the solvers and entries before a release, as an
  adversary: their job is to find what is wrong.
- **Contributors and the authors of the papers** check entries; their names appear on
  the entries they checked, with their consent.

## Why you can check it

Not because of who wrote it, but because of what every entry must show:

1. **The source was opened.** Each entry records the equation number, the page, and
   which version (journal, arXiv, PubMed Central) was read, and transcribes the
   equation as printed. A source that was not opened is labelled "citation only" and
   cannot be an entry's primary source.
2. **An independent check passes.** The check compares the entry with something that
   does not share its derivation: a different solver, a simulation, or the same result
   printed in a second paper. The tolerance and the worst disagreement are published
   in the verification report.
3. **Everything reruns.** `python tools/verify_all.py --slow` reproduces every check.

When a check exposed a mistake, it was more often in our code than in the
literature. When our computation disagrees with a printed equation, the case is listed
in [docs/errata.md](docs/errata.md) with the computation, as a question to the authors
until they answer.

A check only catches what it tests. The first independent review (14 October 2026)
found inputs on which the solvers' own error estimates vouched for wrong or unconverged
answers: a narrow feature between sample points, a position-dependent D at an infinite
end, a well hidden beyond a truncation point, and a slowly converging eigenvalue
iteration. Each was fixed or made to raise an error, each now has a test, and the
limits that remain are listed in the README.

## Rules we keep

- No result is claimed as new. Each entry gives its history.
- No use of the handbook is claimed unless the user has said so in writing.
- No person is named as a checker or contributor without their consent.
- Contributions made with AI tools are welcome under the same rules: a source opened,
  a check that passes, a person who answers for it.
- AI tools do not answer reviewers or authors on our behalf.

## What this does not mean

A passing check shows that an entry agrees with an independent computation of the
same model. It does not show that the model describes a given experiment; each entry's
"when it holds" section says what is assumed.
