#!/usr/bin/env python3
"""Run every check of every entry and write the verification report.

    python tools/verify_all.py            # fast checks only
    python tools/verify_all.py --slow     # everything (the nightly run)

Writes reports/verification.json and reports/verification.md. Exit status 1 if any
check fails, so that a continuous-integration run turns red.
"""

from __future__ import annotations

import datetime as dt
import json
import platform
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy  # noqa: E402
import scipy  # noqa: E402

import kramers  # noqa: E402


def main() -> int:
    slow = "--slow" in sys.argv
    started = time.time()
    results = []
    failed = 0
    for entry in kramers.entries():
        for check in entry.checks:
            if check.slow and not slow:
                continue
            t0 = time.time()
            try:
                outcome = check.run()
                row = {"entry": entry.id, "title": entry.title, "check": check.name, "reference": check.reference,
                       "slow": check.slow, "passed": bool(outcome.passed), "worst": float(outcome.worst),
                       "tolerance": float(outcome.tolerance), "detail": outcome.detail}
            except Exception:  # noqa: BLE001 - a crash is a failed check, recorded with its trace
                row = {"entry": entry.id, "title": entry.title, "check": check.name, "reference": check.reference,
                       "slow": check.slow, "passed": False, "worst": float("nan"), "tolerance": float("nan"),
                       "detail": traceback.format_exc()}
            row["seconds"] = round(time.time() - t0, 2)
            failed += not row["passed"]
            results.append(row)
            mark = "pass" if row["passed"] else "FAIL"
            print(f"{entry.id} [{mark}] {check.name} ({row['seconds']} s)", flush=True)
    meta = {
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "kramers": kramers.__version__, "python": platform.python_version(), "numpy": numpy.__version__,
        "scipy": scipy.__version__, "platform": platform.platform(), "slow_checks_included": slow,
        "entries": len(kramers.entries()), "checks_run": len(results), "failed": failed,
        "seconds": round(time.time() - started, 1),
    }
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    (out / "verification.json").write_text(json.dumps({"meta": meta, "results": results}, indent=1))
    lines = ["# Verification report", "",
             f"Generated {meta['generated']} with kramers {meta['kramers']}, Python {meta['python']}, "
             f"NumPy {meta['numpy']}, SciPy {meta['scipy']}.", "",
             f"**{meta['checks_run'] - failed} of {meta['checks_run']} checks passed** across {meta['entries']} entries "
             f"({'including' if slow else 'without'} the slow simulation checks; {meta['seconds']} s).", "",
             "| Entry | Check | Reference | Worst | Tolerance | Result |", "|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['entry']} | {r['check']} | {r['reference']} | {r['worst']:.2e} | {r['tolerance']:.0e} | "
                     f"{'pass' if r['passed'] else '**FAIL**'} |")
    lines += ["", "The tables behind each line are in `verification.json`.", ""]
    (out / "verification.md").write_text("\n".join(lines))
    print(f"\n{meta['checks_run'] - failed}/{meta['checks_run']} checks passed in {meta['seconds']} s")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
