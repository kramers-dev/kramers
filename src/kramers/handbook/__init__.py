"""Handbook entries. Every module here registers one entry when imported.

To add an entry, copy ``tools/entry_template.py`` to a new module here; nothing else
needs to be edited. See CONTRIBUTING.md.
"""

import importlib
import pkgutil

for _m in sorted(pkgutil.iter_modules(__path__), key=lambda m: m.name):
    if _m.name.startswith("k"):
        importlib.import_module(f"{__name__}.{_m.name}")
