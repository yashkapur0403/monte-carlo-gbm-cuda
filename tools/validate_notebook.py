#!/usr/bin/env python3
"""Validate notebooks/kaggle_gbm_montecarlo.ipynb.

Checks:
  1. valid nbformat JSON with the expected cell/type layout (1 markdown + 5 code cells)
  2. every code cell compiles as Python (no unbound syntax errors, no magics)
  3. the embedded CUDA source is byte-identical to src/gbm_montecarlo.cu
     (guards against notebook/repo drift)
Run:  python tools/validate_notebook.py
"""
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "kaggle_gbm_montecarlo.ipynb"
SRC = ROOT / "src" / "gbm_montecarlo.cu"

nb = json.loads(NB.read_text(encoding="utf-8"))
assert nb["nbformat"] == 4, "notebook is not nbformat 4"
assert nb["metadata"].get("accelerator") == "GPU", "missing GPU accelerator metadata"
cells = nb["cells"]
kinds = [c["cell_type"] for c in cells]
print("cells:", kinds)
assert kinds.count("code") == 5, "expected exactly 5 substantive code cells"
assert kinds.count("markdown") == 1, "expected exactly 1 markdown header cell"

ok = True
for i, c in enumerate(cells):
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    if i != 2:  # cell 2 embeds C++, whose lines may legitimately start with '!'
        # A real magic looks like '%matplotlib inline' or '!pip install x'; a bare
        # '% (' is just a continuation of a Python %-format expression.
        m2 = re.search(r"^[ \t]*(%[A-Za-z]|![A-Za-z/.\"'])", src, re.M)
        assert not m2, "cell %d uses a shell/line magic: %r" % (i, m2.group(0))
    assert not src.startswith("%%"), "cell %d starts with a cell magic" % i
    try:
        compile(src, "<notebook-cell-%d>" % i, "exec")
        print("  cell %d: python syntax OK (%d lines)" % (i, len(src.splitlines())))
    except SyntaxError as e:
        ok = False
        print("  cell %d: SYNTAX ERROR %s" % (i, e))

# --- embedded source must be byte-identical to the repo file -----------------
cell3 = "".join(cells[2]["source"])
m = re.search(r"^SOURCE = r'''(.*)'''\s*$", cell3, re.S | re.M)
assert m, "could not locate the embedded SOURCE literal in the notebook"
embedded = m.group(1)
# The generator writes SOURCE = r'''\n<text>'''\n ; strip exactly that one newline.
if embedded.startswith("\n"):
    embedded = embedded[1:]
if embedded.endswith("\n"):
    embedded = embedded[:-1]
# Cell 3 writes that literal to src/gbm_montecarlo.cu, after stripping the two newlines
# the r'''...''' delimiters contribute. The archived source must be byte-identical to
# the repository file, so pin both halves of that contract here.
WRITE = 'SOURCE = SOURCE.strip("\\n") + "\\n"'
assert WRITE in cell3, "cell 3 no longer normalises the embedded source before writing it"
written = (embedded.strip("\n") + "\n").encode("utf-8")
if written == SRC.read_bytes():
    print("  source written by cell 3: byte-identical to src/gbm_montecarlo.cu (%d bytes)"
          % len(written))
else:
    ok = False
    print("  source written by cell 3: MISMATCH (%d bytes vs %d in the repo)"
          % (len(written), len(SRC.read_bytes())))

repo = SRC.read_text(encoding="utf-8")
if embedded == repo:
    print("  embedded CUDA source: byte-identical to src/gbm_montecarlo.cu (%d bytes)" % len(repo))
else:
    ok = False
    print("  embedded CUDA source: MISMATCH (embedded %d bytes, repo %d bytes)"
          % (len(embedded), len(repo)))
    import difflib
    d = list(difflib.unified_diff(repo.splitlines(), embedded.splitlines(),
                                  "repo", "notebook", lineterm=""))
    print("\n".join(d[:40]))

print("VALIDATION", "PASSED" if ok else "FAILED")
sys.exit(0 if ok else 1)
