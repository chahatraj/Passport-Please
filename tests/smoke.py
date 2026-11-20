"""GPU smoke run of passport_please.ipynb outside Colab.

Runs every code cell in order with chosen form values, saves each figure and a JSON log of
outputs and timings. Usage: python tests/smoke.py OUT_DIR [--share]
"""
import json, re, sys, time
from pathlib import Path

import types
try:
    import IPython  # noqa: F401
except ImportError:  # stand-in for Colab's display helpers
    disp = types.ModuleType("IPython.display")
    disp.display = lambda *a, **k: print(*[getattr(x, "data", x) for x in a])
    disp.Markdown = lambda s: types.SimpleNamespace(data=s)
    ip = types.ModuleType("IPython"); ip.get_ipython = lambda: None; ip.display = disp
    sys.modules["IPython"] = ip; sys.modules["IPython.display"] = disp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

nb_path = Path(__file__).resolve().parent.parent / "passport_please.ipynb"
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
share = "--share" in sys.argv
cells = ["".join(c["source"]) for c in json.load(open(nb_path))["cells"] if c["cell_type"] == "code"]
by_title = {re.search(r"#@title (.*)", c).group(1): c for c in cells}

fig_n = [0]
def show(*a, **k):
    fig_n[0] += 1
    plt.savefig(out / f"fig{fig_n[0]:02d}_{STEP}.png", dpi=80); plt.close("all")
plt.show = show

ns = {}
log = []
STEP = "setup"

def run(title, **params):
    global STEP
    src = by_title[title]
    for k, v in params.items():
        src, n = re.subn(rf"^{k} = .*$", f"{k} = {v!r}", src, flags=re.M)
        assert n == 1, (title, k)
    STEP = re.sub(r"\W+", "_", title)[:20] + "_" + "_".join(re.sub(r"\W+", "", str(v))[:12] for v in params.values())
    t = time.time()
    exec(compile(src, title, "exec"), ns)
    dt = time.time() - t
    rec = ns["RECORDS"][-1] if ns.get("RECORDS") and title not in ("Setup (run once, takes about 3 minutes)",) else {}
    entry = {"step": STEP, "seconds": round(dt, 1), "params": params,
             **{k: rec.get(k) for k in ("labels", "outputs", "guessed", "target", "prompt", "instruction")}}
    log.append(entry)
    print(f"[{dt:6.1f}s] {STEP}: {entry.get('labels') or entry.get('guessed')}", flush=True)
    json.dump(log, open(out / "log.json", "w"), indent=1)

run("Setup (run once, takes about 3 minutes)")
if not share:
    ns["LOG_URL"] = ""
ns["MY_COUNTRY"] = "Nigeria"
print("target", ns["MY_COUNTRY"])
for scene in ["a family breakfast", "a wedding day"]:
    for detail in ["", "jollof rice", "a name Chinedu"[:20]]:
        run("1a. Text to text: a short story", scene=scene, detail=detail)
for scene in ["a family breakfast", "a wedding"]:
    for detail in ["", "jollof rice"]:
        run("1b. Text to image: a picture", scene=scene, detail=detail)
        run("1c. Image to text: a model looks at the picture")
for scene in ["a wedding day", "a kid's first day at school"]:
    for added in ["", "Set it anywhere in the world."]:
        run("2a. Text to text", scene=scene, added=added)
for scene in ["a wedding", "a family breakfast", "a street at night", "a doctor at work"]:
    for added in ["", "anywhere in the world"]:
        run("2b. Text to image", scene=scene, added=added)
        run("2c. Image to text")
run("Your summary")
time.sleep(15)  # let background uploads finish
print("done")
