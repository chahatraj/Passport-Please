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
    rec = ns["RECORDS"][-1] if ns.get("RECORDS") and title not in ("Setup (run once, takes about 4 minutes)",) else {}
    entry = {"step": STEP, "seconds": round(dt, 1), "params": params,
             **{k: rec.get(k) for k in ("labels", "outputs", "guessed", "target", "prompt", "instruction")}}
    log.append(entry)
    print(f"[{dt:6.1f}s] {STEP}: {entry.get('labels') or entry.get('guessed')}", flush=True)
    json.dump(log, open(out / "log.json", "w"), indent=1)

run("Setup (run once, takes about 4 minutes)")
if not share:
    ns["LOG_URL"] = ""
ns["MY_COUNTRY"] = "Nigeria"
print("target", ns["MY_COUNTRY"])
run("1a. Text to text, plain scene", scene_1a="a wedding day")
for detail in ["jollof rice", "Chinedu", "in Lagos"]:
    run("1a. Text to text, your turn", detail=detail)
run("1b. Text to image, plain scene", scene_1b="a family breakfast")
run("1b. Text to image, your turn", detail="jollof rice")
run("1c. Image to text: a second model reads your pictures")
run("2a. Text to text, plain scene", scene_2a="a wedding day")
run("2a. Text to text, your turn", added="Set it anywhere in the world.")
run("2b. Text to image, plain scene", scene_2b="a wedding")
run("2b. Text to image, your turn", added="in a different country each time")
run("2c. Image to text, plain scene")
run("2c. Image to text, your turn", added="in a different country each time")
run("Your summary")
time.sleep(15)  # let background uploads finish
print("done")
