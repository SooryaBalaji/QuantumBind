import sys
import os
import json
import time
import threading
import io
import base64
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from flask import Flask, render_template, jsonify
from concurrent.futures import ProcessPoolExecutor

app = Flask(__name__)

CACHE_FILE = "cached_results.json"
CACHE_TTL  = 3600
_cache_lock = threading.Lock()


def _run_vqe():
    from zne import run_vqe
    return run_vqe()

def _cache_is_fresh():
    if not os.path.exists(CACHE_FILE):
        return False
    return (time.time() - os.path.getmtime(CACHE_FILE)) < CACHE_TTL


def _load_cache():
    with open(CACHE_FILE) as f:
        return json.load(f)


def _save_cache(data: dict):
    with open(CACHE_FILE, "w") as f:
        json.dump(data, f)



def _run_zne():
    print("[zne] process started")
    from zne import run_zne
    print("[zne] imports done")
    result = run_zne()   # VQE is called inside here now
    print("[zne] finished")
    return result        # returns (vqe_energy, sherbrooke_energy, zne_energy)


def compute_results() -> dict:
    with _cache_lock:
        if _cache_is_fresh():
            return _load_cache()

        print("[timer] Starting ZNE pipeline...")
        t0 = time.time()

        vqe_energy, sherbrooke_energy, zne_energy = _run_zne()

        print(f"[timer] Pipeline done: {time.time() - t0:.2f}s")

        from decision import predict_binding
        score = predict_binding(vqe_energy, zne_energy)

        result = {
            "vqe_energy":        round(float(vqe_energy),        6),
            "sherbrooke_energy": round(float(sherbrooke_energy),  6),
            "zne_energy":        round(float(zne_energy),         6),
            "binding_score":     round(float(score),              4),
            "decision":          "WORTH PURSUING" if score > 0.5 else "REJECT",
            "computed_at":       time.time(),
        }
        _save_cache(result)
        return result

def generate_graph(vqe_energy: float, sherbrooke_energy: float, zne_energy: float) -> str:
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('#0f0f1a')
    ax.set_facecolor('#0f0f1a')

    labels = ['VQE (Ideal)', 'IBM Kingston\n(Hardware)', 'ZNE\n(Corrected)']
    values = [vqe_energy, sherbrooke_energy, zne_energy]
    colors = ['#6366f1', '#ef4444', '#10b981']

    bars = ax.bar(labels, values, color=colors, width=0.5)
    ax.set_ylabel('Energy (Hartree)', color='#e2e8f0')
    ax.tick_params(colors='#e2e8f0')
    for spine in ('bottom', 'left'):
        ax.spines[spine].set_color('#1a1a2e')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() - 0.02,
            f'{val:.4f}', ha='center', va='top', color='white', fontsize=10,
        )

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor='#0f0f1a')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return img_b64

def _prewarm():
    print("[prewarm] Starting background quantum computation…")
    try:
        compute_results()
        print("[prewarm] Cache populated.")
    except Exception as exc:
        print(f"[prewarm] Failed: {exc}")


threading.Thread(target=_prewarm, daemon=True).start()

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/run", methods=["POST"])
def run_pipeline():
    data  = compute_results()
    graph = generate_graph(
        data["vqe_energy"],
        data["sherbrooke_energy"],
        data["zne_energy"],
    )
    return jsonify({**data, "graph": graph})


@app.route("/hardware", methods=["GET"])
def hardware_results():
    try:
        with open("hardware_results.json") as f:
            return jsonify(json.load(f))
    except FileNotFoundError:
        return jsonify({"error": "Hardware results not available yet"}), 404


@app.route("/hardware-run", methods=["POST"])
def run_hardware_live():
    from hardware import run_hardware
    run_hardware()                           # writes hardware_results.json
    with open("hardware_results.json") as f:
        return jsonify(json.load(f))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=False, threaded=True)