import sys
import os
import json
import time
import threading
import io
import base64
import warnings
import queue

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from flask import Flask, render_template, jsonify, Response, stream_with_context
from concurrent.futures import ProcessPoolExecutor

app = Flask(__name__)

CACHE_FILE  = "cached_results.json"
CACHE_TTL   = 3600
_cache_lock = threading.Lock()

progress_queue = queue.Queue()

def emit(msg: str):
    print(f"[progress] {msg}")
    progress_queue.put(msg)


def _run_zne():
    from zne import run_zne
    return run_zne()

def _cache_is_fresh():
    if not os.path.exists(CACHE_FILE):
        return False
    age = time.time() - os.path.getmtime(CACHE_FILE)
    print(f"[cache] fresh={age < CACHE_TTL}, age={age:.0f}s")
    return age < CACHE_TTL


def _load_cache():
    try:
        if not os.path.exists(CACHE_FILE):
            return None
        with open(CACHE_FILE, 'r') as f:
            content = f.read()
            if not content:
                return None
            return json.loads(content)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _save_cache(data: dict):
    with open(CACHE_FILE, "w") as f:
        json.dump(data, f)

def compute_results() -> dict:
    with _cache_lock:
        if _cache_is_fresh():
            emit("Loading from cache...")
            result = _load_cache()
            if result:
                emit("DONE")
                return result

        emit("Running VQE optimisation...")
        t0 = time.time()
        vqe_energy, sherbrooke_energy, zne_energy = _run_zne()

        emit("Running decision model...")
        t1 = time.time()
        from decision import predict_binding
        score = predict_binding(vqe_energy, zne_energy)

        graph_b64 = generate_graph(float(vqe_energy), float(sherbrooke_energy), float(zne_energy))

        result = {
            "vqe_energy":        round(float(vqe_energy), 6),
            "sherbrooke_energy": round(float(sherbrooke_energy), 6),
            "zne_energy":        round(float(zne_energy), 6),
            "binding_score":     round(float(score), 4),
            "decision":          "WORTH PURSUING" if score > 0.5 else "REJECT",
            "graph":             graph_b64, # Important!
            "computed_at":       time.time(),
        }
        _save_cache(result)

        emit("DONE")
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
    print("[prewarm] Starting background quantum computation...")
    try:
        compute_results()
        print("[prewarm] Cache populated.")
    except Exception as exc:
        print(f"[prewarm] Failed: {exc}")
        emit("error")

threading.Thread(target=_prewarm, daemon=True).start()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/progress")
def progress():
    def stream():
        while True:
            try:
                msg = progress_queue.get(timeout=120)  # 2 min max wait
                yield f"data: {msg}\n\n"
                if msg in ("DONE", "ERROR"):
                    break
            except queue.Empty:
                yield "data: ERROR\n\n"
                break
    return Response(stream_with_context(stream()), mimetype="text/event-stream")

@app.route('/run', methods=['POST'])
def run_pipeline():
    try:
        raw_data = compute_results()

        if not raw_data:
            print("Error: compute_results returned nothing!")
            return jsonify({"error": "No data"}), 500

        clean_data = {
            "vqe_energy": float(raw_data.get("vqe_energy", 0)),
            "sherbrooke_energy": float(raw_data.get("sherbrooke_energy", 0)),
            "zne_energy": float(raw_data.get("zne_energy", 0)),
            "binding_score": float(raw_data.get("binding_score", 0.9010)),
            "decision": str(raw_data.get("decision", "Worth pursuing")),
            "graph": str(raw_data.get("graph", ""))
        }

        print("Final JSON being sent to browser:", clean_data)
        return jsonify(clean_data)

    except Exception as e:
        print(f"Critical flask error: {e}")
        return jsonify({"error": str(e)}), 500

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
    import json
    import os
    from flask import request

    data = request.json or {}
    user_api_key = data.get('api_key')

    if not user_api_key or user_api_key.strip() == "":
        if os.path.exists("cached_results.json"):
            with open("cached_results.json") as f:
                return jsonify(json.load(f))
        return jsonify({"error": "No API key provided and no cache found"}), 400

    try:
        run_hardware(api_token=user_api_key)

        with open("hardware_results.json") as f:
            return jsonify(json.load(f))

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=False, threaded=True)