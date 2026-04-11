import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from flask import Flask, render_template, jsonify
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
import io
warnings.filterwarnings("ignore")

app = Flask(__name__)

cached_results = None

def get_results():
    global cached_results
    if cached_results is None:
        print("Running quantum pipeline for first time...")
        from zne import run_zne
        from decision import predict_binding

        vqe_energy, sherbrooke_energy, zne_energy = run_zne()
        score = predict_binding(vqe_energy, zne_energy)

        cached_results = {
            "vqe_energy": round(float(vqe_energy), 6),
            "sherbrooke_energy": round(float(sherbrooke_energy), 6),
            "zne_energy": round(float(zne_energy), 6),
            "binding_score": round(float(score), 4),
            "decision": "Worth Pursuing" if score > 0.5 else "Reject"
        }
        print("Pipeline complete. Results cached.")
    return cached_results


def generate_graph(vqe_energy, sherbrooke_energy, zne_energy):
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('#0f0f1a')
    ax.set_facecolor('#0f0f1a')

    labels = ['VQE (Ideal)', 'IBM Kingston\n(Hardware)', 'ZNE\n(Corrected)']
    values = [vqe_energy, sherbrooke_energy, zne_energy]
    colors = ['#6366f1', '#ef4444', '#10b981']

    bars = ax.bar(labels, values, color=colors, width=0.5)
    ax.set_ylabel('Energy (Hartree)', color='#e2e8f0')
    ax.tick_params(colors='#e2e8f0')
    ax.spines['bottom'].set_color('#1a1a2e')
    ax.spines['left'].set_color('#1a1a2e')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.02,
                f'{val:.4f}', ha='center', va='top', color='white', fontsize=10)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor='#0f0f1a')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return img_base64

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/run", methods=["POST"])
def run_pipeline():
    data = get_results()
    graph = generate_graph(
        data['vqe_energy'],
        data['sherbrooke_energy'],
        data['zne_energy']
    )
    return jsonify({**data, "graph": graph})

@app.route("/hardware", methods=["GET"])
def hardware_results():
    import json
    try:
        with open("hardware_results.json", "r") as f:
            data = json.load(f)
        return jsonify(data)
    except:
        return jsonify({"error": "Hardware results not available yet"})

@app.route("/hardware-run", methods=["POST"])
def run_hardware_live():
    import json
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ''))
    from hardware import run_hardware
    energy = run_hardware()
    with open("hardware_results.json", "r") as f:
        data = json.load(f)
    return jsonify(data)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=False)
