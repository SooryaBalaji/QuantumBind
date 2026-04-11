import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from flask import Flask, render_template, jsonify
import warnings
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

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/run", methods=["POST"])
def run_pipeline():
    return jsonify(get_results())

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
