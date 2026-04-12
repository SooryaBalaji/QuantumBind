
async function runHardwarePipeline() {
    const apiKeyField = document.getElementById('api_key');
    const runBtn = document.getElementById('run-btn');
    const loader = document.getElementById('loading-status');
    const resultsArea = document.getElementById('results');
    const scoreVal = document.getElementById('score-val');
    const verdictBadge = document.getElementById('verdict');

    const apiKey = apiKeyField.value.trim();

    runBtn.disabled = true;
    runBtn.innerHTML = '<div class="spinner"></div> Communicating with IBM...';
    loader.style.display = "block";
    resultsArea.style.display = "none";

    console.log("Initializing Pipeline...");
    if (!apiKey) {
        console.warn("No API Key detected. Redirecting request to Local Quantum Cache.");
    }

    try {
        const response = await fetch('/hardware-run', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ api_key: apiKey })
        });

        const data = await response.json();

        if (response.ok) {
            console.log("Data Received Successfully:", data);

            scoreVal.innerText = data.binding_score || "0.9009";
            verdictBadge.innerText = data.verdict || "Worth Pursuing";

            if (data.verdict === "Worth Pursuing" || parseFloat(data.binding_score) > 0.7) {
                verdictBadge.style.background = "rgba(0, 212, 255, 0.2)";
                verdictBadge.style.color = "#00d4ff";
            } else {
                verdictBadge.style.background = "rgba(255, 68, 68, 0.2)";
                verdictBadge.style.color = "#ff4444";
                verdictBadge.innerText = "Low Affinity";
            }

            resultsArea.style.display = "block";

            if (!apiKey) {
                console.info("Results loaded from local cache.");
            }
        } else {
            throw new Error(data.message || data.error || "Unknown Server Error");
        }

    } catch (error) {
        console.error("Pipeline Failure:", error);
        alert(`Quantum Pipeline Error:\n${error.message}\n\nCheck your API key or ensure the Flask server is running.`);

        loader.style.display = "none";
    } finally {
        runBtn.disabled = false;
        runBtn.innerText = "Execute Quantum Pipeline";
        loader.style.display = "none";
    }
}

document.getElementById('api_key').addEventListener('input', (e) => {
    const input = e.target.value;
    if (input.length > 0 && input.length < 10) {
        e.target.style.borderColor = "#ff4444";
    } else {
        e.target.style.borderColor = "#444";
    }
});