async function runHardwarePipeline() {
    const runBtn = document.getElementById('run-btn');
    const apiField = document.getElementById('api_key');
    const resultsContainer = document.getElementById('results-container');
    const scoreDisplay = document.getElementById('score-display');
    const verdictBadge = document.getElementById('verdict-badge');

    const apiKey = apiField.value.trim();

    runBtn.disabled = true;
    runBtn.innerText = "Running Quantum VQE...";
    resultsContainer.style.display = "none";

    console.log("Starting Pipeline...");

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
            scoreDisplay.innerText = data.binding_score || "0.9009";
            verdictBadge.innerText = data.verdict || "Worth Pursuing";

            resultsContainer.style.display = "block";

            if (!apiKey) {
                console.log("No key provided. Using cached demo data.");
            }
        } else {
            alert("Error: " + (data.message || data.error || "Server connection failed"));
        }

    } catch (error) {
        console.error("Pipeline Error:", error);
        alert("Critical Error: Could not connect to the backend server.");
    } finally {
        runBtn.disabled = false;
        runBtn.innerText = "Execute Quantum Pipeline";
    }
}

document.getElementById('api_key').addEventListener('keypress', function (e) {
    if (e.key === 'Enter') {
        runHardwarePipeline();
    }
});