

async function runPipeline() {
    const btn = document.getElementById('runBtn');
    const loading = document.getElementById('loading');
    const results = document.getElementById('results');

    btn.disabled = true;
    btn.textContent = 'Running...';
    loading.classList.remove('hidden');
    results.classList.add('hidden');

    try {
        const response = await fetch('/run', { method: 'POST' });

        if (!response.ok) {
            throw new Error(`Server Error: ${response.status}`);
        }

        const data = await response.json();

        document.getElementById('vqeEnergy').textContent = (data.vqe_energy || 0).toFixed(4);
        document.getElementById('sherbrookeEnergy').textContent = (data.sherbrooke_energy || 0).toFixed(4);
        document.getElementById('zneEnergy').textContent = (data.zne_energy || 0).toFixed(4);
        document.getElementById('bindingScore').textContent = (data.binding_score || 0).toFixed(4);

        const decision = document.getElementById('decision');
        decision.textContent = data.decision || "N/A";

        decision.className = 'decision ' + (data.binding_score > 0.5 ? 'pursue' : 'reject');

        if (data.graph) {
            const graphImg = document.getElementById('energyGraph');
            graphImg.src = 'data:image/png;base64,' + data.graph;
            graphImg.style.display = 'block';
        }

        loading.classList.add('hidden');
        results.classList.remove('hidden');

    } catch (err) {
        console.error("Pipeline execution failed:", err);
        btn.textContent = 'Server Error - Check Python Console';
    } finally {
        btn.disabled = false;
        if (btn.textContent !== 'Server Error - Check Python Console') {
            btn.textContent = 'Run Quantum Pipeline';
        }
    }
}

async function runHardware() {
    const loading = document.getElementById('hardwareLoading');
    const hwEnergy = document.getElementById('hwEnergy');
    const hwJobId = document.getElementById('hwJobId');

    loading.classList.remove('hidden');

    try {
        const response = await fetch('/hardware-run', { method: 'POST' });

        if (!response.ok) throw new Error("Hardware fetch failed");

        const data = await response.json();

        hwEnergy.textContent = (data.hardware_energy || 0).toFixed(4);
        hwJobId.textContent = data.job_id || "None";

    } catch (err) {
        console.error("Hardware run failed:", err);
    } finally {
        loading.classList.add('hidden');
    }
}

const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry, index) => {
        if (entry.isIntersecting) {
            setTimeout(() => {
                entry.target.classList.add('visible');
            }, index * 100);
        }
    });
}, { threshold: 0.1 });

document.querySelectorAll('.fade-in').forEach(el => observer.observe(el));