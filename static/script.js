// Fetches quantum pipeline results from Flask backend and updates the UI
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
        const data = await response.json();

        document.getElementById('vqeEnergy').textContent = data.vqe_energy.toFixed(4);
        document.getElementById('sherbrookeEnergy').textContent = data.sherbrooke_energy.toFixed(4);
        document.getElementById('zneEnergy').textContent = data.zne_energy.toFixed(4);
        document.getElementById('bindingScore').textContent = data.binding_score.toFixed(4);

        const decision = document.getElementById('decision');
        decision.textContent = data.decision;
        decision.className = 'decision ' + (data.binding_score > 0.5 ? 'pursue' : 'reject');

        loading.classList.add('hidden');
        results.classList.remove('hidden');

    } catch (err) {
        console.error(err);
        btn.textContent = 'Error — try again';
    } finally {
        btn.disabled = false;
        btn.textContent = 'Run Quantum Pipeline';
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

async function runHardware() {
    const loading = document.getElementById('hardwareLoading');
    const hwEnergy = document.getElementById('hwEnergy');
    const hwJobId = document.getElementById('hwJobId');

    loading.classList.remove('hidden');

    try {
        const response = await fetch('/hardware-run', { method: 'POST' });
        const data = await response.json();

        document.getElementById('hwEnergy').textContent = data.hardware_energy.toFixed(4);
        document.getElementById('hwJobId').textContent = data.job_id;

    } catch (err) {
        console.error(err);
    } finally {
        loading.classList.add('hidden');
    }
}

const graph = document.getElementById('energyGraph');
if (data.graph) {
    const graph = document.getElementById('energyGraph');
    graph.src = 'data:image/png;base64,' + data.graph;
    graph.style.display = 'block';
}graph.style.display = 'block';