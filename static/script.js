

async function runPipeline() {
    const btn = document.getElementById('runBtn');
    const loading = document.getElementById('loading');
    const results = document.getElementById('results');

    btn.disabled = true;
    btn.textContent = 'Running...';

    try {
        const response = await fetch('/run', { method: 'POST' });

        if (!response.ok) throw new Error("Server Error");

        const data = await response.json();

        // 1. Update the UI with your successful 0.9010 score
        document.getElementById('vqeEnergy').textContent = data.vqe_energy.toFixed(4);
        document.getElementById('bindingScore').textContent = data.binding_score.toFixed(4);
        document.getElementById('decision').textContent = data.decision;

        // 2. Reveal the results and hide the spinner
        loading.classList.add('hidden');
        results.classList.remove('hidden');

        // 3. Change button to "FINISHED"
        btn.textContent = 'FINISHED';
        btn.classList.add('success-state'); // Optional: add a green class in CSS

    } catch (err) {
        console.error(err);
        btn.textContent = 'Server Error - Check Python Console';
        btn.disabled = false;
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