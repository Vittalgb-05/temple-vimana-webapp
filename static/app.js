const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const predictBtn = document.getElementById('predict-btn');
const loading = document.getElementById('loading');
const results = document.getElementById('results');
const errorMsg = document.getElementById('error-msg');

let currentFile = null;

// Drag & Drop
dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
        handleFile(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) {
        handleFile(e.target.files[0]);
    }
});

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        showError('Please upload an image file.');
        return;
    }
    currentFile = file;
    // Show preview logic could go here
    predictBtn.disabled = false;
    predictBtn.innerText = `Analyze ${file.name}`;
    errorMsg.classList.add('hidden');
    results.classList.add('hidden');
}

predictBtn.addEventListener('click', async () => {
    if (!currentFile) return;

    setLoading(true);

    const formData = new FormData();
    formData.append('file', currentFile);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            displayResults(data);
        } else {
            showError(data.error || 'Server error occurred.');
        }
    } catch (err) {
        showError('Network error. Ensure server is running.');
        console.error(err);
    } finally {
        setLoading(false);
    }
});

function displayResults(data) {
    results.classList.remove('hidden');

    const predEl = document.getElementById('pred-class');
    const styleEl = document.getElementById('pred-style');

    predEl.innerText = data.class;
    styleEl.innerText = data.detected_style;

    if (data.class.includes("Not a Temple")) {
        predEl.style.color = "#dc2626"; // Red
        styleEl.style.color = "#dc2626";
    } else {
        predEl.style.color = "#2563eb"; // Blue
        styleEl.style.color = "#9333ea"; // Purple
    }

    if (data.segmentation_overlay) {
        document.getElementById('seg-img').src = `data:image/jpeg;base64,${data.segmentation_overlay}`;
    }
    if (data.gradcam) {
        document.getElementById('cam-img').src = `data:image/jpeg;base64,${data.gradcam}`;
    }
    if (data.cropped_region) {
        document.getElementById('crop-img').src = `data:image/jpeg;base64,${data.cropped_region}`;
    }

    const table = document.getElementById('probs-table');
    table.innerHTML = '';

    // Sort probs
    const sorted = Object.entries(data.probabilities).sort(([, a], [, b]) => b - a);

    sorted.forEach(([label, prob]) => {
        const percent = (prob * 100).toFixed(1);
        const row = document.createElement('div');
        row.className = 'prob-row';
        row.innerHTML = `
            <span style="width: 60px; font-weight: 500;">${label}</span>
            <div class="bar-container">
                <div class="bar" style="width: ${percent}%"></div>
            </div>
            <span style="width: 40px; text-align: right;">${percent}%</span>
        `;
        table.appendChild(row);
    });
}

function setLoading(isLoading) {
    if (isLoading) {
        loading.classList.remove('hidden');
        results.classList.add('hidden');
        predictBtn.disabled = true;
        dropZone.style.opacity = '0.5';
    } else {
        loading.classList.add('hidden');
        predictBtn.disabled = false;
        dropZone.style.opacity = '1';
    }
}

function showError(msg) {
    errorMsg.innerText = msg;
    errorMsg.classList.remove('hidden');
}
