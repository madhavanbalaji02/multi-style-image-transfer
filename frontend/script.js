// Configuration
const API_BASE_URL = "http://127.0.0.1:8000";

document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('fileInput');
    const uploadBox = document.getElementById('uploadBox');
    const previewImage = document.getElementById('previewImage');
    const generateBtn = document.getElementById('generateBtn');
    const resultsSection = document.getElementById('resultsSection');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const resultsGrid = document.getElementById('resultsGrid');

    let uploadedFilename = null;

    // Show/hide GAN checkbox based on style selection
    const styleSelect = document.getElementById("styleSelect");
    const ganCheckboxContainer = document.getElementById("ganCheckboxContainer");
    const ganCheckbox = document.getElementById("ganCheckbox");

    styleSelect.addEventListener("change", () => {
        const style = styleSelect.value;

        // Show GAN checkbox only for Van Gogh
        if (style === "vangogh") {
            ganCheckboxContainer.style.display = "block";
        } else {
            ganCheckboxContainer.style.display = "none";
            ganCheckbox.checked = false;
        }
    });

    // Upload Box Interactions
    uploadBox.addEventListener('click', () => fileInput.click());

    uploadBox.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadBox.style.borderColor = 'var(--primary-color)';
        uploadBox.style.backgroundColor = 'rgba(108, 92, 231, 0.1)';
    });

    uploadBox.addEventListener('dragleave', () => {
        uploadBox.style.borderColor = 'var(--secondary-color)';
        uploadBox.style.backgroundColor = 'var(--surface-dark)';
    });

    uploadBox.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadBox.style.borderColor = 'var(--secondary-color)';
        uploadBox.style.backgroundColor = 'var(--surface-dark)';

        if (e.dataTransfer.files.length > 0) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });

    async function handleFileUpload(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            previewImage.hidden = false;
            document.querySelector('.upload-content').hidden = true;
        };
        reader.readAsDataURL(file);

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch(`${API_BASE_URL}/upload`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error('Upload failed');

            const data = await response.json();
            uploadedFilename = data.filename;
            generateBtn.disabled = false;

        } catch (error) {
            console.error('Error:', error);
            alert('Failed to upload image. Please try again.');
        }
    }

    // Generate Button Interaction
    generateBtn.addEventListener('click', async () => {
        if (!uploadedFilename) return;

        const style = styleSelect.value;

        // Always use Diffusion, add GAN if checkbox is checked (Van Gogh only)
        const useGan = ganCheckbox.checked && style === "vangogh";
        const modelType = useGan ? "both" : "diffusion";

        generateBtn.disabled = true;
        resultsSection.hidden = false;
        loadingSpinner.hidden = false;
        resultsGrid.innerHTML = '';

        try {
            const formData = new FormData();
            formData.append('filename', uploadedFilename);
            formData.append('style', style);
            formData.append('model_type', modelType);

            const response = await fetch(`${API_BASE_URL}/generate`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error('Generation failed');

            const results = await response.json();
            displayResults(results);

        } catch (error) {
            console.error('Error:', error);
            alert(error.message);
        } finally {
            loadingSpinner.hidden = true;
            generateBtn.disabled = false;
        }
    });

    function displayResults(results) {
        addResultCard("Original", previewImage.src, null);

        if (results.diffusion) {
            addResultCard("Stable Diffusion", `${API_BASE_URL}${results.diffusion}`, results.diffusion);
        }

        if (results.gan) {
            addResultCard("GAN (CycleGAN)", `${API_BASE_URL}${results.gan}`, results.gan);
        }
    }

    function addResultCard(title, imageUrl, downloadPath) {
        const card = document.createElement('div');
        card.className = 'result-card';

        let downloadHtml = '';
        if (downloadPath) {
            downloadHtml = `<a href="${API_BASE_URL}${downloadPath}" download class="download-btn">Download</a>`;
        }

        card.innerHTML = `
            <img src="${imageUrl}" alt="${title}">
            <div class="result-info">
                <h3>${title}</h3>
                ${downloadHtml}
            </div>
        `;

        resultsGrid.appendChild(card);
    }
});
