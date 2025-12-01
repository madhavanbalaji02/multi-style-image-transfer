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

    const styleSelect = document.getElementById("styleSelect");

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
        if (!file.type.startsWith('image/')) {
            alert('Please upload an image file');
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            previewImage.hidden = false;
            document.querySelector('.upload-content').hidden = true;
            generateBtn.disabled = false;
        };
        reader.readAsDataURL(file);

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch(`${API_BASE_URL}/upload`, {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            uploadedFilename = data.filename;
        } catch (error) {
            console.error('Upload failed:', error);
            alert('Image upload failed');
        }
    }

    // Generate Button Interaction
    generateBtn.addEventListener('click', async () => {
        if (!uploadedFilename) return;

        const selectedValue = styleSelect.value;
        let backendStyle = selectedValue;
        let backendModelType = "diffusion";

        // Map dropdown selection to backend parameters
        if (selectedValue === "vangogh_lora") {
            backendStyle = "vangogh";
            backendModelType = "diffusion";
        } else if (selectedValue === "vangogh_gan") {
            backendStyle = "vangogh";
            backendModelType = "gan";
        } else {
            // Standard styles (cubism, etc.) always use diffusion
            backendStyle = selectedValue;
            backendModelType = "diffusion";
        }

        generateBtn.disabled = true;
        resultsSection.hidden = false;
        loadingSpinner.hidden = false;
        resultsGrid.innerHTML = '';

        try {
            const formData = new FormData();
            formData.append('filename', uploadedFilename);
            formData.append('style', backendStyle);
            formData.append('model_type', backendModelType);

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
