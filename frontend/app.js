// Configuration
const API_BASE_URL = "http://localhost:8000"; // Changed from 127.0.0.1 to localhost

document.addEventListener('DOMContentLoaded', async () => {
    // Connectivity Check
    try {
        const response = await fetch(`${API_BASE_URL}/docs`);
        if (!response.ok) {
            alert("Warning: Backend seems to be having issues.");
        }
    } catch (e) {
        alert("Error: Cannot connect to backend server at " + API_BASE_URL + ". Is it running?");
    }

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
            // Don't enable generate button yet - wait for upload to finish
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
            console.log("Upload successful, filename:", uploadedFilename);
        } catch (error) {
            console.error('Upload failed:', error);
            alert('Image upload failed');
        }
    }

    // Generate Button Interaction - Direct assignment to ensure it works
    generateBtn.onclick = async () => {
        console.log("Generate Clicked!");

        if (!uploadedFilename) {
            console.log("No Image Uploaded");
            alert("Please upload an image first!");
            return;
        }

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
        generateBtn.textContent = "Generating... 🎨";
        resultsSection.hidden = false;
        loadingSpinner.hidden = false;
        resultsGrid.innerHTML = '';

        console.log(`Generating: ${backendStyle} (${backendModelType})...`);

        try {
            const formData = new FormData();
            formData.append('filename', uploadedFilename);
            formData.append('style', backendStyle);
            formData.append('model', backendModelType);

            console.log("Sending request:", { filename: uploadedFilename, style: backendStyle, model: backendModelType });

            const response = await fetch(`${API_BASE_URL}/generate`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Generation failed: ${response.status} - ${errorText}`);
            }

            const results = await response.json();
            console.log("Results received:", results);
            displayResults(results);

        } catch (error) {
            console.error('Error:', error);
            alert(`Error: ${error.message}`);
        } finally {
            loadingSpinner.hidden = true;
            generateBtn.disabled = false;
            generateBtn.textContent = "Generate Art ✨";
        }
    };

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
