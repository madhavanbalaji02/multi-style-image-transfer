# GitHub Push Guide

## ✅ Repository Ready!

Your code is committed and ready to push to GitHub. **Model weights and large files are excluded** via `.gitignore`.

---

## 📋 What's Included in Git

### ✅ Essential Code (Pushed)
- **Backend**: All Python scripts (`*.py`)
- **Frontend**: HTML, CSS, JavaScript
- **Documentation**: README, deployment guides
- **Dependencies**: requirements.txt
- **HPC Training Scripts**: LoRA training code
- **Utilities**: Shell scripts for deployment

### ❌ Excluded (Not Pushed)
- **Model weights** (*.safetensors, *.pth) - 6.1 MB+
- **Datasets** (images/, *.zip) - Large datasets
- **Uploads/Outputs** - Temporary generated files
- **Virtual environments** (venv/, sd_lora_env/)
- **Cache files** (.cache/, __pycache__/)
- **Logs** (*.log)
- **Screenshots/recordings** (*.webp, *.png artifacts)

**Total committed:** 60 files (~4K lines of code)

---

## 🚀 Push to GitHub

### Option 1: Create New Repository on GitHub

1. **Go to GitHub**: https://github.com/new
2. **Create repository**:
   - Name: `multi-style-image-transfer` (or your choice)
   - Description: "Multi-style image style transfer with custom Van Gogh LoRA"
   - Visibility: Public or Private
   - **DO NOT** initialize with README (we already have one)

3. **Copy the repository URL**, then run:

```bash
cd /Users/madhavanbalaji/Documents/CV/project

# Add remote
git remote add origin <YOUR_GITHUB_REPO_URL>

# Push to GitHub
git branch -M main
git push -u origin main
```

### Option 2: Push to Existing Repository

```bash
cd /Users/madhavanbalaji/Documents/CV/project

# Add remote
git remote add origin <YOUR_EXISTING_REPO_URL>

# Push
git branch -M main
git push -u origin main
```

---

## 📝 After Pushing

### For Collaborators Who Clone

They'll need to:

1. **Install dependencies**:
```bash
cd backend
pip install -r requirements.txt
```

2. **Get model weights** (optional):
   - Train their own LoRA (see HPC_SETUP.md)
   - Or contact you for the trained weights
   - Place in `backend/models/lora/lora_vangogh_final/`

3. **Run the app**:
```bash
./start-local.sh
```

### Download Weights Separately

Since GitHub has 100MB file limit, you can:
- Host model weights on Google Drive
- Use Git LFS (Large File Storage)
- Provide download link in README

Example for Google Drive:
```bash
# Download Van Gogh LoRA weights
# https://drive.google.com/file/d/<YOUR_FILE_ID>
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=<FILE_ID>' -O lora_vangogh_final.zip
unzip lora_vangogh_final.zip -d backend/models/lora/
```

---

## 🔐 Important Notes

- **API Keys**: If you add any API keys later, use `.env` file (already in .gitignore)
- **Secrets**: Never commit passwords, tokens, or API keys
- **Large Files**: Keep models and datasets out of Git
- **Updates**: Use `git add .` and `git commit -m "message"` for future changes

---

## Ready Commands

**After creating GitHub repo**, run these:

```bash
cd /Users/madhavanbalaji/Documents/CV/project

# Replace with your actual GitHub URL
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git

git branch -M main
git push -u origin main
```

**You'll see something like:**
```
Enumerating objects: 60, done.
Counting objects: 100% (60/60), done.
...
To https://github.com/YOUR_USERNAME/YOUR_REPO.git
 * [new branch]      main -> main
```

---

## 🎉 Success!

Your code will be on GitHub with:
- Clean, organized structure
- Comprehensive documentation
- No large model files
- Ready for collaboration
