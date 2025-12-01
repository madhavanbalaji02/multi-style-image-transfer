# Deployment Guide for Render.com

This guide walks you through deploying the Multi-Style Image Style Transfer application to Render.com.

## Prerequisites

- GitHub account
- Render.com account (free tier is sufficient to start)
- Git installed locally

## Step 1: Prepare Your GitHub Repository

### 1.1 Create a New Repository on GitHub

1. Go to [GitHub](https://github.com) and sign in
2. Click the "+" icon in the top right and select "New repository"
3. Name it (e.g., `style-transfer-app`)
4. Choose "Public" or "Private" (both work with Render)
5. **Do NOT** initialize with README (we already have one)
6. Click "Create repository"

### 1.2 Push Your Code to GitHub

```bash
# Navigate to your project directory
cd /Users/madhavanbalaji/Documents/CV/project

# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: Style transfer application"

# Add remote (replace with your repository URL)
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## Step 2: Deploy Backend to Render

### 2.1 Create Web Service

1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click "New +" → "Web Service"
3. Connect your GitHub account if not already connected
4. Select your repository
5. Configure the service:
   - **Name**: `style-transfer-backend` (or your choice)
   - **Region**: Choose closest to you
   - **Root Directory**: `backend`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn -w 1 -k uvicorn.workers.UvicornWorker main:app`
   - **Instance Type**: Start with "Free" (upgrade if needed)

### 2.2 Add Environment Variables (Optional)

If you need any environment variables, add them in the "Environment" section:
- `PYTHON_VERSION`: `3.10.0`

### 2.3 Add Persistent Disk

1. Scroll to "Disks" section
2. Click "Add Disk"
3. **Name**: `style-transfer-disk`
4. **Mount Path**: `/opt/render/project/src/backend/uploads`
5. **Size**: `1 GB` (free tier allows up to 1GB)

### 2.4 Deploy

1. Click "Create Web Service"
2. Wait for deployment (10-15 minutes for first build)
3. Monitor logs for any errors
4. Once deployed, note your backend URL: `https://YOUR_SERVICE_NAME.onrender.com`

### 2.5 Test Backend

Visit `https://YOUR_SERVICE_NAME.onrender.com/docs` to see the FastAPI documentation and verify it's working.

## Step 3: Deploy Frontend to Render

### 3.1 Update Frontend Configuration

**IMPORTANT**: Before deploying frontend, update the API URL in `frontend/script.js`:

```javascript
// Change this line:
const API_BASE_URL = "http://localhost:8000";

// To your deployed backend URL:
const API_BASE_URL = "https://YOUR_BACKEND_SERVICE_NAME.onrender.com";
```

Commit and push this change:
```bash
git add frontend/script.js
git commit -m "Update API URL for production"
git push
```

### 3.2 Create Static Site

1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click "New +" → "Static Site"
3. Select your repository
4. Configure the site:
   - **Name**: `style-transfer-frontend` (or your choice)
   - **Root Directory**: `frontend`
   - **Build Command**: `echo "Static site build"`
   - **Publish Directory**: `./`

### 3.3 Deploy

1. Click "Create Static Site"
2. Wait for deployment (2-3 minutes)
3. Once deployed, note your frontend URL: `https://YOUR_SITE_NAME.onrender.com`

## Step 4: Update CORS Settings

### 4.1 Update Backend CORS

Now that you have your frontend URL, update the backend to allow requests from it:

1. Edit `backend/main.py`:
```python
# Change this:
allow_origins=["*"],

# To:
allow_origins=["https://YOUR_FRONTEND_SITE_NAME.onrender.com"],
```

2. Commit and push:
```bash
git add backend/main.py
git commit -m "Update CORS for production frontend"
git push
```

3. Render will automatically redeploy the backend

## Step 5: Test Your Application

1. Visit your frontend URL: `https://YOUR_SITE_NAME.onrender.com`
2. Upload a test image
3. Select a style (e.g., "mosaic")
4. Choose "GAN" model (faster for testing)
5. Click "Generate Art"
6. Verify the result appears and can be downloaded

## Troubleshooting

### Backend Issues

**Build Fails**:
- Check logs in Render dashboard
- Verify `requirements.txt` is correct
- TensorFlow + PyTorch together is large (~2-3GB), may take time

**Out of Memory**:
- Free tier has limited memory
- Consider upgrading to paid tier
- Or remove Diffusion model (keep only GAN)

**Slow Response**:
- Free tier instances spin down after inactivity
- First request after idle may take 30-60 seconds
- Consider upgrading for better performance

### Frontend Issues

**Can't Connect to Backend**:
- Verify `API_BASE_URL` in `script.js` is correct
- Check browser console for CORS errors
- Verify backend CORS settings include frontend URL

**Images Not Loading**:
- Check backend logs for errors
- Verify uploads directory has write permissions
- Check if disk is properly mounted

### Model Issues

**Diffusion Model Fails**:
- Requires significant resources (GPU/memory)
- Free tier may not support it
- Consider using only GAN model on free tier
- Or upgrade to paid tier with more resources

## Free Tier Limitations

- **Backend**: 512 MB RAM, 0.1 CPU
- **Disk**: 1 GB persistent storage
- **Spin Down**: Services spin down after 15 minutes of inactivity
- **Build Time**: Limited build minutes per month

## Recommended Next Steps

1. **Monitor Usage**: Check Render dashboard for resource usage
2. **Optimize**: Consider removing unused dependencies to reduce build size
3. **Upgrade**: If Diffusion model is needed, upgrade to paid tier
4. **Custom Domain**: Add custom domain in Render settings (optional)
5. **SSL**: Render provides free SSL certificates automatically

## Cost Estimates (if upgrading)

- **Starter Plan**: $7/month per service (512 MB RAM, 0.5 CPU)
- **Standard Plan**: $25/month per service (2 GB RAM, 1 CPU)
- **Disk**: Free up to 1GB, $0.25/GB/month after

For this application with both models, recommended:
- Backend: Standard Plan ($25/month) for Diffusion support
- Frontend: Free tier (static sites are free)
- Total: ~$25/month

## Support

- [Render Documentation](https://render.com/docs)
- [Render Community](https://community.render.com/)
- Check backend logs in Render dashboard for debugging
