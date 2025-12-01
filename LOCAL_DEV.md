# Local Development Quick Start

## Option 1: Manual Start (Recommended for Debugging)

### Terminal 1 - Start Backend
```bash
cd /Users/madhavanbalaji/Documents/CV/project/backend
python3 -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Terminal 2 - Start Frontend
```bash
cd /Users/madhavanbalaji/Documents/CV/project/frontend
python3 -m http.server 3000
```

### Open Browser
Visit: http://localhost:3000

---

## Option 2: Using Scripts

### Start Both Servers
```bash
cd /Users/madhavanbalaji/Documents/CV/project
./start-local.sh
```

### Stop Servers
Press `Ctrl+C` or run:
```bash
./stop-local.sh
```

---

## Troubleshooting

### Port Already in Use
If you get "port already in use" error:

**For port 8000:**
```bash
lsof -ti:8000 | xargs kill
```

**For port 3000:**
```bash
lsof -ti:3000 | xargs kill
```

### Backend Dependencies Missing
```bash
cd backend
pip3 install -r requirements.txt
```

### Check What's Running
```bash
lsof -i:8000  # Check backend
lsof -i:3000  # Check frontend
```

---

## Testing the Application

1. **Upload an image** - Click or drag & drop
2. **Select a style** - Choose from dropdown (e.g., "mosaic")
3. **Choose model** - Start with "GAN" (faster)
4. **Generate** - Click "Generate Art" button
5. **View results** - See original and styled images
6. **Download** - Click download button on results

---

## API Documentation

While backend is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
