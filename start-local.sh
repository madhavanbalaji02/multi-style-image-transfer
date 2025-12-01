#!/bin/bash

# Local Development Server Starter
# Starts both backend and frontend servers for local testing

set -e

echo "🚀 Starting Style Transfer App Locally"
echo "======================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

# Check if we're in the project directory
if [ ! -d "backend" ] || [ ! -d "frontend" ]; then
    echo "❌ Please run this script from the project root directory"
    echo "   cd /Users/madhavanbalaji/Documents/CV/project"
    exit 1
fi

# Check if backend dependencies are installed
echo "📦 Checking backend dependencies..."
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo "⚠️  Backend dependencies not installed."
    echo ""
    read -p "Install backend dependencies now? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Installing dependencies..."
        cd backend
        pip3 install -r requirements.txt
        cd ..
        echo "✅ Dependencies installed"
    else
        echo "❌ Cannot start backend without dependencies."
        echo "   Install manually with: cd backend && pip3 install -r requirements.txt"
        exit 1
    fi
else
    echo "✅ Backend dependencies found"
fi

echo ""
echo "🌐 Starting servers..."
echo ""
echo "Backend will run on:  http://localhost:8000"
echo "Frontend will run on: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop all servers"
echo ""

# Function to cleanup background processes on exit
cleanup() {
    echo ""
    echo "🛑 Stopping servers..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    echo "✅ Servers stopped"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start backend in background
echo "🔧 Starting backend server..."
cd backend
python3 -m uvicorn main:app --reload --host 0.0.0.0 --port 8000 > ../backend.log 2>&1 &
BACKEND_PID=$!
cd ..

# Wait a moment for backend to start
sleep 2

# Check if backend started successfully
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "❌ Backend failed to start. Check backend.log for errors."
    cat backend.log
    exit 1
fi

echo "✅ Backend started (PID: $BACKEND_PID)"

# Start frontend in background
echo "🎨 Starting frontend server..."
cd frontend
python3 -m http.server 3000 > ../frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..

# Wait a moment for frontend to start
sleep 1

# Check if frontend started successfully
if ! kill -0 $FRONTEND_PID 2>/dev/null; then
    echo "❌ Frontend failed to start. Check frontend.log for errors."
    kill $BACKEND_PID 2>/dev/null
    cat frontend.log
    exit 1
fi

echo "✅ Frontend started (PID: $FRONTEND_PID)"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✨ Application is ready!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🌐 Open your browser and visit:"
echo "   👉 http://localhost:3000"
echo ""
echo "📚 API Documentation available at:"
echo "   👉 http://localhost:8000/docs"
echo ""
echo "📋 Logs:"
echo "   Backend:  tail -f backend.log"
echo "   Frontend: tail -f frontend.log"
echo ""
echo "Press Ctrl+C to stop all servers"
echo ""

# Try to open browser automatically (macOS)
if command -v open &> /dev/null; then
    sleep 1
    open http://localhost:3000
fi

# Wait for user interrupt
wait
