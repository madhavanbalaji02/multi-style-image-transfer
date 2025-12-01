#!/bin/bash

# Stop Local Development Servers
# Kills any running uvicorn or http.server processes on ports 8000 and 8080

echo "🛑 Stopping local development servers..."
echo ""

# Find and kill processes on port 8000 (backend)
BACKEND_PID=$(lsof -ti:8000)
if [ ! -z "$BACKEND_PID" ]; then
    kill $BACKEND_PID 2>/dev/null
    echo "✅ Stopped backend server (port 8000)"
else
    echo "ℹ️  No backend server running on port 8000"
fi

# Find and kill processes on port 3000 (frontend)
FRONTEND_PID=$(lsof -ti:3000)
if [ ! -z "$FRONTEND_PID" ]; then
    kill $FRONTEND_PID 2>/dev/null
    echo "✅ Stopped frontend server (port 3000)"
else
    echo "ℹ️  No frontend server running on port 3000"
fi

echo ""
echo "✨ All servers stopped"
