#!/bin/bash
echo "⚡ Starting APIForge..."

# Install dependencies
pip install -r requirements.txt -q

# Start FastAPI backend in background
echo "🚀 Starting backend on http://localhost:8000"
uvicorn backend.main:app --reload --port 8000 &
BACKEND_PID=$!

sleep 2

# Start Streamlit frontend
echo "🎨 Starting frontend on http://localhost:8501"
streamlit run frontend/app.py

# Cleanup on exit
kill $BACKEND_PID
