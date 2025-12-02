#!/bin/bash
# Bullet OS Training Dashboard Startup Script

echo "🔵 Starting Bullet OS Training Dashboard..."

# Navigate to backend directory
cd "$(dirname "$0")/backend"

# Activate virtual environment
if [ -f "../../venv/bin/activate" ]; then
    source ../../venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "❌ Virtual environment not found at ../../venv"
    exit 1
fi

# Check if required packages are installed
python3 -c "import fastapi, uvicorn, torch" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Required packages not installed. Installing..."
    pip install fastapi uvicorn torch python-multipart
fi

# Start the server
echo "🚀 Starting server on http://localhost:8000"
echo "📊 Dashboard will be available at: http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python3 main.py
