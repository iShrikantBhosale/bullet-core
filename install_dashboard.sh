#!/bin/bash
# install_dashboard.sh
# Installer for Bullet OS Training Dashboard

echo "========================================"
echo "🔵 Bullet OS Dashboard Installer"
echo "========================================"

# Check for Python 3
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found! Please install python3."
    exit 1
fi

echo "✅ Python 3 found"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "⬇️  Installing dependencies..."
pip install fastapi uvicorn torch python-multipart numpy

echo "✅ Installation complete!"
echo ""
echo "To start the dashboard:"
echo "  ./dashboard/start_dashboard.sh"
echo ""
echo "========================================"
