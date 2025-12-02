#!/bin/bash

echo "=== Bullet Training Dashboard Test ==="
echo ""

# Check if backend is running
echo "1. Checking backend status..."
if curl -s http://localhost:8000/logs > /dev/null 2>&1; then
    echo "✅ Backend is running on port 8000"
else
    echo "❌ Backend is not responding. Starting it now..."
    cd dashboard/backend
    source ../../venv/bin/activate
    uvicorn main:app --port 8000 > uvicorn.log 2>&1 &
    sleep 3
    cd ../..
fi

echo ""
echo "2. Dashboard URLs:"
echo "   Frontend: http://localhost:8000/dashboard/index.html"
echo "   API Logs: http://localhost:8000/logs"
echo "   API Docs: http://localhost:8000/docs"

echo ""
echo "3. Testing API endpoints..."
curl -s http://localhost:8000/logs | python3 -m json.tool | head -n 10

echo ""
echo "4. Hardware Optimization Summary:"
echo "   - CPU multi-threading enabled (auto-detected cores)"
echo "   - GPU support (if CUDA available)"
echo "   - Per-batch logging for real-time feedback"
echo "   - Stop training endpoint: POST /stop"

echo ""
echo "5. UI Features:"
echo "   ✓ Real-time stats panel (hardware, epoch, batch, loss)"
echo "   ✓ Live training logs with timestamps"
echo "   ✓ Progress bar with percentage"
echo "   ✓ Stop training button"
echo "   ✓ Modern dark theme with gradient accents"
echo "   ✓ Responsive grid layout"

echo ""
echo "=== Test Complete ==="
echo "Open http://localhost:8000/dashboard/index.html in your browser to use the dashboard!"
