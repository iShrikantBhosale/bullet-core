# Bullet-Core Training Dashboard

Modern, real-time training dashboard for monitoring Bullet-Core Engine training.

## Features

- 📊 **Real-time Loss Charts** - Track training and validation loss
- 📈 **Learning Rate Monitoring** - Visualize LR schedule
- 🎮 **Training Controls** - Start, pause, stop training
- 💾 **GPU Monitoring** - VRAM usage and device info
- 📝 **Training Logs** - Real-time event logging
- 🎨 **Modern UI** - Dark theme with glassmorphism

## Quick Start

### 1. Install Dependencies

```bash
pip install flask flask-cors
```

### 2. Start the Server

```bash
cd /home/shri/Desktop/bulletOs/bullet_core/dashboard
python server.py
```

### 3. Open Dashboard

Navigate to: **http://localhost:5000**

## Architecture

```
dashboard/
├── index.html      # Main dashboard UI
├── style.css       # Modern styling
├── app.js          # Frontend logic & charts
├── server.py       # Flask backend API
└── README.md       # This file
```

## API Endpoints

### Get Training Status
```
GET /api/training/status
```

### Start Training
```
POST /api/training/start
Body: { "max_epochs": 100 }
```

### Pause Training
```
POST /api/training/pause
```

### Stop Training
```
POST /api/training/stop
```

### Get GPU Info
```
GET /api/gpu/info
```

## Integration with Bullet-Core

The dashboard automatically integrates with your Bullet-Core models:

```python
from bullet_core import nn
from bullet_core.optim import AdamW
from bullet_core.scheduler import WarmupCosineAnnealing

# Your model
model = nn.Sequential(...)

# The server.py will use this for training
# Just start the server and click "Start Training"
```

## Features in Detail

### Real-time Metrics
- Current epoch
- Train/validation loss
- Learning rate
- Time per epoch
- ETA

### GPU Monitoring
- Device name (GT 730)
- VRAM usage (live)
- Compute capability

### Charts
- Loss curves (train & val)
- Learning rate schedule
- Auto-updating every epoch

### Training Log
- Timestamped events
- Color-coded messages
- Auto-scroll

## Demo Mode

If the backend is not running, the dashboard runs in **demo mode** with simulated training data.

## Customization

### Change Theme Colors

Edit `style.css`:
```css
:root {
    --accent-primary: #6366f1;  /* Change this */
    --accent-secondary: #8b5cf6;
}
```

### Adjust Update Frequency

Edit `server.py`:
```python
time.sleep(0.5)  # Change delay between epochs
```

## Screenshots

The dashboard features:
- Modern dark theme
- Glassmorphism effects
- Smooth animations
- Responsive design

## Troubleshooting

**Dashboard not loading?**
- Check server is running on port 5000
- Try http://127.0.0.1:5000

**No training updates?**
- Check Flask server logs
- Verify Bullet-Core is installed

**CORS errors?**
- flask-cors is installed
- Server allows all origins

## Tech Stack

- **Frontend**: HTML5, CSS3, JavaScript
- **Charts**: Chart.js
- **Backend**: Flask (Python)
- **ML Framework**: Bullet-Core Engine

---

Built for Bullet-Core Engine 🚀
