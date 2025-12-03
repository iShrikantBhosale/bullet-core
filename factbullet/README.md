# FactBullet - Setup Guide

## 1. Setup Local Scraper
FactBullet requires a local companion app to perform headless DuckDuckGo searches (bypassing browser CORS limits).

```bash
cd factbullet/scraper
npm install
node server.js
```
Keep this terminal open. It runs on `http://localhost:3000`.

## 2. Run the Web UI
You can serve the `docs` folder using any static file server.

```bash
# From bullet-core root
cd docs
python3 -m http.server 8000
```

## 3. Use FactBullet
Open `http://localhost:8000/factbullet/` in your browser.

- Enter a question.
- The AI will generate queries, fetch results from your local scraper, and summarize the answer.

## Note on Model
The system is configured to use `model_weights.bin` in the `docs` folder. Ensure this model is loaded.
The runtime has been updated to support English (removed Marathi-specific filters).
