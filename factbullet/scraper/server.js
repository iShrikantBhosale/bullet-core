const express = require('express');
const puppeteer = require('puppeteer');
const cors = require('cors');

const app = express();
const PORT = 3000;

app.use(cors());
app.use(express.json());

app.get('/search', async (req, res) => {
    const query = req.query.q;
    if (!query) {
        return res.status(400).json({ error: 'Query parameter "q" is required' });
    }

    console.log(`Searching for: ${query}`);

    try {
        const browser = await puppeteer.launch({
            headless: "new",
            args: ['--no-sandbox', '--disable-setuid-sandbox']
        });
        const page = await browser.newPage();

        // Go to DuckDuckGo
        await page.goto(`https://duckduckgo.com/?q=${encodeURIComponent(query)}&t=h_&ia=web`, { waitUntil: 'domcontentloaded' });

        // Wait for results to load
        await page.waitForSelector('#react-layout', { timeout: 5000 }).catch(() => console.log("Timeout waiting for selector, proceeding anyway..."));

        // Extract results
        const results = await page.evaluate(() => {
            // DDG selectors can be tricky and change. Using a generic strategy.
            const items = document.querySelectorAll('article');
            const data = [];

            items.forEach(item => {
                const titleElement = item.querySelector('h2 a');
                const linkElement = item.querySelector('a[data-testid="result-title-a"]');
                // Try multiple snippet selectors
                const snippetElement = item.querySelector('[data-result="snippet"]') || item.querySelector('.Ogdw739hrj316lD00w29');

                if (titleElement && linkElement) {
                    data.push({
                        title: titleElement.innerText,
                        url: linkElement.href,
                        snippet: snippetElement ? snippetElement.innerText : ''
                    });
                }
            });
            return data.slice(0, 5); // Return top 5 results
        });

        await browser.close();
        res.json({ results });

    } catch (error) {
        console.error('Scraping error:', error);
        res.status(500).json({ error: 'Failed to fetch search results' });
    }
});

app.listen(PORT, () => {
    console.log(`FactBullet Scraper running at http://localhost:${PORT}`);
});
