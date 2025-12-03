// FactBullet App Logic

const SCRAPER_URL = 'http://localhost:3000/search';
const model = new BulletModel();

// UI Elements
const inputEl = document.getElementById('user-query');
const btnEl = document.getElementById('search-btn');
const statusArea = document.getElementById('status-area');
const resultArea = document.getElementById('result-area');
const debugLog = document.getElementById('debug-log');

// Steps
const step1 = document.getElementById('step-1');
const step2 = document.getElementById('step-2');
const step3 = document.getElementById('step-3');

const SYSTEM_PROMPT = `You are the FACTBULLET MODEL.
Purpose: Extract facts, eliminate noise, produce verified answers with citations.
Reasoning:
1. Understand Query
2. Extract Info
3. Verify Sources
4. Detect Contradictions
5. Compress Facts
6. Map Citations
7. Generate Answer
Rule: Never output raw text. Compress into clean points. Maintain precision.`;

function log(msg) {
    console.log(msg);
    debugLog.textContent += msg + '\n';
    debugLog.scrollTop = debugLog.scrollHeight;
}

async function init() {
    log('Initializing Model...');
    // Point to the parent directory for model files since we are in /factbullet/
    const success = await model.load('../model_weights.bin', '../tokenizer.json');
    if (success) {
        log('Model Loaded!');
        btnEl.disabled = false;
    } else {
        log('Failed to load model.');
        alert('Failed to load WASM model. Check console.');
    }
}

function setStatus(step, active) {
    if (active) {
        step.classList.add('active');
        step.style.opacity = '1';
    } else {
        step.classList.remove('active');
        step.style.opacity = '0.5';
    }
}

async function performSearch(query) {
    try {
        const response = await fetch(`${SCRAPER_URL}?q=${encodeURIComponent(query)}`);
        if (!response.ok) throw new Error('Scraper unreachable');
        const data = await response.json();
        return data.results || [];
    } catch (e) {
        log(`Search Error: ${e.message}`);
        return [];
    }
}

async function runAgent() {
    const userQuery = inputEl.value.trim();
    if (!userQuery) return;

    // Reset UI
    statusArea.classList.remove('hidden');
    resultArea.classList.add('hidden');
    resultArea.innerHTML = '';
    setStatus(step1, true);
    setStatus(step2, false);
    setStatus(step3, false);

    // Step 1: Generate Queries
    log(`User Query: ${userQuery}`);
    log('Generating search queries...');

    // Prompt for queries
    const queryPrompt = `Task: Generate 3 DuckDuckGo search queries for: "${userQuery}"\nFormat:\n- "query 1"\n- "query 2"\n- "query 3"\n\nQueries:`;

    let queries = [];
    try {
        const queryOutput = await model.generate(queryPrompt, {
            maxTokens: 60,
            temperature: 0.3,
            stopSequences: ["\n\n", "Task:"]
        });
        log(`Model Output: ${queryOutput}`);

        // Parse queries (simple regex)
        const matches = queryOutput.match(/"([^"]+)"/g);
        if (matches) {
            queries = matches.map(m => m.replace(/"/g, ''));
        } else {
            // Fallback if model fails to generate quoted queries
            queries = [userQuery];
        }
    } catch (e) {
        log(`Query Gen Error: ${e.message}. Using raw query.`);
        queries = [userQuery];
    }

    log(`Queries: ${JSON.stringify(queries)}`);

    // Step 2: Search
    setStatus(step1, false);
    setStatus(step2, true);

    log('Fetching from DuckDuckGo...');
    let allResults = [];

    // Search for each query (limit to top 2 queries to save time/bandwidth)
    for (const q of queries.slice(0, 2)) {
        const res = await performSearch(q);
        allResults = [...allResults, ...res];
    }

    // Deduplicate by URL
    const uniqueResults = [];
    const seenUrls = new Set();
    for (const r of allResults) {
        if (!seenUrls.has(r.url)) {
            seenUrls.add(r.url);
            uniqueResults.push(r);
        }
    }

    log(`Found ${uniqueResults.length} unique results.`);

    if (uniqueResults.length === 0) {
        resultArea.innerHTML = '<p>No results found or scraper is not running.</p>';
        resultArea.classList.remove('hidden');
        return;
    }

    // Step 3: Synthesize
    setStatus(step2, false);
    setStatus(step3, true);

    // Prepare context
    let context = "";
    uniqueResults.slice(0, 5).forEach((r, i) => { // Use top 5
        context += `[${i + 1}] ${r.snippet}\n`;
    });

    const answerPrompt = `${SYSTEM_PROMPT}\n\nFacts:\n${context}\nQuestion: ${userQuery}\nAnswer (summarize facts with citations [1], [2]...):`;
    log(`Prompting Model for Answer...`);

    // Generate Answer
    const answer = await model.generate(answerPrompt, {
        maxTokens: 300,
        temperature: 0.3, // Lower temperature for deterministic behavior as per blueprint
        stopSequences: ["Question:", "Facts:", "You are"]
    });

    log(`Generated: ${answer}`);

    // Render Result
    let html = `<h3>Answer</h3><p>${answer.replace(/\n/g, '<br>')}</p>`;
    html += `<h4>Sources</h4><ul>`;
    uniqueResults.slice(0, 5).forEach((r, i) => {
        html += `<li><a href="${r.url}" target="_blank">[${i + 1}] ${r.title}</a></li>`;
    });
    html += `</ul>`;

    resultArea.innerHTML = html;
    resultArea.classList.remove('hidden');
    setStatus(step3, false);
}

btnEl.addEventListener('click', runAgent);
inputEl.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') runAgent();
});

// Start
init();
