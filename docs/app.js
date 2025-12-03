// FactBullet App Logic - Professional Edition

const SCRAPER_URL = 'http://localhost:3000/search';
const model = new BulletModel();

// UI Elements
const header = document.getElementById('app-header');
const inputEl = document.getElementById('user-query');
const btnEl = document.getElementById('search-btn');
const statusBar = document.getElementById('status-bar');
const mainContent = document.getElementById('main-content');
const sourcesList = document.getElementById('sources-list');
const summaryText = document.getElementById('summary-text');
const keyPointsList = document.getElementById('key-points');

const steps = {
    1: document.getElementById('step-1'),
    2: document.getElementById('step-2'),
    3: document.getElementById('step-3')
};

// System Prompt from Blueprint
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

async function init() {
    console.log('Initializing Model...');
    const success = await model.load('model_weights.bin', 'tokenizer.json');
    if (success) {
        console.log('Model Loaded!');
        inputEl.placeholder = "Ask anything... (Model Ready)";
    } else {
        alert('Failed to load model. Check console.');
    }
}

function setStatus(step, active) {
    Object.values(steps).forEach(s => s.classList.remove('active'));
    if (active && steps[step]) {
        steps[step].classList.add('active');
    }
}

function formatCitations(text) {
    // Replace [1], [2] with <sup class="citation">1</sup>
    return text.replace(/\[(\d+)\]/g, '<a href="#source-$1" class="citation">$1</a>');
}

function renderSources(results) {
    sourcesList.innerHTML = '';
    results.slice(0, 5).forEach((r, i) => {
        const id = i + 1;
        const card = document.createElement('a');
        card.className = 'source-card';
        card.href = r.url;
        card.target = '_blank';
        card.id = `source-${id}`;

        // Extract domain
        let domain = '';
        try { domain = new URL(r.url).hostname; } catch (e) { }

        card.innerHTML = `
            <div class="source-title">${r.title}</div>
            <div class="source-domain">${id}. ${domain}</div>
        `;
        sourcesList.appendChild(card);
    });
}

function parseModelOutput(output) {
    // Simple tag extraction
    const summaryMatch = output.match(/<summary>([\s\S]*?)<\/summary>/i);
    const keyPointsMatch = output.match(/<key_points>([\s\S]*?)<\/key_points>/i);

    let summary = summaryMatch ? summaryMatch[1].trim() : output; // Fallback to raw output if no tags
    let keyPoints = keyPointsMatch ? keyPointsMatch[1].trim() : "";

    // If fallback, try to split by newlines or bullets
    if (!keyPointsMatch && summary.includes('- ')) {
        const parts = summary.split('- ');
        summary = parts[0];
        keyPoints = parts.slice(1).map(p => '- ' + p).join('\n');
    }

    return { summary, keyPoints };
}

async function performSearch(query) {
    try {
        const response = await fetch(`${SCRAPER_URL}?q=${encodeURIComponent(query)}`);
        if (!response.ok) throw new Error('Scraper unreachable');
        const data = await response.json();
        return data.results || [];
    } catch (e) {
        console.error(e);
        return [];
    }
}

async function runAgent() {
    const userQuery = inputEl.value.trim();
    if (!userQuery) return;

    // UI Transition
    header.classList.add('compact');
    statusBar.classList.remove('hidden');
    mainContent.classList.add('hidden');

    setStatus(1, true); // Thinking

    // Step 1: Generate Queries
    console.log(`User Query: ${userQuery}`);

    // For speed/demo, we might skip complex query generation if the model is slow
    // But let's try a simple one-shot
    let queries = [userQuery];

    // Step 2: Search
    setStatus(2, true); // Searching

    let allResults = [];
    for (const q of queries) {
        const res = await performSearch(q);
        allResults = [...allResults, ...res];
    }

    // Deduplicate
    const uniqueResults = [];
    const seenUrls = new Set();
    for (const r of allResults) {
        if (!seenUrls.has(r.url)) {
            seenUrls.add(r.url);
            uniqueResults.push(r);
        }
    }

    if (uniqueResults.length === 0) {
        summaryText.innerHTML = "No results found. Please ensure the local scraper is running.";
        mainContent.classList.remove('hidden');
        return;
    }

    renderSources(uniqueResults);

    // Step 3: Synthesize
    setStatus(3, true); // Synthesizing

    // Prepare context
    let context = "";
    uniqueResults.slice(0, 5).forEach((r, i) => {
        context += `[${i + 1}] ${r.snippet}\n`;
    });

    // Strict format prompt
    const answerPrompt = `${SYSTEM_PROMPT}

Context:
${context}

User Question: ${userQuery}

Output Format:
<answer>
  <summary>
    Direct answer sentences.
  </summary>
  <key_points>
    - Point 1 [1]
    - Point 2 [2]
  </key_points>
</answer>

Generate Answer:`;

    const answer = await model.generate(answerPrompt, {
        maxTokens: 400,
        temperature: 0.3,
        stopSequences: ["</answer>", "User Question:"]
    });

    console.log("Raw Answer:", answer);
    const { summary, keyPoints } = parseModelOutput(answer);

    // Render
    summaryText.innerHTML = formatCitations(summary);

    keyPointsList.innerHTML = '';
    if (keyPoints) {
        const points = keyPoints.split('\n').filter(line => line.trim().length > 0);
        points.forEach(point => {
            const li = document.createElement('li');
            // Remove leading dash/bullet if present
            const cleanPoint = point.replace(/^[-•*]\s*/, '');
            li.innerHTML = formatCitations(cleanPoint);
            keyPointsList.appendChild(li);
        });
    }

    mainContent.classList.remove('hidden');
    setStatus(3, false); // Done
}

btnEl.addEventListener('click', runAgent);
inputEl.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') runAgent();
});

init();
