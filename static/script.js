/**
 * =============================================================================
 * Fake News Detection System - Frontend JavaScript (script.js)
 * =============================================================================
 * Connects the web user interface to the Flask '/predict' backend API.
 * Handles client-side validation, loading animations, asynchronous fetch requests,
 * and dynamic DOM manipulation without full page reloads.
 * =============================================================================
 */

document.addEventListener("DOMContentLoaded", () => {
    // -------------------------------------------------------------------------
    // 1. DOM Element Selectors
    // -------------------------------------------------------------------------
    const newsInput = document.getElementById("newsInput");
    const charCount = document.getElementById("charCount");
    const analyzeBtn = document.getElementById("analyzeBtn");
    
    // Error notification elements
    const errorBox = document.getElementById("errorBox");
    const errorMessage = document.getElementById("errorMessage");

    // Result card elements
    const resultsCard = document.getElementById("resultsCard");
    const verdictBanner = document.getElementById("verdictBanner");
    const verdictIcon = document.getElementById("verdictIcon");
    const verdictTitle = document.getElementById("verdictTitle");
    const confidenceValue = document.getElementById("confidenceValue");
    const timestamp = document.getElementById("timestamp");

    // Probability meters and score cards
    const meterReal = document.getElementById("meterReal");
    const meterFake = document.getElementById("meterFake");
    const realProbScore = document.getElementById("realProbScore");
    const fakeProbScore = document.getElementById("fakeProbScore");
    const snippetText = document.getElementById("snippetText");

    // Demo sample buttons
    const sampleRealBtn = document.getElementById("sampleRealBtn");
    const sampleFakeBtn = document.getElementById("sampleFakeBtn");
    const clearBtn = document.getElementById("clearBtn");

    // -------------------------------------------------------------------------
    // 2. Sample Data for Viva / College Demo Presentations
    // -------------------------------------------------------------------------
    const SAMPLES = {
        real: "U.S. Secretary of State John Kerry said Monday that he will stop in Paris later this week, amid criticism that no top American official attended Sunday's unity march following the terror attacks. Kerry will hold diplomatic talks on bilateral security cooperation.",
        fake: "SHOCKING: Secret leaked military documents reveal government officials are concealing contact with extraterrestrial alien craft deep beneath top-secret underground bunkers! Share before it gets deleted by mainstream media!"
    };

    // -------------------------------------------------------------------------
    // 3. Helper Functions: UI State Management
    // -------------------------------------------------------------------------
    
    // Shows the error banner with a custom message
    function showError(message) {
        errorMessage.textContent = message;
        errorBox.style.display = "flex";
        resultsCard.style.display = "none";
    }

    // Hides the error banner
    function hideError() {
        errorBox.style.display = "none";
        errorMessage.textContent = "";
    }

    // Sets the button into a loading state (spinner + disabled)
    function setLoading(isLoading) {
        if (isLoading) {
            analyzeBtn.disabled = true;
            analyzeBtn.innerHTML = `
                <span class="spinner"></span>
                <span class="btn-text">Analyzing News...</span>
            `;
            hideError();
        } else {
            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = `
                <span class="btn-icon">🔍</span>
                <span class="btn-text">Analyze News</span>
            `;
        }
    }

    // Updates live character count display as user types
    function updateCharCounter() {
        const count = newsInput.value.length;
        charCount.textContent = `${count} character${count === 1 ? "" : "s"}`;
    }

    // -------------------------------------------------------------------------
    // 4. Prediction Request Logic (Fetch API)
    // -------------------------------------------------------------------------
    async function handleNewsAnalysis() {
        const rawText = newsInput.value.trim();

        // Client-side Input Validation
        if (!rawText) {
            showError("Please enter or paste a news headline or article body before analyzing.");
            newsInput.focus();
            return;
        }

        if (rawText.length < 10) {
            showError("Text is too short for reliable evaluation. Please provide at least 10 characters.");
            newsInput.focus();
            return;
        }

        // Enter Loading State
        setLoading(true);

        try {
            // Send asynchronous POST request to Flask '/predict' API endpoint
            const response = await fetch("/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    news_text: rawText
                })
            });

            const data = await response.json();

            // If backend returned an error (e.g. 400 Bad Request)
            if (!response.ok || !data.success) {
                showError(data.error || "An error occurred while evaluating the news.");
                setLoading(false);
                return;
            }

            // Successfully received prediction! Render result
            displayResults(data, rawText);

        } catch (networkError) {
            console.error("Network or parsing error:", networkError);
            showError("Could not reach the server. Make sure your Flask backend (app.py) is running.");
        } finally {
            // Restore button to ready state
            setLoading(false);
        }
    }

    // -------------------------------------------------------------------------
    // 5. Render Prediction Results on UI
    // -------------------------------------------------------------------------
    function displayResults(data, originalText) {
        hideError();

        const verdict = data.verdict; // 'REAL' or 'FAKE'
        const confidence = data.confidence;
        const realProb = data.probabilities.REAL;
        const fakeProb = data.probabilities.FAKE;

        // Reset previous banner classes
        verdictBanner.classList.remove("real-news", "fake-news");

        if (verdict === "REAL") {
            verdictBanner.classList.add("real-news");
            verdictIcon.textContent = "✅";
            verdictTitle.textContent = "VERIFIED REAL NEWS";
        } else {
            verdictBanner.classList.add("fake-news");
            verdictIcon.textContent = "⚠️";
            verdictTitle.textContent = "SUSPICIOUS / FAKE NEWS";
        }

        // Update Confidence Score
        confidenceValue.textContent = `${confidence.toFixed(1)}%`;

        // Update Probability Bars
        meterReal.style.width = `${realProb}%`;
        meterFake.style.width = `${fakeProb}%`;

        // Update Individual Metric Labels
        realProbScore.textContent = `${realProb.toFixed(1)}%`;
        fakeProbScore.textContent = `${fakeProb.toFixed(1)}%`;

        // Set Text Snippet Preview
        snippetText.textContent = `"${originalText.length > 200 ? originalText.substring(0, 200) + '...' : originalText}"`;

        // Timestamp
        timestamp.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

        // Display results card smoothly
        resultsCard.style.display = "block";
        resultsCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }

    // -------------------------------------------------------------------------
    // 6. Event Listeners
    // -------------------------------------------------------------------------
    
    // Analyze button click
    analyzeBtn.addEventListener("click", handleNewsAnalysis);

    // Ctrl+Enter or Cmd+Enter keyboard shortcut in textarea to analyze
    newsInput.addEventListener("keydown", (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
            e.preventDefault();
            handleNewsAnalysis();
        }
    });

    // Character counter listener
    newsInput.addEventListener("input", updateCharCounter);

    // Load Sample Real News
    sampleRealBtn.addEventListener("click", () => {
        newsInput.value = SAMPLES.real;
        updateCharCounter();
        hideError();
        resultsCard.style.display = "none";
        newsInput.focus();
    });

    // Load Sample Fake News
    sampleFakeBtn.addEventListener("click", () => {
        newsInput.value = SAMPLES.fake;
        updateCharCounter();
        hideError();
        resultsCard.style.display = "none";
        newsInput.focus();
    });

    // Clear input
    clearBtn.addEventListener("click", () => {
        newsInput.value = "";
        updateCharCounter();
        hideError();
        resultsCard.style.display = "none";
        newsInput.focus();
    });
});
