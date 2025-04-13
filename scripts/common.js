// Navigation Functions
function openNewTab(url) {
    window.open(url, "_self");
}

// API Fetch Wrapper
async function fetchData(url, options = {}) {
    try {
        const response = await fetch(url, options);
        if (!response.ok) throw new Error(`HTTP ${response.status} - ${response.statusText}`);
        return await response.json();
    } catch (error) {
        console.error("Error fetching data:", error);
        alert("An error occurred while fetching data.");
        return null;
    }
}

async function fetchPrediction(ticker) {
    try {
        const response = await fetch('http://localhost:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ ticker }),
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Failed to fetch predictions');
        }

        const data = await response.json();
        return data; 
    } catch (error) {
        console.error('Error fetching prediction:', error);
        return { error: error.message };
    }
}

// Event Listener for Footer Icon
document.querySelectorAll('.footer-icon').forEach(icon => {
    icon.addEventListener('click', () => openNewTab('index.html'));
});
