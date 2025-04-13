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

// Event Listener for Footer Icon
document.querySelectorAll('.footer-icon').forEach(icon => {
    icon.addEventListener('click', () => openNewTab('index.html'));
});
