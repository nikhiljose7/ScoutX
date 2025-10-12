// Common utility functions

// Format number to 2 decimal places
function formatNumber(num) {
    return parseFloat(num).toFixed(2);
}

// Show loading state
function showLoading(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.classList.remove('hidden');
    }
}

// Hide loading state
function hideLoading(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.classList.add('hidden');
    }
}

// Show error message
function showError(elementId, message) {
    const element = document.getElementById(elementId);
    if (element) {
        element.textContent = message;
        element.classList.remove('hidden');
    }
}

// Hide error message
function hideError(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.classList.add('hidden');
    }
}

// Format league name
function formatLeague(comp) {
    const leagues = {
        'eng Premier League': 'Premier League',
        'es La Liga': 'La Liga',
        'it Serie A': 'Serie A',
        'fr Ligue 1': 'Ligue 1',
        'de Bundesliga': 'Bundesliga'
    };
    return leagues[comp] || comp;
}

// Make API call
async function apiCall(url, method = 'GET', data = null) {
    const options = {
        method: method,
        headers: {
            'Content-Type': 'application/json'
        }
    };
    
    if (data && method !== 'GET') {
        options.body = JSON.stringify(data);
    }
    
    try {
        const response = await fetch(url, options);
        return await response.json();
    } catch (error) {
        console.error('API Error:', error);
        throw error;
    }
}

// Export functions for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        formatNumber,
        showLoading,
        hideLoading,
        showError,
        hideError,
        formatLeague,
        apiCall
    };
}