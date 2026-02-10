// Global variables
let selectedCases = [];
let availableData = [];

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
    loadAvailableData();
    initializeAnimations();
});

// Initialize app
function initializeApp() {
    // Set up navigation
    setupNavigation();
    
    // Set up event listeners
    setupEventListeners();
    
    // Initialize page
    showPage('home');
}

// Navigation setup
function setupNavigation() {
    const navLinks = document.querySelectorAll('.nav-link');
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetPage = this.getAttribute('href').substring(1);
            showPage(targetPage);
            
            // Update active nav link
            navLinks.forEach(l => l.classList.remove('active'));
            this.classList.add('active');
        });
    });
}

// Show specific page
function showPage(pageId) {
    const pages = document.querySelectorAll('.page');
    pages.forEach(page => {
        page.classList.remove('active');
    });
    
    const targetPage = document.getElementById(pageId);
    if (targetPage) {
        targetPage.classList.add('active');
    }
}

// Navigation functions
function goHome() {
    showPage('home');
    const navLinks = document.querySelectorAll('.nav-link');
    navLinks.forEach(l => l.classList.remove('active'));
    document.querySelector('[href="#home"]').classList.add('active');
}

function goToSegmentation() {
    showPage('segmentation');
    const navLinks = document.querySelectorAll('.nav-link');
    navLinks.forEach(l => l.classList.remove('active'));
    document.querySelector('[href="#segmentation"]').classList.add('active');
}

function goToAbout() {
    showPage('about');
    const navLinks = document.querySelectorAll('.nav-link');
    navLinks.forEach(l => l.classList.remove('active'));
    document.querySelector('[href="#about"]').classList.add('active');
}

// Initialize animations
function initializeAnimations() {
    // Animate counters
    const counters = document.querySelectorAll('.stat-number[data-target]');
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const target = parseInt(entry.target.getAttribute('data-target'));
                animateCounter(entry.target, target);
                observer.unobserve(entry.target);
            }
        });
    });
    
    counters.forEach(counter => observer.observe(counter));
    
    // Add hover effects to feature cards
    const featureCards = document.querySelectorAll('.feature-card');
    featureCards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-10px) scale(1.02)';
            this.style.transition = 'all 0.3s ease';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0) scale(1)';
        });
    });
    
    // Add click effects to buttons
    const buttons = document.querySelectorAll('.cta-button, .run-button');
    buttons.forEach(button => {
        button.addEventListener('click', function(e) {
            createRippleEffect(this, e);
        });
    });
    
    // Gentle parallax effect for particles
    window.addEventListener('scroll', function() {
        const scrolled = window.pageYOffset;
        const particles = document.querySelectorAll('.particle');
        
        particles.forEach((particle, index) => {
            const speed = 0.1 + (index * 0.05);
            particle.style.transform = `translateY(${scrolled * speed}px)`;
        });
    });
    
    // Simple fade-in animation for hero title
    const typingElement = document.querySelector('.typing-text');
    if (typingElement) {
        typingElement.style.opacity = '0';
        typingElement.style.transform = 'translateY(20px)';
        typingElement.style.transition = 'all 1s ease-out';
        
        setTimeout(() => {
            typingElement.style.opacity = '1';
            typingElement.style.transform = 'translateY(0)';
        }, 500);
    }
}

// Animated Counter
function animateCounter(element, target, duration = 2000) {
    let start = 0;
    const increment = target / (duration / 16);
    
    function updateCounter() {
        start += increment;
        if (start < target) {
            element.textContent = Math.floor(start);
            requestAnimationFrame(updateCounter);
        } else {
            element.textContent = target;
        }
    }
    
    updateCounter();
}

// Create ripple effect
function createRippleEffect(button, event) {
    const ripple = document.createElement('span');
    const rect = button.getBoundingClientRect();
    const size = Math.max(rect.width, rect.height);
    const x = event.clientX - rect.left - size / 2;
    const y = event.clientY - rect.top - size / 2;
    
    ripple.style.width = ripple.style.height = size + 'px';
    ripple.style.left = x + 'px';
    ripple.style.top = y + 'px';
    ripple.style.position = 'absolute';
    ripple.style.borderRadius = '50%';
    ripple.style.background = 'rgba(255, 255, 255, 0.3)';
    ripple.style.transform = 'scale(0)';
    ripple.style.animation = 'ripple 0.6s linear';
    ripple.style.pointerEvents = 'none';
    
    button.style.position = 'relative';
    button.style.overflow = 'hidden';
    button.appendChild(ripple);
    
    setTimeout(() => {
        ripple.remove();
    }, 600);
}

// Add CSS for ripple animation
const style = document.createElement('style');
style.textContent = `
    @keyframes ripple {
        to {
            transform: scale(4);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// Event listeners setup
function setupEventListeners() {
    // Data selection
    document.addEventListener('click', function(e) {
        if (e.target.closest('.data-item')) {
            const dataItem = e.target.closest('.data-item');
            const caseIndex = parseInt(dataItem.dataset.index);
            
            if (selectedCases.includes(caseIndex)) {
                selectedCases = selectedCases.filter(i => i !== caseIndex);
                dataItem.classList.remove('selected');
            } else {
                selectedCases.push(caseIndex);
                dataItem.classList.add('selected');
            }
        }
    });
    
    // Style selection
    document.addEventListener('change', function(e) {
        if (e.target.type === 'checkbox' && e.target.closest('.style-option')) {
            // Style selection is handled by CSS
        }
    });
}

// Load available data
async function loadAvailableData() {
    try {
        const response = await fetch('/api/data');
        const data = await response.json();
        
        if (data.success) {
            availableData = data.cases;
            renderDataGrid();
        } else {
            console.error('Failed to load data:', data.error);
        }
    } catch (error) {
        console.error('Error loading data:', error);
    }
}

// Render data grid
function renderDataGrid() {
    const dataGrid = document.getElementById('dataGrid');
    if (!dataGrid) return;
    
    dataGrid.innerHTML = '';
    
    availableData.forEach((caseData, index) => {
        const dataItem = document.createElement('div');
        dataItem.className = 'data-item';
        dataItem.dataset.index = index;
        
        dataItem.innerHTML = `
            <h4>${caseData.name}</h4>
            <p>Type: ${caseData.type}</p>
            <p>Status: ${caseData.status}</p>
        `;
        
        dataGrid.appendChild(dataItem);
    });
}

// Run segmentation
async function runSegmentation() {
    if (selectedCases.length === 0) {
        alert('Please select at least one case');
        return;
    }
    
    const selectedStyles = Array.from(document.querySelectorAll('.style-option input:checked'))
        .map(input => input.value);
    
    if (selectedStyles.length === 0) {
        alert('Please select at least one visualization style');
        return;
    }
    
    const sliceInput = document.getElementById('sliceInput');
    const slices = sliceInput.value || '50,60,70,80';
    
    // Show loading
    showLoading();
    
    try {
        const response = await fetch('/api/segmentation', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                cases: selectedCases,
                styles: selectedStyles,
                slices: slices
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            renderResults(data.results);
        } else {
            alert('Segmentation failed: ' + data.error);
        }
    } catch (error) {
        console.error('Error running segmentation:', error);
        alert('Error running segmentation: ' + error.message);
    } finally {
        hideLoading();
    }
}

// Show loading overlay
function showLoading() {
    const loadingOverlay = document.getElementById('loadingOverlay');
    if (loadingOverlay) {
        loadingOverlay.classList.add('active');
    }
}

// Hide loading overlay
function hideLoading() {
    const loadingOverlay = document.getElementById('loadingOverlay');
    if (loadingOverlay) {
        loadingOverlay.classList.remove('active');
    }
}

// Render results
function renderResults(results) {
    const resultsSection = document.getElementById('resultsSection');
    const resultsGrid = document.getElementById('resultsGrid');
    
    if (!resultsSection || !resultsGrid) return;
    
    resultsSection.style.display = 'block';
    resultsGrid.innerHTML = '';
    
    results.forEach(result => {
        const resultCard = document.createElement('div');
        resultCard.className = 'result-card';
        
        const statusClass = result.status === 'completed' ? 'status-completed' : 'status-failed';
        const imageFileName = result.image ? result.image.split('/').pop() : '';
        
        resultCard.innerHTML = `
            <div class="result-header">
                <h4>${result.case} - ${result.style.toUpperCase()} (Slice ${result.slice})</h4>
                ${result.status === 'completed' ? 
                    `<button class="view-full-btn" data-image="${imageFileName}" data-title="${result.case} - ${result.style.toUpperCase()} (Slice ${result.slice})">
                        <i class="fas fa-expand"></i>
                        <span>View</span>
                    </button>` : ''
                }
            </div>
            <div class="result-image">
                ${result.status === 'completed' ? 
                    `<img src="/api/image/${imageFileName}" alt="Segmentation Result" style="width: 100%; height: 100%; object-fit: cover; border-radius: 10px;">` :
                    `<div style="display: flex; align-items: center; justify-content: center; height: 100%; color: #ff4444;">
                        <i class="fas fa-exclamation-triangle"></i>
                        <span style="margin-left: 0.5rem;">${result.error || 'Image not found'}</span>
                    </div>`
                }
            </div>
            <div class="result-status">
                <span class="${statusClass}">Status: ${result.status}</span>
                <span style="color: #00d4ff;">Style: ${result.style}</span>
            </div>
        `;
        
        resultsGrid.appendChild(resultCard);
    });
    
    // Add event listeners to all view buttons
    const viewButtons = resultsGrid.querySelectorAll('.view-full-btn');
    viewButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            console.log('View button clicked!'); // Debug log
            const imageFileName = this.getAttribute('data-image');
            const title = this.getAttribute('data-title');
            console.log('Opening modal with:', imageFileName, title); // Debug log
            openImageModal(imageFileName, title);
        });
    });
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

// Open image modal
function openImageModal(imageFileName, title) {
    console.log('openImageModal called with:', imageFileName, title);
    const modal = document.getElementById('imageModal');
    const modalImage = document.getElementById('modalImage');
    const modalTitle = document.getElementById('modalTitle');
    
    console.log('Modal elements found:', {
        modal: !!modal,
        modalImage: !!modalImage,
        modalTitle: !!modalTitle
    });
    
    if (modal && modalImage && modalTitle) {
        modalImage.src = `/api/image/${imageFileName}`;
        modalTitle.textContent = title;
        modal.style.display = 'flex';
        document.body.style.overflow = 'hidden'; // Prevent background scrolling
        console.log('Modal opened successfully');
    } else {
        console.error('Modal elements not found!');
    }
}

// Close image modal
function closeImageModal() {
    const modal = document.getElementById('imageModal');
    if (modal) {
        modal.style.display = 'none';
        document.body.style.overflow = 'auto'; // Restore scrolling
    }
}

// Add modal event listeners
document.addEventListener('DOMContentLoaded', function() {
    // Close modal when clicking the X button
    const closeBtn = document.querySelector('.modal-close');
    if (closeBtn) {
        closeBtn.addEventListener('click', closeImageModal);
    }
    
    // Close modal when clicking the back button
    const backBtn = document.querySelector('.modal-back');
    if (backBtn) {
        backBtn.addEventListener('click', closeImageModal);
    }
    
    // Close modal when clicking outside the modal content
    const modal = document.getElementById('imageModal');
    if (modal) {
        modal.addEventListener('click', function(e) {
            if (e.target === modal) {
                closeImageModal();
            }
        });
    }
});

// Add keyboard support for modal
document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
        closeImageModal();
    }
});

// Add smooth scrolling
document.addEventListener('DOMContentLoaded', function() {
    // Smooth scroll for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
});

// Add intersection observer for animations
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
}, observerOptions);

// Observe elements for animation
document.addEventListener('DOMContentLoaded', function() {
    const animatedElements = document.querySelectorAll('.feature-card, .stat-card, .tech-stat');
    animatedElements.forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(30px)';
        el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(el);
    });
});