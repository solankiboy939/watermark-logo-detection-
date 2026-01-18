// Watermark Detector Pro - Frontend Application
class WatermarkDetector {
    constructor() {
        this.currentFile = null;
        this.isProcessing = false;
        this.apiBaseUrl = window.location.origin;
        
        this.initializeElements();
        this.bindEvents();
        this.checkApiHealth();
    }

    initializeElements() {
        // Upload elements
        this.uploadArea = document.getElementById('uploadArea');
        this.fileInput = document.getElementById('fileInput');
        this.fileInfo = document.getElementById('fileInfo');
        this.fileName = document.getElementById('fileName');
        this.fileSize = document.getElementById('fileSize');
        this.removeFileBtn = document.getElementById('removeFile');

        // Controls elements
        this.controlsSection = document.getElementById('controlsSection');
        this.confidenceSlider = document.getElementById('confidenceSlider');
        this.confidenceValue = document.getElementById('confidenceValue');
        this.detectBtn = document.getElementById('detectBtn');

        // Results elements
        this.resultsSection = document.getElementById('resultsSection');
        this.resultsStats = document.getElementById('resultsStats');
        this.originalImage = document.getElementById('originalImage');
        this.resultImage = document.getElementById('resultImage');
        this.detectionDetails = document.getElementById('detectionDetails');

        // Loading elements
        this.loadingOverlay = document.getElementById('loadingOverlay');
        this.progressFill = document.getElementById('progressFill');

        // Toast container
        this.toastContainer = document.getElementById('toastContainer');
    }

    bindEvents() {
        // Upload events
        this.uploadArea.addEventListener('click', () => this.fileInput.click());
        this.uploadArea.addEventListener('dragover', this.handleDragOver.bind(this));
        this.uploadArea.addEventListener('dragleave', this.handleDragLeave.bind(this));
        this.uploadArea.addEventListener('drop', this.handleDrop.bind(this));
        this.fileInput.addEventListener('change', this.handleFileSelect.bind(this));
        this.removeFileBtn.addEventListener('click', this.removeFile.bind(this));

        // Controls events
        this.confidenceSlider.addEventListener('input', this.updateConfidenceValue.bind(this));
        this.detectBtn.addEventListener('click', this.detectWatermarksHandler.bind(this));

        // Prevent default drag behaviors
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            document.addEventListener(eventName, this.preventDefaults, false);
        });
    }

    preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    handleDragOver(e) {
        this.uploadArea.classList.add('dragover');
    }

    handleDragLeave(e) {
        this.uploadArea.classList.remove('dragover');
    }

    handleDrop(e) {
        this.uploadArea.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.handleFile(files[0]);
        }
    }

    handleFileSelect(e) {
        const files = e.target.files;
        if (files.length > 0) {
            this.handleFile(files[0]);
        }
    }

    handleFile(file) {
        // Validate file type
        if (!file.type.startsWith('image/')) {
            this.showToast('Please select an image file (JPG, PNG, JPEG)', 'error');
            return;
        }

        // Validate file size (10MB limit)
        const maxSize = 10 * 1024 * 1024; // 10MB
        if (file.size > maxSize) {
            this.showToast('File size must be less than 10MB', 'error');
            return;
        }

        this.currentFile = file;
        this.displayFileInfo(file);
        this.showControls();
        this.hideResults();
    }

    displayFileInfo(file) {
        this.fileName.textContent = file.name;
        this.fileSize.textContent = this.formatFileSize(file.size);
        this.fileInfo.style.display = 'block';
        
        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            this.originalImage.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    removeFile() {
        this.currentFile = null;
        this.fileInput.value = '';
        this.fileInfo.style.display = 'none';
        this.hideControls();
        this.hideResults();
    }

    showControls() {
        this.controlsSection.style.display = 'block';
    }

    hideControls() {
        this.controlsSection.style.display = 'none';
    }

    showResults() {
        this.resultsSection.style.display = 'block';
    }

    hideResults() {
        this.resultsSection.style.display = 'none';
    }

    updateConfidenceValue() {
        const value = parseFloat(this.confidenceSlider.value);
        this.confidenceValue.textContent = value.toFixed(2);
    }

    async detectWatermarksHandler() {
        if (!this.currentFile || this.isProcessing) return;

        this.isProcessing = true;
        this.showLoading();
        this.detectBtn.disabled = true;

        try {
            const confidence = parseFloat(this.confidenceSlider.value);
            const result = await this.detectWatermarksAPI(this.currentFile, confidence);
            
            this.displayResults(result);
            this.showToast(`Detection complete! Found ${result.num_detections} watermark(s)`, 'success');
            
        } catch (error) {
            console.error('Detection error:', error);
            this.showToast('Detection failed. Please try again.', 'error');
        } finally {
            this.hideLoading();
            this.detectBtn.disabled = false;
            this.isProcessing = false;
        }
    }

    async detectWatermarksAPI(file, confidence) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('confidence', confidence.toString());

        // Simulate progress
        this.updateProgress(30);

        const response = await fetch(`${this.apiBaseUrl}/detect`, {
            method: 'POST',
            body: formData
        });

        this.updateProgress(80);

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Detection failed');
        }

        const result = await response.json();
        this.updateProgress(100);

        return result;
    }

    displayResults(result) {
        // Update stats
        this.resultsStats.innerHTML = `
            <div class="stat-item">
                <div class="stat-value">${result.num_detections}</div>
                <div class="stat-label">Detections</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">${(result.confidence_used * 100).toFixed(0)}%</div>
                <div class="stat-label">Confidence</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">${result.image_size.width}×${result.image_size.height}</div>
                <div class="stat-label">Resolution</div>
            </div>
        `;

        // Update images
        this.originalImage.src = result.original_image;
        this.resultImage.src = result.result_image;

        // Update detection details
        this.displayDetectionDetails(result.detections);

        // Show results section
        this.showResults();

        // Scroll to results
        this.resultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    displayDetectionDetails(detections) {
        if (detections.length === 0) {
            this.detectionDetails.innerHTML = `
                <div style="text-align: center; padding: 2rem; color: var(--gray-600);">
                    <i class="fas fa-check-circle" style="font-size: 3rem; color: var(--success); margin-bottom: 1rem;"></i>
                    <h4>No Watermarks Detected</h4>
                    <p>The image appears to be clean with no watermarks found.</p>
                </div>
            `;
            return;
        }

        const detectionHTML = detections.map((detection, index) => {
            const confidenceClass = this.getConfidenceClass(detection.confidence);
            const confidencePercent = (detection.confidence * 100).toFixed(1);
            
            return `
                <div class="detection-item">
                    <div class="detection-info">
                        <div class="detection-icon">${index + 1}</div>
                        <div class="detection-text">
                            <h5>Watermark Detection #${index + 1}</h5>
                            <p>Position: (${Math.round(detection.bbox[0])}, ${Math.round(detection.bbox[1])}) - (${Math.round(detection.bbox[2])}, ${Math.round(detection.bbox[3])})</p>
                        </div>
                    </div>
                    <div class="confidence-badge ${confidenceClass}">
                        ${confidencePercent}%
                    </div>
                </div>
            `;
        }).join('');

        this.detectionDetails.innerHTML = `
            <h4 style="margin-bottom: 1rem; color: var(--gray-800);">
                <i class="fas fa-list"></i> Detection Details
            </h4>
            ${detectionHTML}
        `;
    }

    getConfidenceClass(confidence) {
        if (confidence >= 0.8) return 'confidence-high';
        if (confidence >= 0.5) return 'confidence-medium';
        return 'confidence-low';
    }

    showLoading() {
        this.loadingOverlay.style.display = 'flex';
        this.updateProgress(0);
    }

    hideLoading() {
        this.loadingOverlay.style.display = 'none';
    }

    updateProgress(percent) {
        this.progressFill.style.width = `${percent}%`;
    }

    showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        
        const icon = this.getToastIcon(type);
        toast.innerHTML = `
            <div style="display: flex; align-items: center; gap: 0.75rem;">
                <i class="${icon}" style="font-size: 1.25rem;"></i>
                <div>
                    <div style="font-weight: 600; margin-bottom: 0.25rem;">${this.getToastTitle(type)}</div>
                    <div style="font-size: 0.875rem; opacity: 0.8;">${message}</div>
                </div>
            </div>
        `;

        this.toastContainer.appendChild(toast);

        // Auto remove after 5 seconds
        setTimeout(() => {
            if (toast.parentNode) {
                toast.style.animation = 'slideOut 0.3s ease-in forwards';
                setTimeout(() => {
                    if (toast.parentNode) {
                        this.toastContainer.removeChild(toast);
                    }
                }, 300);
            }
        }, 5000);

        // Add click to dismiss
        toast.addEventListener('click', () => {
            if (toast.parentNode) {
                this.toastContainer.removeChild(toast);
            }
        });
    }

    getToastIcon(type) {
        const icons = {
            success: 'fas fa-check-circle',
            error: 'fas fa-exclamation-circle',
            warning: 'fas fa-exclamation-triangle',
            info: 'fas fa-info-circle'
        };
        return icons[type] || icons.info;
    }

    getToastTitle(type) {
        const titles = {
            success: 'Success',
            error: 'Error',
            warning: 'Warning',
            info: 'Info'
        };
        return titles[type] || titles.info;
    }

    async checkApiHealth() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/health`);
            const data = await response.json();
            
            if (data.status === 'healthy' && data.model_loaded) {
                this.showToast('Application ready for watermark detection', 'success');
            } else {
                this.showToast('Model not loaded. Please check server logs.', 'warning');
            }
        } catch (error) {
            console.error('Health check failed:', error);
            this.showToast('Unable to connect to detection service', 'error');
        }
    }
}

// Global functions for footer links
function showAbout() {
    const app = window.watermarkDetector;
    app.showToast('Watermark Detector Pro v1.0 - Powered by YOLOv8 AI technology', 'info');
}

function showHelp() {
    const app = window.watermarkDetector;
    app.showToast('Upload an image, adjust confidence threshold, and click detect to find watermarks', 'info');
}

// Add slideOut animation
const style = document.createElement('style');
style.textContent = `
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.watermarkDetector = new WatermarkDetector();
});

// Handle page visibility changes
document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') {
        // Refresh health check when page becomes visible
        if (window.watermarkDetector) {
            window.watermarkDetector.checkApiHealth();
        }
    }
});