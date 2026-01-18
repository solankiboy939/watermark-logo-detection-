// Watermark Detector Pro - TypeScript Frontend Application

interface DetectionResult {
    success: boolean;
    num_detections: number;
    detections: Detection[];
    original_image: string;
    result_image: string;
    confidence_used: number;
    image_size: {
        width: number;
        height: number;
    };
}

interface Detection {
    id: number;
    confidence: number;
    class_id: number;
    bbox: [number, number, number, number]; // [x1, y1, x2, y2]
}

interface HealthResponse {
    status: string;
    model_loaded: boolean;
    message: string;
}

interface ModelInfo {
    model_loaded: boolean;
    model_type?: string;
    model_file?: string;
    classes?: Record<string, string>;
    input_size?: number;
    error?: string;
}

type ToastType = 'success' | 'error' | 'warning' | 'info';

class WatermarkDetectorTS {
    private currentFile: File | null = null;
    private isProcessing: boolean = false;
    private readonly apiBaseUrl: string;
    
    // DOM Elements
    private readonly uploadArea: HTMLElement;
    private readonly fileInput: HTMLInputElement;
    private readonly fileInfo: HTMLElement;
    private readonly fileName: HTMLElement;
    private readonly fileSize: HTMLElement;
    private readonly removeFileBtn: HTMLButtonElement;
    
    private readonly controlsSection: HTMLElement;
    private readonly confidenceSlider: HTMLInputElement;
    private readonly confidenceValue: HTMLElement;
    private readonly detectBtn: HTMLButtonElement;
    
    private readonly resultsSection: HTMLElement;
    private readonly resultsStats: HTMLElement;
    private readonly originalImage: HTMLImageElement;
    private readonly resultImage: HTMLImageElement;
    private readonly detectionDetails: HTMLElement;
    
    private readonly loadingOverlay: HTMLElement;
    private readonly progressFill: HTMLElement;
    private readonly toastContainer: HTMLElement;

    constructor() {
        this.apiBaseUrl = window.location.origin;
        
        // Initialize DOM elements with type checking
        this.uploadArea = this.getElement('uploadArea');
        this.fileInput = this.getElement('fileInput') as HTMLInputElement;
        this.fileInfo = this.getElement('fileInfo');
        this.fileName = this.getElement('fileName');
        this.fileSize = this.getElement('fileSize');
        this.removeFileBtn = this.getElement('removeFile') as HTMLButtonElement;
        
        this.controlsSection = this.getElement('controlsSection');
        this.confidenceSlider = this.getElement('confidenceSlider') as HTMLInputElement;
        this.confidenceValue = this.getElement('confidenceValue');
        this.detectBtn = this.getElement('detectBtn') as HTMLButtonElement;
        
        this.resultsSection = this.getElement('resultsSection');
        this.resultsStats = this.getElement('resultsStats');
        this.originalImage = this.getElement('originalImage') as HTMLImageElement;
        this.resultImage = this.getElement('resultImage') as HTMLImageElement;
        this.detectionDetails = this.getElement('detectionDetails');
        
        this.loadingOverlay = this.getElement('loadingOverlay');
        this.progressFill = this.getElement('progressFill');
        this.toastContainer = this.getElement('toastContainer');
        
        this.bindEvents();
        this.checkApiHealth();
    }

    private getElement(id: string): HTMLElement {
        const element = document.getElementById(id);
        if (!element) {
            throw new Error(`Element with id '${id}' not found`);
        }
        return element;
    }

    private bindEvents(): void {
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

    private preventDefaults(e: Event): void {
        e.preventDefault();
        e.stopPropagation();
    }

    private handleDragOver(e: DragEvent): void {
        e.preventDefault();
        this.uploadArea.classList.add('dragover');
    }

    private handleDragLeave(e: DragEvent): void {
        e.preventDefault();
        this.uploadArea.classList.remove('dragover');
    }

    private handleDrop(e: DragEvent): void {
        e.preventDefault();
        this.uploadArea.classList.remove('dragover');
        const files = e.dataTransfer?.files;
        if (files && files.length > 0) {
            this.handleFile(files[0]);
        }
    }

    private handleFileSelect(e: Event): void {
        const target = e.target as HTMLInputElement;
        const files = target.files;
        if (files && files.length > 0) {
            this.handleFile(files[0]);
        }
    }

    private handleFile(file: File): void {
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

    private displayFileInfo(file: File): void {
        this.fileName.textContent = file.name;
        this.fileSize.textContent = this.formatFileSize(file.size);
        this.fileInfo.style.display = 'block';
        
        // Show preview
        const reader = new FileReader();
        reader.onload = (e: ProgressEvent<FileReader>) => {
            if (e.target?.result) {
                this.originalImage.src = e.target.result as string;
            }
        };
        reader.readAsDataURL(file);
    }

    private formatFileSize(bytes: number): string {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    private removeFile(): void {
        this.currentFile = null;
        this.fileInput.value = '';
        this.fileInfo.style.display = 'none';
        this.hideControls();
        this.hideResults();
    }

    private showControls(): void {
        this.controlsSection.style.display = 'block';
    }

    private hideControls(): void {
        this.controlsSection.style.display = 'none';
    }

    private showResults(): void {
        this.resultsSection.style.display = 'block';
    }

    private hideResults(): void {
        this.resultsSection.style.display = 'none';
    }

    private updateConfidenceValue(): void {
        const value = parseFloat(this.confidenceSlider.value);
        this.confidenceValue.textContent = value.toFixed(2);
    }

    private async detectWatermarksHandler(): Promise<void> {
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
            const errorMessage = error instanceof Error ? error.message : 'Detection failed. Please try again.';
            this.showToast(errorMessage, 'error');
        } finally {
            this.hideLoading();
            this.detectBtn.disabled = false;
            this.isProcessing = false;
        }
    }

    private async detectWatermarksAPI(file: File, confidence: number): Promise<DetectionResult> {
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

        const result: DetectionResult = await response.json();
        this.updateProgress(100);

        return result;
    }

    private displayResults(result: DetectionResult): void {
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

    private displayDetectionDetails(detections: Detection[]): void {
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

    private getConfidenceClass(confidence: number): string {
        if (confidence >= 0.8) return 'confidence-high';
        if (confidence >= 0.5) return 'confidence-medium';
        return 'confidence-low';
    }

    private showLoading(): void {
        this.loadingOverlay.style.display = 'flex';
        this.updateProgress(0);
    }

    private hideLoading(): void {
        this.loadingOverlay.style.display = 'none';
    }

    private updateProgress(percent: number): void {
        this.progressFill.style.width = `${percent}%`;
    }

    private showToast(message: string, type: ToastType = 'info'): void {
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

    private getToastIcon(type: ToastType): string {
        const icons: Record<ToastType, string> = {
            success: 'fas fa-check-circle',
            error: 'fas fa-exclamation-circle',
            warning: 'fas fa-exclamation-triangle',
            info: 'fas fa-info-circle'
        };
        return icons[type];
    }

    private getToastTitle(type: ToastType): string {
        const titles: Record<ToastType, string> = {
            success: 'Success',
            error: 'Error',
            warning: 'Warning',
            info: 'Info'
        };
        return titles[type];
    }

    private async checkApiHealth(): Promise<void> {
        try {
            const response = await fetch(`${this.apiBaseUrl}/health`);
            const data: HealthResponse = await response.json();
            
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

    public async getModelInfo(): Promise<ModelInfo> {
        try {
            const response = await fetch(`${this.apiBaseUrl}/model-info`);
            return await response.json();
        } catch (error) {
            console.error('Failed to get model info:', error);
            return { model_loaded: false, error: 'Failed to fetch model info' };
        }
    }
}

// Global functions for footer links
function showAbout(): void {
    const app = (window as any).watermarkDetector as WatermarkDetectorTS;
    app.showToast('Watermark Detector Pro v1.0 - Powered by YOLOv8 AI technology', 'info');
}

function showHelp(): void {
    const app = (window as any).watermarkDetector as WatermarkDetectorTS;
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
    (window as any).watermarkDetector = new WatermarkDetectorTS();
});

// Handle page visibility changes
document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') {
        // Refresh health check when page becomes visible
        const app = (window as any).watermarkDetector as WatermarkDetectorTS;
        if (app) {
            app.checkApiHealth();
        }
    }
});