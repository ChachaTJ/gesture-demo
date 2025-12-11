/**
 * 👁️ Gaze Controller Module
 * Provides eye-tracking based control for web pages using WebGazer.js
 * 
 * Features:
 * - 9-point calibration
 * - Gaze events (gazeEnter, gazeLeave, gazeDwell)
 * - Dwell-to-click (1.5s)
 * - Easy integration API
 * 
 * Usage:
 *   <script src="https://webgazer.cs.brown.edu/webgazer.js"></script>
 *   <script src="gaze_controller.js"></script>
 *   <script>
 *     GazeController.init();
 *     GazeController.on('gazeDwell', (el) => el.click());
 *   </script>
 */

const GazeController = (function () {
    'use strict';

    // ============ Configuration ============
    const CONFIG = {
        DWELL_TIME: 1500,           // 1.5 seconds to trigger click
        GAZE_TOLERANCE: 60,         // Pixel tolerance for gaze detection (increased for accuracy)
        CALIBRATION_POINTS: 13,     // 13 points for better coverage (research-backed)
        CALIBRATION_CLICKS: 8,      // More clicks per point for accuracy
        SHOW_PREDICTION: true,      // Show gaze dot on screen
        SHOW_VIDEO: false,          // Show webcam feed
        USE_KALMAN_FILTER: true,    // Smooth predictions with Kalman filter
        VALIDATION_ENABLED: true,   // Show accuracy test after calibration
        REGRESSION_TYPE: 'ridge'    // 'ridge' or 'weightedRidge' for better accuracy
    };

    // ============ State ============
    let isInitialized = false;
    let isCalibrated = false;
    let activeElement = null;
    let dwellTimer = null;
    let dwellProgress = 0;
    let gazeTargets = [];
    let eventListeners = {
        gazeEnter: [],
        gazeLeave: [],
        gazeDwell: [],
        calibrationComplete: []
    };

    // ============ Calibration UI ============
    const CALIBRATION_HTML = `
    <div id="gaze-calibration-overlay" style="
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background: rgba(0, 0, 0, 0.95);
        z-index: 999999;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        font-family: -apple-system, BlinkMacSystemFont, sans-serif;
        color: white;
    ">
        <div id="calibration-header" style="text-align: center; margin-bottom: 40px;">
            <h1 style="font-size: 28px; margin-bottom: 10px;">👁️ Eye Calibration</h1>
            <p style="color: rgba(255,255,255,0.7); font-size: 16px;">
                Look at each dot and click it ${CONFIG.CALIBRATION_CLICKS} times
            </p>
            <p id="calibration-progress" style="color: #7bed9f; font-size: 14px; margin-top: 10px;">
                Point 1 of ${CONFIG.CALIBRATION_POINTS}
            </p>
        </div>
        <div id="calibration-area" style="
            position: relative;
            width: 80vw;
            height: 60vh;
        "></div>
        <button id="calibration-skip" style="
            margin-top: 30px;
            padding: 10px 20px;
            background: transparent;
            border: 1px solid rgba(255,255,255,0.3);
            color: rgba(255,255,255,0.5);
            border-radius: 20px;
            cursor: pointer;
            font-size: 14px;
        ">Skip Calibration</button>
    </div>
    `;

    // Calibration point positions (relative %) - 13 points for maximum accuracy
    const CALIBRATION_POSITIONS = [
        // Corners
        { x: 5, y: 5 },    // Top-left
        { x: 95, y: 5 },   // Top-right
        { x: 5, y: 95 },   // Bottom-left
        { x: 95, y: 95 },  // Bottom-right
        // Edges
        { x: 50, y: 5 },   // Top-center
        { x: 50, y: 95 },  // Bottom-center
        { x: 5, y: 50 },   // Left-center
        { x: 95, y: 50 },  // Right-center
        // Inner points (for precision)
        { x: 25, y: 25 },  // Inner top-left
        { x: 75, y: 25 },  // Inner top-right
        { x: 25, y: 75 },  // Inner bottom-left
        { x: 75, y: 75 },  // Inner bottom-right
        // Center (most important)
        { x: 50, y: 50 }   // Center
    ];

    // ============ Core Functions ============

    /**
     * Initialize the Gaze Controller
     * @param {Object} options - Configuration options
     */
    async function init(options = {}) {
        if (isInitialized) {
            console.warn('GazeController already initialized');
            return;
        }

        // Merge options
        Object.assign(CONFIG, options);

        // Check for WebGazer
        if (typeof webgazer === 'undefined') {
            console.error('WebGazer.js not loaded. Include it before gaze_controller.js');
            return;
        }

        try {
            // Configure WebGazer with improved settings
            webgazer.setRegression(CONFIG.REGRESSION_TYPE)
                .setTracker('TFFacemesh')
                .showVideoPreview(CONFIG.SHOW_VIDEO)
                .showPredictionPoints(CONFIG.SHOW_PREDICTION);

            // Apply Kalman filter for smoother, more accurate predictions
            if (CONFIG.USE_KALMAN_FILTER) {
                webgazer.applyKalmanFilter(true);
            }

            // Start WebGazer
            await webgazer.begin();

            // Set up gaze listener
            webgazer.setGazeListener((data, elapsedTime) => {
                if (data == null) return;
                handleGaze(data.x, data.y);
            });

            isInitialized = true;
            console.log('✓ GazeController initialized');

            // Style the prediction dot
            stylePredictionDot();

            // Auto-start calibration if not calibrated
            if (!isCalibrated && !localStorage.getItem('gazeCalibrated')) {
                setTimeout(() => showCalibration(), 1000);
            }

        } catch (err) {
            console.error('GazeController init failed:', err);
        }
    }

    /**
     * Show calibration UI
     */
    function showCalibration() {
        // Inject calibration overlay
        const overlay = document.createElement('div');
        overlay.innerHTML = CALIBRATION_HTML;
        document.body.appendChild(overlay.firstElementChild);

        const area = document.getElementById('calibration-area');
        const progress = document.getElementById('calibration-progress');
        const skipBtn = document.getElementById('calibration-skip');

        let currentPoint = 0;
        let clicksOnPoint = 0;

        function createPoint(index) {
            area.innerHTML = '';
            const pos = CALIBRATION_POSITIONS[index];

            const point = document.createElement('div');
            point.id = 'cal-point';
            point.style.cssText = `
                position: absolute;
                left: ${pos.x}%;
                top: ${pos.y}%;
                width: 40px;
                height: 40px;
                margin: -20px 0 0 -20px;
                background: #7bed9f;
                border-radius: 50%;
                cursor: pointer;
                transition: transform 0.2s, background 0.2s;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 14px;
                font-weight: bold;
                color: #1e3c72;
            `;
            point.textContent = CONFIG.CALIBRATION_CLICKS - clicksOnPoint;

            point.addEventListener('click', () => {
                clicksOnPoint++;
                point.style.transform = 'scale(0.8)';
                setTimeout(() => point.style.transform = 'scale(1)', 100);
                point.textContent = CONFIG.CALIBRATION_CLICKS - clicksOnPoint;

                if (clicksOnPoint >= CONFIG.CALIBRATION_CLICKS) {
                    currentPoint++;
                    clicksOnPoint = 0;

                    if (currentPoint >= CONFIG.CALIBRATION_POINTS) {
                        completeCalibration();
                    } else {
                        progress.textContent = `Point ${currentPoint + 1} of ${CONFIG.CALIBRATION_POINTS}`;
                        createPoint(currentPoint);
                    }
                }
            });

            area.appendChild(point);
        }

        skipBtn.addEventListener('click', completeCalibration);

        createPoint(0);
    }

    /**
     * Complete calibration and remove overlay
     */
    function completeCalibration() {
        const overlay = document.getElementById('gaze-calibration-overlay');
        if (overlay) {
            overlay.style.opacity = '0';
            overlay.style.transition = 'opacity 0.3s';
            setTimeout(() => overlay.remove(), 300);
        }

        isCalibrated = true;
        localStorage.setItem('gazeCalibrated', 'true');
        console.log('✓ Calibration complete');

        // Show accuracy validation if enabled
        if (CONFIG.VALIDATION_ENABLED) {
            showValidation();
        } else {
            emit('calibrationComplete');
        }
    }

    /**
     * Show accuracy validation after calibration
     */
    function showValidation() {
        const validationHTML = `
        <div id="gaze-validation-overlay" style="
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0, 0, 0, 0.95);
            z-index: 999999;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            color: white;
        ">
            <h2 style="margin-bottom: 20px;">🎯 Accuracy Test</h2>
            <p style="color: rgba(255,255,255,0.7); margin-bottom: 30px;">Look at the green dot - we'll measure accuracy</p>
            <div id="validation-target" style="
                width: 30px;
                height: 30px;
                background: #7bed9f;
                border-radius: 50%;
                position: absolute;
                transition: all 0.3s;
            "></div>
            <p id="validation-result" style="
                position: fixed;
                bottom: 50px;
                font-size: 18px;
                color: #7bed9f;
            "></p>
        </div>
        `;

        const overlay = document.createElement('div');
        overlay.innerHTML = validationHTML;
        document.body.appendChild(overlay.firstElementChild);

        const target = document.getElementById('validation-target');
        const result = document.getElementById('validation-result');

        const testPoints = [
            { x: 25, y: 25 }, { x: 75, y: 25 },
            { x: 50, y: 50 },
            { x: 25, y: 75 }, { x: 75, y: 75 }
        ];

        let errors = [];
        let currentTest = 0;

        function runTest(index) {
            if (index >= testPoints.length) {
                // Calculate average error
                const avgError = errors.reduce((a, b) => a + b, 0) / errors.length;
                result.textContent = `Average Accuracy: ${Math.round(avgError)}px`;

                setTimeout(() => {
                    document.getElementById('gaze-validation-overlay').remove();
                    emit('calibrationComplete', { accuracy: avgError });
                }, 2000);
                return;
            }

            const pos = testPoints[index];
            target.style.left = pos.x + '%';
            target.style.top = pos.y + '%';
            target.style.marginLeft = '-15px';
            target.style.marginTop = '-15px';

            result.textContent = `Looking at point ${index + 1} of ${testPoints.length}...`;

            // Measure after 1 second
            setTimeout(() => {
                const prediction = webgazer.getCurrentPrediction();
                if (prediction) {
                    const targetX = window.innerWidth * pos.x / 100;
                    const targetY = window.innerHeight * pos.y / 100;
                    const error = Math.sqrt(
                        Math.pow(prediction.x - targetX, 2) +
                        Math.pow(prediction.y - targetY, 2)
                    );
                    errors.push(error);
                }
                runTest(index + 1);
            }, 1500);
        }

        setTimeout(() => runTest(0), 500);
    }

    /**
     * Handle incoming gaze data
     */
    function handleGaze(x, y) {
        if (!isCalibrated) return;

        let foundElement = null;

        // Check all registered targets
        for (const target of gazeTargets) {
            const rect = target.getBoundingClientRect();
            const inBounds = (
                x >= rect.left - CONFIG.GAZE_TOLERANCE &&
                x <= rect.right + CONFIG.GAZE_TOLERANCE &&
                y >= rect.top - CONFIG.GAZE_TOLERANCE &&
                y <= rect.bottom + CONFIG.GAZE_TOLERANCE
            );

            if (inBounds) {
                foundElement = target;
                break;
            }
        }

        // Handle state changes
        if (foundElement !== activeElement) {
            // Leave previous
            if (activeElement) {
                emit('gazeLeave', activeElement);
                activeElement.classList.remove('gaze-active');
                clearDwellTimer();
            }

            // Enter new
            if (foundElement) {
                emit('gazeEnter', foundElement);
                foundElement.classList.add('gaze-active');
                startDwellTimer(foundElement);
            }

            activeElement = foundElement;
        }
    }

    /**
     * Start dwell timer for an element
     */
    function startDwellTimer(element) {
        clearDwellTimer();
        dwellProgress = 0;

        // Create progress indicator
        let progressRing = element.querySelector('.gaze-dwell-progress');
        if (!progressRing) {
            progressRing = document.createElement('div');
            progressRing.className = 'gaze-dwell-progress';
            progressRing.style.cssText = `
                position: absolute;
                top: 0; left: 0; right: 0; bottom: 0;
                border: 3px solid transparent;
                border-top-color: #7bed9f;
                border-radius: inherit;
                animation: gaze-dwell-spin ${CONFIG.DWELL_TIME}ms linear forwards;
                pointer-events: none;
            `;
            element.style.position = 'relative';
            element.appendChild(progressRing);
        }

        dwellTimer = setTimeout(() => {
            emit('gazeDwell', element);
            element.classList.add('gaze-clicked');
            setTimeout(() => element.classList.remove('gaze-clicked'), 300);
            clearDwellTimer();
        }, CONFIG.DWELL_TIME);
    }

    /**
     * Clear dwell timer
     */
    function clearDwellTimer() {
        if (dwellTimer) {
            clearTimeout(dwellTimer);
            dwellTimer = null;
        }
        // Remove progress indicators
        document.querySelectorAll('.gaze-dwell-progress').forEach(el => el.remove());
    }

    /**
     * Register an element as a gaze target
     */
    function registerTarget(element) {
        if (!gazeTargets.includes(element)) {
            gazeTargets.push(element);
            element.classList.add('gaze-target');
        }
    }

    /**
     * Register multiple elements
     */
    function registerTargets(selector) {
        document.querySelectorAll(selector).forEach(el => registerTarget(el));
    }

    /**
     * Event system
     */
    function on(event, callback) {
        if (eventListeners[event]) {
            eventListeners[event].push(callback);
        }
    }

    function emit(event, data) {
        if (eventListeners[event]) {
            eventListeners[event].forEach(cb => cb(data));
        }
    }

    /**
     * Style the WebGazer prediction dot
     */
    function stylePredictionDot() {
        const style = document.createElement('style');
        style.textContent = `
            #webgazerGazeDot {
                background: rgba(123, 237, 159, 0.6) !important;
                border: 2px solid #7bed9f !important;
                width: 20px !important;
                height: 20px !important;
                border-radius: 50% !important;
                pointer-events: none !important;
                z-index: 99999 !important;
            }
            
            .gaze-target {
                transition: box-shadow 0.2s, transform 0.2s;
            }
            
            .gaze-active {
                box-shadow: 0 0 0 3px rgba(123, 237, 159, 0.5);
                transform: scale(1.02);
            }
            
            .gaze-clicked {
                transform: scale(0.95);
                box-shadow: 0 0 20px rgba(123, 237, 159, 0.8);
            }
            
            @keyframes gaze-dwell-spin {
                0% { transform: rotate(0deg); border-color: transparent; border-top-color: #7bed9f; }
                25% { border-right-color: #7bed9f; }
                50% { border-bottom-color: #7bed9f; }
                75% { border-left-color: #7bed9f; }
                100% { transform: rotate(360deg); border-color: #7bed9f; }
            }
        `;
        document.head.appendChild(style);
    }

    /**
     * Pause/resume tracking
     */
    function pause() {
        if (webgazer) webgazer.pause();
    }

    function resume() {
        if (webgazer) webgazer.resume();
    }

    /**
     * Reset calibration
     */
    function resetCalibration() {
        localStorage.removeItem('gazeCalibrated');
        isCalibrated = false;
        if (webgazer) webgazer.clearData();
        showCalibration();
    }

    /**
     * Cleanup
     */
    function destroy() {
        if (webgazer) webgazer.end();
        gazeTargets = [];
        isInitialized = false;
        isCalibrated = false;
    }

    // ============ Public API ============
    return {
        init,
        showCalibration,
        resetCalibration,
        registerTarget,
        registerTargets,
        on,
        pause,
        resume,
        destroy,

        // Expose config for customization
        get config() { return CONFIG; },
        set dwellTime(ms) { CONFIG.DWELL_TIME = ms; }
    };

})();

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = GazeController;
}
