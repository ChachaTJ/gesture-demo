/**
 * 📱 Gesture Controller Module
 * Maps eye gaze + mouth-opening (click) to smartphone touch gestures
 * 
 * Gestures:
 * - Tap: Short click (<300ms) with minimal movement (<20px)
 * - Scroll: Click-hold + vertical gaze movement
 * - Swipe Back: Click-hold + fast leftward gaze movement
 * - Swipe Forward: Click-hold + fast rightward gaze movement
 * - Long Press: Extended click (>1.5s) in place
 */

const GestureController = (function () {
    'use strict';

    // ============ Configuration ============
    const CONFIG = {
        TAP_MAX_DURATION: 600,      // Max ms for tap gesture
        TAP_MAX_MOVEMENT: 30,       // Max px movement for tap
        SCROLL_THRESHOLD: 15,       // Min px for scroll detection
        SWIPE_THRESHOLD: 30,        // Min px for swipe detection
        SWIPE_SPEED_THRESHOLD: 240, // Min px/s for swipe
        LONG_PRESS_DURATION: 1500,  // Ms for long press
        SCROLL_SENSITIVITY: 3,      // Scroll multiplier
    };

    // ============ State ============
    let isClickActive = false;
    let clickStartTime = null;
    let clickStartPosition = { x: 0, y: 0 };
    let currentPosition = { x: 0, y: 0 };
    let positionHistory = [];
    let activeGesture = null;
    let longPressTimer = null;
    let scrollAccumulator = 0;

    const eventListeners = {
        tap: [],
        scroll: [],
        swipeLeft: [],
        swipeRight: [],
        longPress: [],
        gestureStart: [],
        gestureEnd: [],
        gestureUpdate: []
    };

    // ============ Core Functions ============

    /**
     * Start a touch (mouth opened / click down)
     */
    function startTouch(x, y) {
        if (isClickActive) return;

        isClickActive = true;
        clickStartTime = Date.now();
        clickStartPosition = { x, y };
        currentPosition = { x, y };
        positionHistory = [{ x, y, t: clickStartTime }];
        activeGesture = null;
        scrollAccumulator = 0;

        emit('gestureStart', { x, y });

        // Start long press timer
        longPressTimer = setTimeout(() => {
            if (isClickActive && getMovementDistance() < CONFIG.TAP_MAX_MOVEMENT) {
                activeGesture = 'longPress';
                emit('longPress', { x: currentPosition.x, y: currentPosition.y });
            }
        }, CONFIG.LONG_PRESS_DURATION);
    }

    /**
     * Update gaze position during touch
     */
    function updatePosition(x, y) {
        if (!isClickActive) return;

        const prevPosition = { ...currentPosition };
        currentPosition = { x, y };
        positionHistory.push({ x, y, t: Date.now() });

        // Keep only recent history (last 500ms)
        const now = Date.now();
        positionHistory = positionHistory.filter(p => now - p.t < 500);

        // Calculate deltas
        const deltaX = x - clickStartPosition.x;
        const deltaY = y - clickStartPosition.y;
        const frameDeltaY = y - prevPosition.y;

        // Detect scroll gesture (vertical movement during hold)
        if (Math.abs(deltaY) > CONFIG.SCROLL_THRESHOLD &&
            Math.abs(deltaY) > Math.abs(deltaX)) {

            if (activeGesture !== 'longPress') {
                activeGesture = 'scroll';
                clearTimeout(longPressTimer);

                // Accumulate scroll
                scrollAccumulator += frameDeltaY * CONFIG.SCROLL_SENSITIVITY;

                emit('scroll', {
                    deltaY: frameDeltaY * CONFIG.SCROLL_SENSITIVITY,
                    totalDeltaY: deltaY,
                    direction: deltaY > 0 ? 'down' : 'up'
                });
            }
        }

        emit('gestureUpdate', { x, y, deltaX, deltaY, gesture: activeGesture });
    }

    /**
     * End touch (mouth closed / click up)
     */
    function endTouch(x, y) {
        if (!isClickActive) return;

        clearTimeout(longPressTimer);

        const duration = Date.now() - clickStartTime;
        const movement = getMovementDistance();
        const velocity = getSwipeVelocity();

        // Determine gesture type
        let gestureType = activeGesture;

        if (!gestureType) {
            // Check for swipe
            const deltaX = x - clickStartPosition.x;

            if (Math.abs(velocity.x) > CONFIG.SWIPE_SPEED_THRESHOLD &&
                Math.abs(deltaX) > CONFIG.SWIPE_THRESHOLD) {
                if (velocity.x < 0) {
                    gestureType = 'swipeLeft';
                    emit('swipeLeft', { velocity: velocity.x });
                } else {
                    gestureType = 'swipeRight';
                    emit('swipeRight', { velocity: velocity.x });
                }
            }
            // Check for tap
            else if (duration < CONFIG.TAP_MAX_DURATION && movement < CONFIG.TAP_MAX_MOVEMENT) {
                gestureType = 'tap';
                emit('tap', { x, y, duration });
            }
        }

        emit('gestureEnd', {
            gesture: gestureType || 'none',
            duration,
            movement,
            velocity
        });

        // Reset state
        isClickActive = false;
        clickStartTime = null;
        activeGesture = null;
        scrollAccumulator = 0;
    }

    /**
     * Calculate total movement distance from start
     */
    function getMovementDistance() {
        const dx = currentPosition.x - clickStartPosition.x;
        const dy = currentPosition.y - clickStartPosition.y;
        return Math.sqrt(dx * dx + dy * dy);
    }

    /**
     * Calculate swipe velocity from recent positions
     */
    function getSwipeVelocity() {
        if (positionHistory.length < 2) return { x: 0, y: 0 };

        const recent = positionHistory.slice(-5);
        const first = recent[0];
        const last = recent[recent.length - 1];
        const dt = (last.t - first.t) / 1000; // seconds

        if (dt === 0) return { x: 0, y: 0 };

        return {
            x: (last.x - first.x) / dt,
            y: (last.y - first.y) / dt
        };
    }

    // ============ Event System ============

    function on(event, callback) {
        if (eventListeners[event]) {
            eventListeners[event].push(callback);
        }
    }

    function off(event, callback) {
        if (eventListeners[event]) {
            eventListeners[event] = eventListeners[event].filter(cb => cb !== callback);
        }
    }

    function emit(event, data) {
        if (eventListeners[event]) {
            eventListeners[event].forEach(cb => cb(data));
        }
    }

    // ============ Utilities ============

    function isActive() {
        return isClickActive;
    }

    function getCurrentGesture() {
        return activeGesture;
    }

    function getConfig() {
        return { ...CONFIG };
    }

    function setConfig(key, value) {
        if (CONFIG.hasOwnProperty(key)) {
            CONFIG[key] = value;
        }
    }

    // ============ Public API ============
    return {
        startTouch,
        updatePosition,
        endTouch,
        on,
        off,
        isActive,
        getCurrentGesture,
        getConfig,
        setConfig
    };

})();

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = GestureController;
}
