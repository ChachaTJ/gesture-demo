/**
 * 🎯 Head Tracker Module
 * Human.js 기반 head-tracking 마우스 제어
 * 
 * Nutshell 프로젝트에서 추출 및 standalone 변환
 * 
 * Features:
 * - One-Euro Filter (jitter 제거)
 * - Head pose → 화면 좌표 변환
 * - 입 벌림 클릭 (토글 가능)
 * - 5점 캘리브레이션 지원
 * 
 * Usage:
 *   await HeadTracker.init();
 *   HeadTracker.on('point', (x, y) => moveCursor(x, y));
 *   HeadTracker.on('mouthClick', () => doClick());
 */

const HeadTracker = (function () {
    'use strict';

    // ============ Constants ============
    const HEAD_FILTER_MIN_CUTOFF = 0.4;
    const HEAD_FILTER_BETA = 0.0025;
    const HEAD_FILTER_D_CUTOFF = 1.0;
    const HEAD_POINTER_LERP = 0.12;
    const HEAD_CENTER_LERP = 0.06;
    const HEAD_EDGE_LERP = 0.10;
    const HEAD_CENTER_THRESHOLD = 0.25;
    const HEAD_EDGE_THRESHOLD = 0.7;
    const HEAD_ROTATION_INFLUENCE = 0.22;
    const HEAD_ROTATION_EDGE_GAIN = 0.35;
    const HEAD_YAW_SCALE = 25;
    const HEAD_PITCH_SCALE = 20;
    const PITCH_FALLBACK_THRESHOLD = 0.32;
    const TRANSLATION_MIN_RATIO = 0.24;
    const VERTICAL_EDGE_SCALE = 1.35;
    const HEAD_MIRROR_X = -1;
    const HEAD_MIRROR_Y = 1;
    const AUTO_CENTER_ALPHA = 0.05;
    const POINT_THROTTLE_MS = 33;
    const MOUTH_OPEN_COOLDOWN_MS = 800;
    const MOUTH_THRESHOLD_RATIO = 0.7; // 70% between closed and open

    const STORAGE_KEY_HEAD_CAL = 'headTrackerCalV2';
    const STORAGE_KEY_MOUTH_CAL = 'headTrackerMouthCalV1';

    const LEFT_EYE_CANDIDATES = [33, 246, 161, 160, 159, 130];
    const RIGHT_EYE_CANDIDATES = [362, 466, 388, 387, 386, 359];
    const NOSE_CANDIDATES = [1, 4, 5, 6, 197, 168, 2, 94];

    const DEFAULT_HEAD_CAL = {
        cx: 0, cy: 0,
        left: 0.4, right: 0.4,
        up: 0.3, down: 0.35,
        version: 2, ts: 0
    };

    // ============ State ============
    let human = null;
    let video = null;
    let stream = null;
    let rafHandle = null;
    let detectInProgress = false;
    let isInitialized = false;
    let isRunning = false;

    let headCal = null;
    let mouthCal = null;
    let headFilterX = null;
    let headFilterY = null;
    let lastHeadPoint = null;
    let lastPointTs = 0;
    let headAutoCenter = { nx: 0, ny: 0, ready: false };

    let mouthClickEnabled = true; // 토글 가능
    let lastMouthClickTime = 0;
    let lastMouthRatio = 0;

    const eventListeners = {
        point: [],
        mouthClick: [],
        status: [],
        headFrame: [],
        mouthRatio: []
    };

    // ============ One-Euro Filter ============
    function computeAlpha(fc, dtSeconds) {
        const tau = 1 / (2 * Math.PI * Math.max(1e-3, fc));
        return 1 / (1 + tau / Math.max(1e-4, dtSeconds));
    }

    function createOneEuroFilter(minCut = HEAD_FILTER_MIN_CUTOFF, beta = HEAD_FILTER_BETA, dCut = HEAD_FILTER_D_CUTOFF) {
        let prevValue = null;
        let prevTimestamp = null;
        let dxEstimate = null;
        let smoothed = null;

        return (value, timestampMs) => {
            if (!Number.isFinite(value) || !Number.isFinite(timestampMs)) {
                return value;
            }
            if (prevTimestamp === null) {
                prevTimestamp = timestampMs;
                prevValue = value;
                smoothed = value;
                dxEstimate = 0;
                return value;
            }
            const dt = Math.max(1e-3, (timestampMs - prevTimestamp) / 1000);
            prevTimestamp = timestampMs;
            const rawDerivative = (value - prevValue) / dt;
            prevValue = value;
            const alphaDerivative = computeAlpha(dCut, dt);
            dxEstimate = dxEstimate == null ? rawDerivative : (alphaDerivative * rawDerivative) + ((1 - alphaDerivative) * dxEstimate);
            const cutoff = minCut + beta * Math.abs(dxEstimate);
            const alpha = computeAlpha(cutoff, dt);
            smoothed = smoothed == null ? value : (alpha * value) + ((1 - alpha) * smoothed);
            return smoothed;
        };
    }

    // ============ Point/Mesh Helpers ============
    function normalizePoint(point) {
        if (!point) return null;
        if (Array.isArray(point) || ArrayBuffer.isView(point)) {
            const x = point[0];
            const y = point[1];
            if (Number.isFinite(x) && Number.isFinite(y)) {
                return [x, y];
            }
            return null;
        }
        if (typeof point === 'object' && point) {
            const x = Number(point.x ?? point[0]);
            const y = Number(point.y ?? point[1]);
            if (Number.isFinite(x) && Number.isFinite(y)) {
                return [x, y];
            }
        }
        return null;
    }

    function pick(mesh, idx) {
        if (!mesh || idx < 0) return null;
        if (Array.isArray(mesh)) {
            if (idx < mesh.length) {
                const direct = normalizePoint(mesh[idx]);
                if (direct) return direct;
            }
            if (typeof mesh[0] === 'number') {
                const base = idx * 3;
                if (base + 1 < mesh.length) {
                    const x = mesh[base];
                    const y = mesh[base + 1];
                    if (Number.isFinite(x) && Number.isFinite(y)) {
                        return [x, y];
                    }
                }
            }
            return null;
        }
        const raw = mesh[idx];
        return normalizePoint(raw);
    }

    function centroid(points) {
        if (!Array.isArray(points) || !points.length) return null;
        let x = 0, y = 0, count = 0;
        for (let i = 0; i < points.length; i++) {
            const p = normalizePoint(points[i]);
            if (!p) continue;
            x += p[0];
            y += p[1];
            count++;
        }
        if (!count) return null;
        return [x / count, y / count];
    }

    function pointFromAnnotations(annotations, keys) {
        if (!annotations || !Array.isArray(keys)) return null;
        for (let i = 0; i < keys.length; i++) {
            const arr = annotations[keys[i]];
            const c = centroid(arr);
            if (c) return c;
        }
        return null;
    }

    function resolvePoint(mesh, annotations, candidates, annotationKeys) {
        for (let i = 0; i < candidates.length; i++) {
            const p = pick(mesh, candidates[i]);
            if (p) return p;
        }
        if (annotations) {
            const annPoint = pointFromAnnotations(annotations, annotationKeys);
            if (annPoint) return annPoint;
        }
        return null;
    }

    // ============ Head Frame Computation ============
    function computeHeadFrame(face) {
        if (!face) return null;
        const mesh = face.mesh;
        if (!mesh) return null;
        const annotations = face.annotations || {};

        let leftEye = resolvePoint(mesh, annotations, LEFT_EYE_CANDIDATES, ['leftEyeUpper0', 'leftEyeLower0', 'leftEyeUpper1']);
        let rightEye = resolvePoint(mesh, annotations, RIGHT_EYE_CANDIDATES, ['rightEyeUpper0', 'rightEyeLower0', 'rightEyeUpper1']);
        let noseTip = resolvePoint(mesh, annotations, NOSE_CANDIDATES, ['noseTip', 'midwayBetweenEyes']);

        if ((!leftEye || !rightEye) && annotations.midwayBetweenEyes) {
            const fallback = centroid(annotations.midwayBetweenEyes);
            if (fallback) {
                if (!leftEye) leftEye = [fallback[0] - 5, fallback[1]];
                if (!rightEye) rightEye = [fallback[0] + 5, fallback[1]];
            }
        }

        if (!leftEye || !rightEye) return null;

        if (!noseTip) {
            const centerGuess = [(leftEye[0] + rightEye[0]) / 2, (leftEye[1] + rightEye[1]) / 2];
            noseTip = centerGuess;
        }
        if (!noseTip) return null;

        const eyeVec = [rightEye[0] - leftEye[0], rightEye[1] - leftEye[1]];
        const iod = Math.hypot(eyeVec[0], eyeVec[1]);
        if (!iod || !Number.isFinite(iod)) return null;

        eyeVec[0] /= iod;
        eyeVec[1] /= iod;
        const vertical = [-eyeVec[1], eyeVec[0]];
        const center = [(leftEye[0] + rightEye[0]) / 2, (leftEye[1] + rightEye[1]) / 2];
        const noseVec = [noseTip[0] - center[0], noseTip[1] - center[1]];
        const u = (noseVec[0] * eyeVec[0]) + (noseVec[1] * eyeVec[1]);
        const v = (noseVec[0] * vertical[0]) + (noseVec[1] * vertical[1]);
        const normalization = Math.max(0.01, iod);
        const nx = Math.max(-1.5, Math.min(1.5, u / normalization));
        const ny = Math.max(-1.5, Math.min(1.5, v / normalization));

        return {
            nx: nx * HEAD_MIRROR_X,
            ny: ny * HEAD_MIRROR_Y,
            iod,
            center,
            leftEye,
            rightEye,
            nose: noseTip,
            ex: eyeVec,
            ey: vertical
        };
    }

    // ============ Coordinate Mapping ============
    function mapHeadLocalToXY(nx, ny, cal) {
        if (!cal) return null;
        if (!Number.isFinite(nx) || !Number.isFinite(ny)) return null;

        const centerX = Number.isFinite(cal.cx) ? cal.cx : 0;
        const centerY = Number.isFinite(cal.cy) ? cal.cy : 0;

        const dx = nx - centerX;
        const dy = ny - centerY;

        const leftRange = Math.max(1e-3, cal.left || 0.01);
        const rightRange = Math.max(1e-3, cal.right || 0.01);
        const upRange = Math.max(1e-3, cal.up || 0.01);
        const downRange = Math.max(1e-3, cal.down || 0.01);

        let tx;
        if (dx < 0) {
            const ratio = Math.max(-1, Math.min(0, dx / leftRange));
            tx = 0.5 + 0.5 * ratio;
        } else {
            const ratio = Math.max(0, Math.min(1, dx / rightRange));
            tx = 0.5 + 0.5 * ratio;
        }

        let ty;
        if (dy < 0) {
            const ratio = Math.max(-1, Math.min(0, dy / upRange));
            ty = 0.5 + 0.5 * ratio;
        } else {
            const ratio = Math.max(0, Math.min(1, dy / downRange));
            ty = 0.5 + 0.5 * ratio;
        }

        tx = Math.max(0, Math.min(1, tx));
        ty = Math.max(0, Math.min(1, ty));

        const viewportWidth = Math.max(1, window.innerWidth || 1);
        const viewportHeight = Math.max(1, window.innerHeight || 1);

        const px = Math.max(0, Math.min(viewportWidth - 1, tx * viewportWidth));
        const py = Math.max(0, Math.min(viewportHeight - 1, ty * viewportHeight));
        return [px, py];
    }

    // ============ Mouth Detection ============
    function calculateMouthRatio(annotations) {
        if (!annotations) return 0;
        if (!annotations.lipsUpperOuter || !annotations.lipsLowerOuter) return 0;

        const upperLip = annotations.lipsUpperOuter;
        const lowerLip = annotations.lipsLowerOuter;

        const topPoint = upperLip[Math.floor(upperLip.length / 2)];
        const bottomPoint = lowerLip[Math.floor(lowerLip.length / 2)];

        if (!topPoint || !bottomPoint) return 0;

        const mouthHeight = Math.abs(bottomPoint[1] - topPoint[1]);
        const leftPoint = upperLip[0];
        const rightPoint = upperLip[upperLip.length - 1];

        if (!leftPoint || !rightPoint) return 0;

        const mouthWidth = Math.abs(rightPoint[0] - leftPoint[0]);
        if (mouthWidth === 0) return 0;

        return mouthHeight / mouthWidth;
    }

    // ============ Event System ============
    function emit(event, ...args) {
        if (eventListeners[event]) {
            eventListeners[event].forEach(cb => {
                try { cb(...args); } catch (e) { console.error('[HeadTracker] Event error:', e); }
            });
        }
    }

    function on(event, callback) {
        if (eventListeners[event]) {
            eventListeners[event].push(callback);
        }
        return () => off(event, callback);
    }

    function off(event, callback) {
        if (eventListeners[event]) {
            const idx = eventListeners[event].indexOf(callback);
            if (idx >= 0) eventListeners[event].splice(idx, 1);
        }
    }

    // ============ Storage ============
    function loadCalibration() {
        try {
            const headData = localStorage.getItem(STORAGE_KEY_HEAD_CAL);
            if (headData) headCal = JSON.parse(headData);
        } catch (e) { console.warn('[HeadTracker] Failed to load head calibration:', e); }

        try {
            const mouthData = localStorage.getItem(STORAGE_KEY_MOUTH_CAL);
            if (mouthData) mouthCal = JSON.parse(mouthData);
        } catch (e) { console.warn('[HeadTracker] Failed to load mouth calibration:', e); }
    }

    function saveHeadCalibration(cal) {
        headCal = cal;
        try {
            localStorage.setItem(STORAGE_KEY_HEAD_CAL, JSON.stringify(cal));
            console.log('[HeadTracker] Head calibration saved');
        } catch (e) { console.warn('[HeadTracker] Failed to save head calibration:', e); }
    }

    function saveMouthCalibration(cal) {
        mouthCal = cal;
        try {
            localStorage.setItem(STORAGE_KEY_MOUTH_CAL, JSON.stringify(cal));
            console.log('[HeadTracker] Mouth calibration saved');
        } catch (e) { console.warn('[HeadTracker] Failed to save mouth calibration:', e); }
    }

    // ============ Detection Loop ============
    function processDetection(result, ts) {
        const face = result && result.face && result.face[0] ? result.face[0] : null;

        if (!face) {
            headFilterX = null;
            headFilterY = null;
            lastHeadPoint = null;
            headAutoCenter = { nx: headCal && headCal.cx || 0, ny: headCal && headCal.cy || 0, ready: Boolean(headCal) };
            return;
        }

        const yawRad = Number(face.rotation && face.rotation.angle ? face.rotation.angle.yaw : 0);
        const pitchRad = Number(face.rotation && face.rotation.angle ? face.rotation.angle.pitch : 0);
        const yawDeg = yawRad * (180 / Math.PI);
        const pitchDeg = pitchRad * (180 / Math.PI);

        let headFrame = null;
        try {
            headFrame = computeHeadFrame(face);
        } catch (error) {
            headFrame = null;
        }

        if (headFrame) {
            emit('headFrame', { nx: headFrame.nx, ny: headFrame.ny, yawDeg, pitchDeg, ts });
        }

        // Mouth detection
        if (face.annotations) {
            const mouthRatio = calculateMouthRatio(face.annotations);
            lastMouthRatio = mouthRatio;
            emit('mouthRatio', mouthRatio);

            // 입 벌림 클릭 체크
            if (mouthClickEnabled && mouthCal && mouthCal.threshold) {
                if (mouthRatio > mouthCal.threshold && (ts - lastMouthClickTime) > MOUTH_OPEN_COOLDOWN_MS) {
                    lastMouthClickTime = ts;
                    emit('mouthClick', { mouthRatio, ts });
                    console.log(`[HeadTracker] 👄 MOUTH CLICK! MAR: ${mouthRatio.toFixed(3)} > ${mouthCal.threshold.toFixed(3)}`);
                }
            }
        }

        // Head tracking point
        if (!headFrame) return;

        if (!headFilterX || !headFilterY) {
            headFilterX = createOneEuroFilter();
            headFilterY = createOneEuroFilter();
            lastHeadPoint = null;
        }

        let activeCal;
        if (headCal && headCal.version === 2) {
            activeCal = headCal;
        } else {
            if (!headAutoCenter.ready) {
                headAutoCenter = { nx: headFrame.nx, ny: headFrame.ny, ready: true };
            } else {
                headAutoCenter.nx += (headFrame.nx - headAutoCenter.nx) * AUTO_CENTER_ALPHA;
                headAutoCenter.ny += (headFrame.ny - headAutoCenter.ny) * AUTO_CENTER_ALPHA;
            }
            activeCal = { ...DEFAULT_HEAD_CAL, cx: headAutoCenter.nx, cy: headAutoCenter.ny };
        }

        const yawNorm = Math.max(-1, Math.min(1, yawDeg / HEAD_YAW_SCALE));
        const pitchNorm = Math.max(-1, Math.min(1, pitchDeg / HEAD_PITCH_SCALE));
        const centerNx = activeCal.cx || 0;
        const centerNy = activeCal.cy || 0;
        const leftRange = Math.max(1e-3, activeCal.left || 0.01);
        const rightRange = Math.max(1e-3, activeCal.right || 0.01);
        const upRange = Math.max(1e-3, activeCal.up || 0.01);
        const downRange = Math.max(1e-3, activeCal.down || 0.01);

        const offsetNx = headFrame.nx - centerNx;
        const offsetNy = headFrame.ny - centerNy;

        let normX = offsetNx < 0 ? offsetNx / leftRange : offsetNx / rightRange;
        const normYTrans = offsetNy < 0 ? offsetNy / upRange : offsetNy / downRange;
        let normY = (pitchDeg / HEAD_PITCH_SCALE) + normYTrans * 0.35;

        const translationRatioX = Math.abs(offsetNx) / (offsetNx < 0 ? leftRange : rightRange);
        const translationRatioY = Math.abs(offsetNy) / (offsetNy < 0 ? upRange : downRange);

        if (Math.abs(normX) < 1) {
            normX += yawNorm * HEAD_ROTATION_INFLUENCE * (1 - Math.min(1, Math.abs(normX)));
        }
        if (Math.abs(normY) < 1) {
            normY += pitchNorm * (HEAD_ROTATION_INFLUENCE * 0.6) * (1 - Math.min(1, Math.abs(normY)));
        }

        if (Math.abs(normY) > HEAD_EDGE_THRESHOLD && translationRatioY < TRANSLATION_MIN_RATIO && Math.abs(pitchNorm) > PITCH_FALLBACK_THRESHOLD) {
            const edgeBlend = HEAD_ROTATION_EDGE_GAIN * Math.sign(pitchNorm);
            normY = Math.max(-1.4, Math.min(1.4, normY + edgeBlend));
        }

        normX = Math.max(-1.2, Math.min(1.2, normX));
        normY = Math.max(-1.4, Math.min(1.4, normY));

        const scaledUpRange = upRange * (normY < 0 ? VERTICAL_EDGE_SCALE : 1);
        const scaledDownRange = downRange * (normY > 0 ? VERTICAL_EDGE_SCALE : 1);

        const targetNx = normX < 0 ? centerNx + normX * leftRange : centerNx + normX * rightRange;
        const targetNy = normY < 0 ? centerNy + normY * scaledUpRange : centerNy + normY * scaledDownRange;
        const mapped = mapHeadLocalToXY(targetNx, targetNy, activeCal);

        if (mapped) {
            const filteredX = headFilterX(mapped[0], ts);
            const filteredY = headFilterY(mapped[1], ts);
            let finalX = Number.isFinite(filteredX) ? filteredX : mapped[0];
            let finalY = Number.isFinite(filteredY) ? filteredY : mapped[1];

            let smoothingAlpha = HEAD_POINTER_LERP;
            if (Math.abs(normX) < HEAD_CENTER_THRESHOLD && Math.abs(normY) < HEAD_CENTER_THRESHOLD) {
                smoothingAlpha = HEAD_CENTER_LERP;
            } else if (Math.abs(normX) > HEAD_EDGE_THRESHOLD || Math.abs(normY) > HEAD_EDGE_THRESHOLD) {
                smoothingAlpha = HEAD_EDGE_LERP;
            }

            if (lastHeadPoint) {
                finalX = lastHeadPoint[0] + smoothingAlpha * (finalX - lastHeadPoint[0]);
                finalY = lastHeadPoint[1] + smoothingAlpha * (finalY - lastHeadPoint[1]);
                lastHeadPoint[0] = finalX;
                lastHeadPoint[1] = finalY;
            } else {
                lastHeadPoint = [finalX, finalY];
            }

            if (ts - lastPointTs >= POINT_THROTTLE_MS) {
                lastPointTs = ts;
                emit('point', finalX, finalY, { ts });
            }
        }
    }

    async function detectionLoop() {
        if (!isRunning || !human || !video) {
            rafHandle = null;
            return;
        }

        rafHandle = requestAnimationFrame(detectionLoop);

        if (detectInProgress) return;

        try {
            if (video.readyState >= 2) {
                detectInProgress = true;
                const startTs = performance.now();
                const result = await human.detect(video);
                processDetection(result, startTs);
                detectInProgress = false;
            }
        } catch (error) {
            console.warn('[HeadTracker] detect failed:', error);
            detectInProgress = false;
        }
    }

    // ============ Public API ============
    async function init(options = {}) {
        if (isInitialized) {
            console.log('[HeadTracker] Already initialized');
            return true;
        }

        emit('status', 'loading', 'Loading Human.js models...');

        // Load Human.js dynamically
        const humanUrl = options.humanPath || './lib/human/human.esm.js';
        const modelsPath = options.modelsPath || './lib/human/models/';

        try {
            const module = await import(humanUrl);
            const HumanCtor = module.default || module.Human || module;

            human = new HumanCtor({
                backend: 'webgl',
                modelBasePath: modelsPath,
                cacheSensitivity: 0,
                face: {
                    enabled: true,
                    detector: { enabled: true, rotation: true, return: true, maxDetected: 1 },
                    mesh: { enabled: true },
                    iris: { enabled: false },
                    attention: false,
                    description: false,
                    emotion: { enabled: false },
                    antispoof: false,
                    liveness: false
                },
                filter: {
                    enabled: true,
                    equalization: false,
                    temporalSmoothing: 0.5
                }
            });

            await human.load();
            console.log('[HeadTracker] Human.js loaded successfully');

            if (typeof human.warmup === 'function') {
                try { await human.warmup(); } catch (e) { /* ignore */ }
            }

            loadCalibration();
            isInitialized = true;
            emit('status', 'ready', 'Head tracker ready');
            return true;
        } catch (error) {
            console.error('[HeadTracker] Failed to load Human.js:', error);
            emit('status', 'error', 'Failed to load Human.js');
            throw error;
        }
    }

    async function start() {
        if (!isInitialized) {
            throw new Error('HeadTracker not initialized. Call init() first.');
        }

        if (isRunning) {
            console.log('[HeadTracker] Already running');
            return;
        }

        emit('status', 'starting', 'Starting camera...');

        // Create hidden video element
        video = document.createElement('video');
        video.style.cssText = 'position: fixed; top: -10000px; left: -10000px; width: 1px; height: 1px;';
        video.autoplay = true;
        video.playsInline = true;
        video.muted = true;
        document.body.appendChild(video);

        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: 'user',
                    width: { ideal: 320 },
                    height: { ideal: 240 },
                    frameRate: { ideal: 30, max: 30 }
                },
                audio: false
            });
            video.srcObject = stream;
            await video.play();

            isRunning = true;
            emit('status', 'live', 'Head tracking active');
            console.log('[HeadTracker] Camera started, head tracking active');

            detectionLoop();
        } catch (error) {
            console.error('[HeadTracker] Camera access denied:', error);
            emit('status', 'error', 'Camera access rejected');
            throw error;
        }
    }

    function stop() {
        isRunning = false;

        if (rafHandle) {
            cancelAnimationFrame(rafHandle);
            rafHandle = null;
        }

        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            stream = null;
        }

        if (video && video.parentElement) {
            video.srcObject = null;
            video.parentElement.removeChild(video);
            video = null;
        }

        emit('status', 'stopped', 'Head tracking stopped');
        console.log('[HeadTracker] Stopped');
    }

    function setMouthClickEnabled(enabled) {
        mouthClickEnabled = Boolean(enabled);
        console.log('[HeadTracker] Mouth click:', mouthClickEnabled ? 'enabled' : 'disabled');
    }

    function getMouthClickEnabled() {
        return mouthClickEnabled;
    }

    function getLastMouthRatio() {
        return lastMouthRatio;
    }

    function getCalibration() {
        return { head: headCal, mouth: mouthCal };
    }

    function setHeadCalibration(cal) {
        saveHeadCalibration(cal);
        headFilterX = null;
        headFilterY = null;
        lastHeadPoint = null;
    }

    function setMouthCalibration(cal) {
        saveMouthCalibration(cal);
    }

    function resetCalibration() {
        headCal = null;
        mouthCal = null;
        localStorage.removeItem(STORAGE_KEY_HEAD_CAL);
        localStorage.removeItem(STORAGE_KEY_MOUTH_CAL);
        headFilterX = null;
        headFilterY = null;
        lastHeadPoint = null;
        headAutoCenter = { nx: 0, ny: 0, ready: false };
        console.log('[HeadTracker] Calibration reset');
    }

    function isActive() {
        return isRunning;
    }

    function getState() {
        return {
            initialized: isInitialized,
            running: isRunning,
            hasHeadCal: Boolean(headCal),
            hasMouthCal: Boolean(mouthCal),
            mouthClickEnabled,
            lastMouthRatio
        };
    }

    return {
        init,
        start,
        stop,
        on,
        off,
        setMouthClickEnabled,
        getMouthClickEnabled,
        getLastMouthRatio,
        getCalibration,
        setHeadCalibration,
        setMouthCalibration,
        resetCalibration,
        isActive,
        getState,
        // Expose for calibration UI
        _internal: {
            saveHeadCalibration,
            saveMouthCalibration,
            STORAGE_KEY_HEAD_CAL,
            STORAGE_KEY_MOUTH_CAL,
            MOUTH_THRESHOLD_RATIO
        }
    };
})();

// Export for ES modules and global
export default HeadTracker;

// Also expose globally for non-module scripts
if (typeof window !== 'undefined') {
    window.HeadTracker = HeadTracker;
}
