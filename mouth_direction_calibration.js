/**
 * 👄🧭 Mouth Direction Calibration
 * 입 벌린 상태에서 상/하/좌/우 방향 캘리브레이션
 * 
 * Usage:
 *   MouthDirectionCalibration.start();  // Alt+D 로도 트리거 가능
 */

const MouthDirectionCalibration = (function () {
    'use strict';

    const MIN_SAMPLES = 15;
    const SAMPLE_INTERVAL_MS = 100;

    let calUI = null;
    let currentStep = 'idle';
    let samples = [];
    let sampleInterval = null;
    let headFrameListener = null;
    let lastHeadFrame = null;
    let calibrationData = {};

    const DIRECTIONS = ['center', 'up', 'down', 'left', 'right'];
    const DIRECTION_LABELS = {
        center: { icon: '⏺️', label: '센터 (정면)', instruction: '정면을 봐주세요' },
        up: { icon: '⬆️', label: '위', instruction: '위를 봐주세요 (입 열고)' },
        down: { icon: '⬇️', label: '아래', instruction: '아래를 봐주세요 (입 열고)' },
        left: { icon: '⬅️', label: '왼쪽', instruction: '왼쪽을 봐주세요 (입 열고)' },
        right: { icon: '➡️', label: '오른쪽', instruction: '오른쪽을 봐주세요 (입 열고)' }
    };

    function createCalibrationUI() {
        const overlay = document.createElement('div');
        overlay.id = 'mouth-dir-cal-overlay';
        overlay.style.cssText = `
            position: fixed;
            top: 0; left: 0;
            width: 100%; height: 100%;
            background: rgba(0, 0, 0, 0.92);
            z-index: 2147483647;
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        `;

        const panel = document.createElement('div');
        panel.style.cssText = `
            background: linear-gradient(135deg, #2a2a2a 0%, #1a1a1a 100%);
            border-radius: 20px;
            padding: 48px;
            max-width: 700px;
            text-align: center;
            color: white;
            box-shadow: 0 8px 32px rgba(0,0,0,0.6);
            border: 1px solid rgba(255,255,255,0.1);
        `;

        panel.innerHTML = `
            <div style="font-size: 56px; margin-bottom: 16px;">👄🧭</div>
            <h2 style="margin: 0 0 12px 0; font-size: 28px; font-weight: 600;">
                방향 캘리브레이션
            </h2>
            <p style="font-size: 14px; color: #888; margin-bottom: 24px;">
                입 열린 상태에서 4방향 동작 캘리브레이션
            </p>
            
            <div id="direction-grid" style="
                display: grid;
                grid-template-columns: repeat(3, 60px);
                grid-template-rows: repeat(3, 60px);
                gap: 8px;
                justify-content: center;
                margin: 24px 0;
            ">
                <div></div>
                <div class="dir-indicator" data-dir="up" style="background: rgba(255,255,255,0.1); border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 28px; transition: all 0.3s;">⬆️</div>
                <div></div>
                <div class="dir-indicator" data-dir="left" style="background: rgba(255,255,255,0.1); border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 28px; transition: all 0.3s;">⬅️</div>
                <div class="dir-indicator" data-dir="center" style="background: rgba(255,255,255,0.1); border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 28px; transition: all 0.3s;">⏺️</div>
                <div class="dir-indicator" data-dir="right" style="background: rgba(255,255,255,0.1); border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 28px; transition: all 0.3s;">➡️</div>
                <div></div>
                <div class="dir-indicator" data-dir="down" style="background: rgba(255,255,255,0.1); border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 28px; transition: all 0.3s;">⬇️</div>
                <div></div>
            </div>
            
            <p id="mouth-dir-instructions" style="font-size: 20px; line-height: 1.6; margin: 0 0 24px 0; color: #ddd; min-height: 60px;">
                입을 벌린 상태를 유지하면서<br>다섯 방향을 캘리브레이션합니다.
            </p>
            
            <div id="mouth-dir-progress" style="display: none; margin-bottom: 24px;">
                <div style="width: 100%; height: 8px; background: rgba(255,255,255,0.1); border-radius: 4px; overflow: hidden;">
                    <div id="progress-bar" style="width: 0%; height: 100%; background: linear-gradient(90deg, #7bed9f, #70a1ff); transition: width 0.3s;"></div>
                </div>
                <div style="font-size: 14px; color: #999; margin-top: 8px;" id="sample-count">0 / ${MIN_SAMPLES}</div>
            </div>
            
            <button id="mouth-dir-action" style="
                background: linear-gradient(90deg, #7bed9f, #70a1ff);
                color: #1a1a1a;
                border: none;
                padding: 16px 48px;
                font-size: 18px;
                font-weight: 600;
                border-radius: 50px;
                cursor: pointer;
                transition: all 0.3s;
            ">시작하기</button>
            <div style="margin-top: 20px; font-size: 13px; color: #666;">
                Space: 다음 단계 &nbsp;|&nbsp; ESC: 취소
            </div>
        `;

        overlay.appendChild(panel);
        return overlay;
    }

    function highlightDirection(dir) {
        document.querySelectorAll('.dir-indicator').forEach(el => {
            el.style.background = 'rgba(255,255,255,0.1)';
            el.style.transform = 'scale(1)';
            el.style.boxShadow = 'none';
        });

        if (dir) {
            const el = document.querySelector(`.dir-indicator[data-dir="${dir}"]`);
            if (el) {
                el.style.background = 'linear-gradient(135deg, #7bed9f, #70a1ff)';
                el.style.transform = 'scale(1.15)';
                el.style.boxShadow = '0 0 20px rgba(123, 237, 159, 0.5)';
            }
        }
    }

    function markDirectionComplete(dir) {
        const el = document.querySelector(`.dir-indicator[data-dir="${dir}"]`);
        if (el) {
            el.style.background = '#4CAF50';
            el.textContent = '✓';
        }
    }

    function updateUI(step, count = 0) {
        const instructions = document.getElementById('mouth-dir-instructions');
        const progress = document.getElementById('mouth-dir-progress');
        const progressBar = document.getElementById('progress-bar');
        const sampleCount = document.getElementById('sample-count');
        const button = document.getElementById('mouth-dir-action');

        if (!instructions || !button) return;

        if (step === 'idle') {
            instructions.innerHTML = `입을 벌린 상태를 유지하면서<br>다섯 방향을 캘리브레이션합니다.`;
            progress.style.display = 'none';
            button.textContent = '시작하기';
            button.disabled = false;
            highlightDirection(null);
        } else if (DIRECTIONS.includes(step)) {
            const info = DIRECTION_LABELS[step];
            instructions.innerHTML = `
                <strong style="color: #7bed9f; font-size: 24px;">${info.icon} ${info.label}</strong><br>
                ${info.instruction}
            `;
            progress.style.display = 'none';
            button.textContent = '캡처 시작 (Space)';
            button.disabled = false;
            highlightDirection(step);
        } else if (step.endsWith('-collecting')) {
            const dir = step.replace('-collecting', '');
            const info = DIRECTION_LABELS[dir];
            instructions.innerHTML = `
                <strong style="color: #ffa502;">${info.icon} 수집 중...</strong><br>
                그 자세 유지!
            `;
            progress.style.display = 'block';
            progressBar.style.width = `${(count / MIN_SAMPLES) * 100}%`;
            sampleCount.textContent = `${count} / ${MIN_SAMPLES}`;
            button.textContent = '수집 중...';
            button.disabled = true;
        } else if (step === 'done') {
            instructions.innerHTML = `
                <strong style="color: #4CAF50; font-size: 28px;">🎉 캘리브레이션 완료!</strong><br>
                방향 감도가 저장되었습니다.
            `;
            progress.style.display = 'none';
            button.textContent = '완료';
            button.disabled = false;
            highlightDirection(null);
        }
    }

    function collectSamples(direction) {
        samples = [];
        currentStep = `${direction}-collecting`;
        updateUI(currentStep, 0);

        sampleInterval = setInterval(() => {
            if (lastHeadFrame) {
                samples.push({
                    nx: lastHeadFrame.nx,
                    ny: lastHeadFrame.ny,
                    yaw: lastHeadFrame.yawDeg,
                    pitch: lastHeadFrame.pitchDeg
                });
                updateUI(currentStep, samples.length);

                if (samples.length >= MIN_SAMPLES) {
                    clearInterval(sampleInterval);
                    sampleInterval = null;

                    // Calculate averages
                    const avgNx = samples.reduce((a, b) => a + b.nx, 0) / samples.length;
                    const avgNy = samples.reduce((a, b) => a + b.ny, 0) / samples.length;
                    const avgYaw = samples.reduce((a, b) => a + b.yaw, 0) / samples.length;
                    const avgPitch = samples.reduce((a, b) => a + b.pitch, 0) / samples.length;

                    calibrationData[direction] = { nx: avgNx, ny: avgNy, yaw: avgYaw, pitch: avgPitch };
                    console.log(`[MouthDirCal] ${direction}:`, calibrationData[direction]);

                    markDirectionComplete(direction);

                    // Move to next direction
                    const currentIndex = DIRECTIONS.indexOf(direction);
                    if (currentIndex < DIRECTIONS.length - 1) {
                        currentStep = DIRECTIONS[currentIndex + 1];
                        updateUI(currentStep);
                    } else {
                        finishCalibration();
                    }
                }
            }
        }, SAMPLE_INTERVAL_MS);
    }

    function finishCalibration() {
        // Calculate movement ranges
        const center = calibrationData.center;
        const ranges = {
            up: {
                dy: center.ny - calibrationData.up.ny,
                dpitch: center.pitch - calibrationData.up.pitch
            },
            down: {
                dy: calibrationData.down.ny - center.ny,
                dpitch: calibrationData.down.pitch - center.pitch
            },
            left: {
                dx: center.nx - calibrationData.left.nx,
                dyaw: center.yaw - calibrationData.left.yaw
            },
            right: {
                dx: calibrationData.right.nx - center.nx,
                dyaw: calibrationData.right.yaw - center.yaw
            }
        };

        const finalCalibration = {
            version: 1,
            center: center,
            directions: calibrationData,
            ranges: ranges,
            timestamp: Date.now()
        };

        console.log('[MouthDirCal] Final calibration:', finalCalibration);

        // Save to localStorage
        localStorage.setItem('mouthDirectionCalibration', JSON.stringify(finalCalibration));

        // Make available globally
        window.mouthDirectionCalibration = finalCalibration;

        currentStep = 'done';
        updateUI(currentStep);

        setTimeout(() => closeCalibration(), 2500);
    }

    function startCalibration() {
        // HeadTracker가 실행 중인지 확인
        if (typeof HeadTracker === 'undefined' || !HeadTracker.isActive()) {
            alert('Head tracking이 먼저 활성화되어야 합니다.');
            return;
        }

        console.log('[MouthDirCal] Starting direction calibration');

        // headFrame 이벤트 수신
        headFrameListener = (data) => {
            lastHeadFrame = data;
        };
        HeadTracker.on('headFrame', headFrameListener);

        calibrationData = {};
        calUI = createCalibrationUI();
        document.body.appendChild(calUI);

        const button = document.getElementById('mouth-dir-action');

        button.addEventListener('click', () => {
            if (currentStep === 'idle') {
                currentStep = 'center';
                updateUI(currentStep);
            } else if (DIRECTIONS.includes(currentStep)) {
                collectSamples(currentStep);
            } else if (currentStep === 'done') {
                closeCalibration();
            }
        });

        const handleKeydown = (e) => {
            if (e.code === 'Space') {
                e.preventDefault();
                button.click();
            } else if (e.code === 'Escape') {
                e.preventDefault();
                closeCalibration();
            }
        };

        document.addEventListener('keydown', handleKeydown);
        calUI.__keydownHandler = handleKeydown;

        currentStep = 'idle';
        updateUI(currentStep);
    }

    function closeCalibration() {
        console.log('[MouthDirCal] Closing calibration');

        if (sampleInterval) {
            clearInterval(sampleInterval);
            sampleInterval = null;
        }

        if (headFrameListener && typeof HeadTracker !== 'undefined') {
            HeadTracker.off('headFrame', headFrameListener);
            headFrameListener = null;
        }

        if (calUI) {
            if (calUI.__keydownHandler) {
                document.removeEventListener('keydown', calUI.__keydownHandler);
            }
            calUI.remove();
            calUI = null;
        }

        currentStep = 'idle';
        samples = [];
    }

    function getCalibration() {
        const saved = localStorage.getItem('mouthDirectionCalibration');
        if (saved) {
            return JSON.parse(saved);
        }
        return window.mouthDirectionCalibration || null;
    }

    // Keyboard shortcut: Alt+D
    document.addEventListener('keydown', (event) => {
        const code = event.code || '';
        if (event.altKey && !event.ctrlKey && !event.metaKey && code === 'KeyD') {
            event.preventDefault();
            event.stopPropagation();
            startCalibration();
        }
    }, true);

    return {
        start: startCalibration,
        close: closeCalibration,
        isActive: () => currentStep !== 'idle',
        getCalibration: getCalibration
    };
})();

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = MouthDirectionCalibration;
}
