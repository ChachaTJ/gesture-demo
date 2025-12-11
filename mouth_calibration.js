/**
 * 👄 Mouth Click Calibration UI
 * 입 벌림 클릭 캘리브레이션
 * 
 * Usage:
 *   MouthCalibration.start();  // Alt+M 로도 트리거 가능
 */

const MouthCalibration = (function () {
    'use strict';

    const MIN_SAMPLES = 20;
    const SAMPLE_INTERVAL_MS = 100;

    let calUI = null;
    let currentStep = 'idle';
    let samples = [];
    let sampleInterval = null;
    let mouthRatioListener = null;
    let lastMouthRatio = 0;

    function createCalibrationUI() {
        const overlay = document.createElement('div');
        overlay.id = 'mouth-cal-overlay';
        overlay.style.cssText = `
            position: fixed;
            top: 0; left: 0;
            width: 100%; height: 100%;
            background: rgba(0, 0, 0, 0.9);
            z-index: 2147483647;
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        `;

        const panel = document.createElement('div');
        panel.style.cssText = `
            background: #2a2a2a;
            border-radius: 16px;
            padding: 48px;
            max-width: 600px;
            text-align: center;
            color: white;
            box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        `;

        panel.innerHTML = `
            <div style="font-size: 48px; margin-bottom: 16px;">👄</div>
            <h2 style="margin: 0 0 16px 0; font-size: 28px; font-weight: 600;">입 벌림 클릭 캘리브레이션</h2>
            <p id="mouth-cal-instructions" style="font-size: 18px; line-height: 1.6; margin: 0 0 32px 0; color: #ccc;">
                "시작"을 클릭하여 입 벌림 감지를 캘리브레이션합니다.<br>
                입을 열고 닫는 두 가지 상태를 캡처합니다.
            </p>
            <div id="mouth-cal-progress" style="display: none; margin-bottom: 24px;">
                <div style="font-size: 64px; font-weight: bold; color: #8b5a3c;" id="mouth-cal-count">0</div>
                <div style="font-size: 14px; color: #999;">샘플 수집됨</div>
            </div>
            <button id="mouth-cal-action" style="
                background: #8b5a3c;
                color: white;
                border: none;
                padding: 16px 48px;
                font-size: 18px;
                font-weight: 600;
                border-radius: 8px;
                cursor: pointer;
                transition: all 0.2s;
            ">캘리브레이션 시작</button>
            <div style="margin-top: 24px; font-size: 14px; color: #999;">ESC로 취소</div>
        `;

        overlay.appendChild(panel);
        return overlay;
    }

    function updateUI(step, count = 0) {
        const instructions = document.getElementById('mouth-cal-instructions');
        const progress = document.getElementById('mouth-cal-progress');
        const countEl = document.getElementById('mouth-cal-count');
        const button = document.getElementById('mouth-cal-action');

        if (!instructions || !button) return;

        countEl.textContent = count;

        if (step === 'start') {
            instructions.innerHTML = `
                <strong style="font-size: 24px; color: #8b5a3c;">Step 1: 입 벌리기</strong><br><br>
                입을 크게 벌리세요 ("아~~" 하듯이)<br>
                Space 또는 버튼을 클릭하세요.
            `;
            progress.style.display = 'block';
            button.textContent = '입 열림 캡처';
            button.disabled = false;
            button.style.background = '#8b5a3c';
            button.style.cursor = 'pointer';
        } else if (step === 'open-collecting') {
            instructions.innerHTML = `
                <strong style="font-size: 24px; color: #4CAF50;">입을 열고 유지하세요!</strong><br><br>
                샘플 수집 중... ${count}/${MIN_SAMPLES}
            `;
            button.textContent = '수집 중...';
            button.disabled = true;
            button.style.background = '#666';
            button.style.cursor = 'not-allowed';
        } else if (step === 'open-done') {
            instructions.innerHTML = `
                <strong style="font-size: 24px; color: #4CAF50;">✓ 입 열림 캡처 완료!</strong><br><br>
                <strong style="font-size: 24px; color: #8b5a3c;">Step 2: 입 다물기</strong><br><br>
                입을 편안하게 다무세요.<br>
                Space 또는 버튼을 클릭하세요.
            `;
            button.textContent = '입 다문 상태 캡처';
            button.disabled = false;
            button.style.background = '#8b5a3c';
            button.style.cursor = 'pointer';
        } else if (step === 'closed-collecting') {
            instructions.innerHTML = `
                <strong style="font-size: 24px; color: #4CAF50;">입을 다물고 유지하세요!</strong><br><br>
                샘플 수집 중... ${count}/${MIN_SAMPLES}
            `;
            button.textContent = '수집 중...';
            button.disabled = true;
            button.style.background = '#666';
            button.style.cursor = 'not-allowed';
        } else if (step === 'done') {
            instructions.innerHTML = `
                <strong style="font-size: 32px; color: #4CAF50;">🎉 캘리브레이션 완료!</strong><br><br>
                입 벌림 클릭이 활성화되었습니다.<br>
                입을 벌려서 클릭을 트리거하세요!
            `;
            progress.style.display = 'none';
            button.textContent = '완료';
            button.style.background = '#4CAF50';
            button.disabled = false;
            button.style.cursor = 'pointer';
        }
    }

    function collectSamples(type) {
        samples = [];
        currentStep = `${type}-collecting`;
        updateUI(currentStep, 0);

        sampleInterval = setInterval(() => {
            if (lastMouthRatio > 0) {
                samples.push(lastMouthRatio);
                updateUI(currentStep, samples.length);

                if (samples.length >= MIN_SAMPLES) {
                    clearInterval(sampleInterval);
                    sampleInterval = null;

                    const avg = samples.reduce((a, b) => a + b, 0) / samples.length;
                    console.log(`[MouthCalibration] ${type} average: ${avg.toFixed(3)}`);

                    if (type === 'open') {
                        window.__mouthCalOpen = avg;
                        currentStep = 'open-done';
                        updateUI(currentStep);
                    } else {
                        window.__mouthCalClosed = avg;
                        finishCalibration();
                    }
                }
            }
        }, SAMPLE_INTERVAL_MS);
    }

    function finishCalibration() {
        const openRatio = window.__mouthCalOpen;
        const closedRatio = window.__mouthCalClosed;

        // 70% threshold between closed and open
        const threshold = closedRatio + (openRatio - closedRatio) * 0.7;

        const calibration = {
            version: 1,
            closedRatio,
            openRatio,
            threshold,
            timestamp: Date.now()
        };

        console.log('[MouthCalibration] Calibration complete:', calibration);

        // HeadTracker에 입 캘리브레이션 저장
        if (typeof HeadTracker !== 'undefined') {
            HeadTracker.setMouthCalibration(calibration);
        }

        currentStep = 'done';
        updateUI(currentStep);

        setTimeout(() => closeCalibration(), 2000);
    }

    function startCalibration() {
        // HeadTracker가 실행 중인지 확인
        if (typeof HeadTracker === 'undefined' || !HeadTracker.isActive()) {
            alert('Head tracking이 먼저 활성화되어야 합니다.');
            return;
        }

        console.log('[MouthCalibration] Starting mouth calibration');

        // mouthRatio 이벤트 수신
        mouthRatioListener = (ratio) => {
            lastMouthRatio = ratio;
        };
        HeadTracker.on('mouthRatio', mouthRatioListener);

        calUI = createCalibrationUI();
        document.body.appendChild(calUI);

        const button = document.getElementById('mouth-cal-action');

        button.addEventListener('click', () => {
            if (currentStep === 'idle' || currentStep === 'start') {
                currentStep = 'open';
                collectSamples('open');
            } else if (currentStep === 'open-done') {
                currentStep = 'closed';
                collectSamples('closed');
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

        currentStep = 'start';
        updateUI(currentStep);
    }

    function closeCalibration() {
        console.log('[MouthCalibration] Closing calibration');

        if (sampleInterval) {
            clearInterval(sampleInterval);
            sampleInterval = null;
        }

        if (mouthRatioListener && typeof HeadTracker !== 'undefined') {
            HeadTracker.off('mouthRatio', mouthRatioListener);
            mouthRatioListener = null;
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

    // Keyboard shortcut: Alt+M
    document.addEventListener('keydown', (event) => {
        const code = event.code || '';
        if (event.altKey && !event.ctrlKey && !event.metaKey && code === 'KeyM') {
            event.preventDefault();
            event.stopPropagation();
            startCalibration();
        }
    }, true);

    return {
        start: startCalibration,
        close: closeCalibration,
        isActive: () => currentStep !== 'idle'
    };
})();

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = MouthCalibration;
}
