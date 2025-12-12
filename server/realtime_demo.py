"""
Real-Time Raw PCM Streaming -> Phonemes + Eye Gaze Tracking
Uses direct float32 sampling from AudioContext to bypass WebM container issues.
Integrates EyeGestures for gaze tracking.
"""
from flask import Flask, render_template_string, render_template
from flask_socketio import SocketIO, emit
import numpy as np
import whisper
import pronouncing
import io
import struct
import threading
import cv2
import sys
import os
import time

# EyeGestures 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'EyeGestures'))

try:
    from eyeGestures.utils import VideoCapture
    from eyeGestures import EyeGestures_v3
    EYE_GESTURES_AVAILABLE = True
except ImportError:
    print("EyeGestures not found. Gaze tracking disabled.")
    EYE_GESTURES_AVAILABLE = False

app = Flask(__name__, template_folder='.')
app.config['SECRET_KEY'] = 'demo'
socketio = SocketIO(app, cors_allowed_origins="*")

model = None
gaze_thread = None
gaze_running = False

def get_model():
    global model
    if model is None:
        print("Loading Whisper base.en...")
        model = whisper.load_model("base.en")  # Better accuracy than tiny
        print("✓ Loaded!")
    return model

def gaze_tracking_loop():
    """Background thread for eye tracking"""
    global gaze_running
    
    if not EYE_GESTURES_AVAILABLE:
        return

    print("Starting Eye Tracking Loop...")
    gestures = EyeGestures_v3()
    cap = VideoCapture(0)
    
    # 가상 스크린 크기 (정규화된 좌표 계산용)
    VIRTUAL_W, VIRTUAL_H = 1920, 1080
    
    # 캘리브레이션 포인트 생성
    x = np.arange(0.1, 1.0, 0.4)
    y = np.arange(0.1, 1.0, 0.4)
    xx, yy = np.meshgrid(x, y)
    calibration_map = np.column_stack([xx.ravel(), yy.ravel()])
    n_points = len(calibration_map)
    np.random.shuffle(calibration_map)
    
    gestures.uploadCalibrationMap(calibration_map, context="web_gaze")
    gestures.setFixation(1.0)

    iterator = 0
    prev_x, prev_y = 0, 0
    
    while gaze_running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.flip(frame, axis=1)

        calibrate_mode = (iterator < n_points)
        
        try:
            event, cevent = gestures.step(
                frame, 
                "web_gaze",
                calibrate_mode, 
                VIRTUAL_W, 
                VIRTUAL_H
            )

            data = {
                "calibrating": calibrate_mode,
                "x": 0, "y": 0,
                "calib_point": None,
                "calib_progress": f"{iterator}/{n_points}"
            }

            if calibrate_mode:
                if cevent and cevent.point is not None:
                    if cevent.point[0] != prev_x or cevent.point[1] != prev_y:
                        iterator += 1
                        prev_x, prev_y = cevent.point[0], cevent.point[1]
                    
                    # 정규화된 좌표 (0.0 ~ 1.0)로 변환하여 전송
                    data["calib_point"] = {
                        "x": cevent.point[0] / VIRTUAL_W,
                        "y": cevent.point[1] / VIRTUAL_H
                    }
            elif event:
                # 시선 좌표 정규화
                data["x"] = event.point[0] / VIRTUAL_W
                data["y"] = event.point[1] / VIRTUAL_H
                data["blink"] = event.blink
                
                # v3 specific: fixation and saccades
                if hasattr(event, 'fixation'):
                    data["fixation"] = bool(event.fixation)
                if hasattr(event, 'saccades'):
                    data["saccades"] = bool(event.saccades)
            
            # 웹소켓으로 전송 (모든 클라이언트에게)
            socketio.emit('gaze_data', data)
            
        except Exception as e:
            print(f"Gaze Error: {e}")
            
        time.sleep(0.03) # ~30fps

    cap.release()
    print("Eye Tracking Stopped.")

def text_to_arpabet(text):
    """Convert text to ARPAbet phonemes."""
    if not text:
        return ""
    words = text.lower().split()
    phonemes = []
    for word in words:
        phones = pronouncing.phones_for_word(word)
        if phones:
            # Phones comes like "HH AH0 L OW1"
            # We want to strip numbers to match LOGIT_TO_PHONEME (e.g., "AH0" -> "AH")
            raw_phones = phones[0].split()
            clean_phones = [''.join([c for c in p if not c.isdigit()]) for p in raw_phones]
            phonemes.extend(clean_phones)
        else:
            for char in word:
                if char.isalpha():
                    # Fallback for unknown words
                    phonemes.append(char.upper())
    return ' '.join(phonemes) if phonemes else ''

@app.route('/')
def index():
    return render_template('eyegestures_demo.html')

@socketio.on('start_gaze')
def handle_start_gaze():
    global gaze_thread, gaze_running
    if not gaze_running:
        gaze_running = True
        gaze_thread = threading.Thread(target=gaze_tracking_loop)
        gaze_thread.daemon = True
        gaze_thread.start()
        print("Gaze tracking started via WebSocket request")

@socketio.on('stop_gaze')
def handle_stop_gaze():
    global gaze_running
    gaze_running = False
    print("Gaze tracking stopped via WebSocket request")

@socketio.on('audio_raw')
def handle_raw_audio(array_buffer):
    """Handle raw float32 audio stream."""
    try:
        # data comes as binary buffer, float32 from JS
        # Convert to numpy array
        audio_data = np.frombuffer(array_buffer, dtype=np.float32)
        
        # Debug print every ~20 chunks to avoid spam but confirm life
        if np.random.rand() < 0.05: 
            print(f"Received audio chunk: {len(audio_data)} samples")

        if len(audio_data) > 0:
            mdl = get_model()
            
            # Transcribe raw audio buffer
            # Whisper `transcribe` can take numpy array directly
            result = mdl.transcribe(audio_data, language='en', fp16=False)
            text = result['text'].strip()
            
            if text:
                phonemes = text_to_arpabet(text)
                # Broadcast=True makes sure the External App (listening only) also gets the data
                emit('result', {'text': text, 'phonemes': phonemes}, broadcast=True)
                if len(text) > 1:
                    print(f"Detected: {text}")

    except Exception as e:
        print(f"Process Error: {e}")

if __name__ == '__main__':
    print("=" * 60)
    print("⚡ Real-Time Phoneme Engine (Port 5003)")
    print("=" * 60)
    get_model()
    print("\n📍 http://localhost:5003")
    print("✓ Raw Audio Streaming Enabled")
    
    # Broadcast ensures all clients (demo page + external app) get the results
    socketio.run(app, host='0.0.0.0', port=5003, debug=False)
