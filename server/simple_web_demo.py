import sys
import os
import time
import threading
import numpy as np
import cv2
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

# Add EyeGestures to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'EyeGestures'))

try:
    from eyeGestures.utils import VideoCapture
    from eyeGestures import EyeGestures_v3
    EYE_GESTURES_AVAILABLE = True
except ImportError as e:
    print(f"EyeGestures import failed: {e}")
    EYE_GESTURES_AVAILABLE = False

app = Flask(__name__, template_folder='.')
app.config['SECRET_KEY'] = 'simple_demo'
socketio = SocketIO(app, cors_allowed_origins="*")

gaze_thread = None
gaze_running = False

def gaze_tracking_loop():
    global gaze_running
    
    if not EYE_GESTURES_AVAILABLE:
        print("EyeGestures not available, cannot start tracking.")
        return

    print("Starting Eye Tracking Loop...")
    gestures = EyeGestures_v3()
    cap = VideoCapture(0)
    
    # Virtual screen size for normalization
    VIRTUAL_W, VIRTUAL_H = 1920, 1080
    
    # Create calibration points
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

        # Process frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.flip(frame, axis=1)

        calibrate_mode = (iterator < n_points)
        
        try:
            event, cevent = gestures.step(
                frame, 
                calibrate_mode, 
                VIRTUAL_W, 
                VIRTUAL_H,
                context="web_gaze"
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
                    
                    # Normalize calibration point
                    data["calib_point"] = {
                        "x": cevent.point[0] / VIRTUAL_W,
                        "y": cevent.point[1] / VIRTUAL_H
                    }
            elif event:
                # Normalize gaze point
                data["x"] = event.point[0] / VIRTUAL_W
                data["y"] = event.point[1] / VIRTUAL_H
                data["blink"] = event.blink
            
            socketio.emit('gaze_data', data)
            
        except Exception as e:
            print(f"Gaze Error: {e}")
            
        time.sleep(0.03) # ~30fps

    cap.release()
    print("Eye Tracking Stopped.")

@app.route('/')
def index():
    return render_template('simple_web_demo.html')

@socketio.on('start_gaze')
def handle_start_gaze():
    global gaze_thread, gaze_running
    if not gaze_running:
        gaze_running = True
        gaze_thread = threading.Thread(target=gaze_tracking_loop)
        gaze_thread.daemon = True
        gaze_thread.start()
        print("Gaze tracking started")

@socketio.on('stop_gaze')
def handle_stop_gaze():
    global gaze_running
    gaze_running = False
    print("Gaze tracking stopped")

if __name__ == '__main__':
    print("Starting Simple EyeGestures Web Demo...")
    print("Open http://localhost:5005 in your browser")
    socketio.run(app, host='0.0.0.0', port=5005, debug=True, allow_unsafe_werkzeug=True)
