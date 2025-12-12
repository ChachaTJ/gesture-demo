# Smartphone Gesture Demo & B2TXT Server

A web-based smartphone accessibility demo controlled by head tracking, gaze, and voice phonemes.

## Live Demo
*   **Main Demo**: [https://chachatj.github.io/gesture-demo/](https://chachatj.github.io/gesture-demo/)
*   **Gaze Tracking Demo**: [https://chachatj.github.io/gesture-demo/gaze_demo.html](https://chachatj.github.io/gesture-demo/gaze_demo.html)

## Features
*   **Head Tracking**: Control the cursor with head movements.
*   **Gaze Interaction**: Dwell-click and smart targeting.
*   **Voice Control**: Phoneme-based command triggers (requires backend).
*   **Elastic Long Press**: Enhanced touch interaction simulation.

---

## 🛠 Backend Server Setup
The `server/` directory contains the Python backend for voice processing and phoneme decoding.

### ⚠️ Missing Model Files
Due to GitHub's file size limits (100MB+), the following model files are **NOT included** in this repository. You must manually download/place them for the server to function correctly:

1.  **Phoneme Decoder Model** (`v4_model_1_final.pt`)
    *   **Placement**: Put this file inside the `server/` directory.
    *   **Usage**: Required for `CPUPhonemeDecoder` to load the trained model.

2.  **Whisper Model** (`ggml-base.en-q5_1.bin`)
    *   **Placement**: External `whisper.cpp` directory (as configured in `api_server.py`).
    *   **Note**: The server script currently points to a local path. You may need to update `WHISPER_CPP_PATH` in `server/api_server.py`.

### Running the Server
1.  Navigate to the root directory.
2.  Run the helper script:
    ```bash
    ./start_server.sh
    ```
    (Ensure you have the required Python environment/packages installed).

---

## 🙌 Credits & References

### WebGPU Head Tracking
*   **Library**: [Human.js](https://github.com/vladmandic/human) by Vladimir Mandic.
*   **Description**: This project utilizes `human.js` for high-performance, WebGPU-accelerated face and iris tracking directly in the browser. It enables the precise head cursor and gaze interactions without server latency.
