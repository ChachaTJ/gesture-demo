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
The **Phoneme Decoder Model** (`v4_model_1_final.pt`, ~1.4GB) is too large for GitHub and is **NOT included**.
1.  **Download/Locate**: `v4_model_1_final.pt`
2.  **Placement**: Put it inside the `server/` directory.

### ✅ Included Components
*   **Whisper Model**: `ggml-base.en-q5_1.bin` is included in `server/whisper/models/`.
*   **Whisper Binary**: A macOS (Apple Silicon) compatible `stream` binary is included in `server/whisper/`.
    *   *Note for Non-Mac users*: You may need to recompile `whisper.cpp` and replace the `stream` binary.

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

### Inspiration & Core Logic
*   **Nutshell**: Hands-free browsing extension (Chrome Built-in AI Hackathon).
    *   The head tracking cursor implementation, dwell interactions, and calibration logic in this demo are adapted from the **Nutshell** project.
    *   **Repository**: [https://github.com/tanhanwei/Nutshell](https://github.com/tanhanwei/Nutshell)
