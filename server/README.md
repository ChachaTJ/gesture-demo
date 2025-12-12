# B2TXT Backend Server

This directory contains the Python backend for the Smartphone Gesture Demo.

## Prerequisites
- Python 3.8+ or Conda environment `b2txt25_UI`
- Required packages: `flask`, `flask_socketio`, `numpy`, `scipy`, `h5py`, `g2p_en`, `whisper`

## Setup
**Important:** Large model files are NOT included in this repository.
1. Place your model files (`v4_model_1_final.pt`, etc.) in this directory.
2. Ensure `whisper.cpp` is configured if using voice features.

## Running
Use the `start_server.sh` script in the root directory to launch both the API server and the frontend.
