"""
API Server for CPU Phoneme Decoder
Exposes endpoints for decoding and batch processing.
"""
from flask import Flask, request, jsonify, render_template_string, render_template
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import os
import sys
import tempfile
import threading
from pathlib import Path
import h5py
import numpy as np
import scipy.io
import json
import whisper
import pronouncing

# HTML for Real-Time Demo UI (Connectivity Test)
HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>⚡ B2TXT Real-Time Demo</title>
    <meta charset="utf-8">
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <style>
        body { font-family: -apple-system, sans-serif; background: #222; color: #fff; padding: 20px; text-align: center; }
        .container { max-width: 800px; margin: 0 auto; }
        .status { padding: 15px; background: #333; margin: 20px 0; border-radius: 10px; }
        .active { color: #7bed9f; font-weight: bold; }
        
        .display-box { 
            background: #000; 
            border: 1px solid #444; 
            border-radius: 15px; 
            padding: 30px; 
            min-height: 200px;
            margin-top: 20px;
            text-align: left;
        }
        
        .label { font-size: 0.8em; color: #888; margin-bottom: 5px; text-transform: uppercase; letter-spacing: 1px; }
        
        #text-stream { 
            font-size: 1.8em; 
            color: #fff; 
            margin-bottom: 30px; 
            min-height: 1.2em;
            line-height: 1.4;
        }
        
        #phoneme-stream { font-family: 'Courier New', monospace; font-size: 1.2em; color: #7bed9f; }
        .p-tag { background: rgba(123, 237, 159, 0.15); padding: 4px 8px; margin: 0 4px 4px 0; border-radius: 6px; display: inline-block; border: 1px solid rgba(123, 237, 159, 0.3); }

        button { padding: 12px 24px; font-size: 1.1em; border-radius: 30px; border: none; background: #ff4757; color: white; cursor: pointer; transition: 0.2s; }
        button:hover { transform: scale(1.05); }
        button.recording { background: #fff; color: #ff4757; animation: pulse 1.5s infinite; }
        @keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(255,255,255,0.7); } 70% { box-shadow: 0 0 0 15px rgba(255,255,255,0); } 100% { box-shadow: 0 0 0 0 rgba(255,255,255,0); } }
    </style>
</head>
<body>
    <div class="container">
        <h1>⚡ B2TXT Real-Time Demo</h1>
        
        <div class="status">
            <span id="connection-status">🟡 Connecting...</span>
        </div>

        <button id="rec-btn" onclick="toggleMic()">🎤 Start Microphone</button>

        <div class="display-box">
            <div class="label">Recognized Sentence</div>
            <div id="text-stream"></div>
            
            <div class="label">Phoneme Stream (ARPAbet)</div>
            <div id="phoneme-stream"></div>
        </div>
    </div>

    <script>
        const socket = io(); 
        let isRecording = false;
        let context, processor, source;

        const btn = document.getElementById('rec-btn');
        const textDiv = document.getElementById('text-stream');
        const phonemeDiv = document.getElementById('phoneme-stream');
        const statusSpan = document.getElementById('connection-status');

        socket.on('connect', () => {
            statusSpan.innerText = "🟢 Connected (Ready)";
            statusSpan.className = "active";
        });

        socket.on('disconnect', () => {
            statusSpan.innerText = "🔴 Disconnected";
            statusSpan.className = "";
        });

        socket.on('result', (data) => {
            console.log(data);
            if (data.text) {
                textDiv.innerText = data.text;
            }
            if (data.phonemes) {
                phonemeDiv.innerHTML = data.phonemes.split(' ').map(p => 
                    `<span class="p-tag">${p}</span>`
                ).join('');
            }
        });

        async function toggleMic() {
            if (isRecording) {
                stopMic();
            } else {
                await startMic();
            }
        }

        async function startMic() {
            try {
                context = new (window.AudioContext || window.webkitAudioContext)({sampleRate: 16000});
                const stream = await navigator.mediaDevices.getUserMedia({audio: true});
                source = context.createMediaStreamSource(stream);
                processor = context.createScriptProcessor(4096, 1, 1);
                
                source.connect(processor);
                processor.connect(context.destination);
                
                processor.onaudioprocess = (e) => {
                    if (!isRecording) return;
                    const data = e.inputBuffer.getChannelData(0);
                    
                    // Visual confirmation of audio energy
                    let sum = 0;
                    for(let i=0; i<data.length; i++) sum += Math.abs(data[i]);
                    const avg = sum / data.length;
                    
                    if (avg > 0.01) {
                         document.getElementById('connection-status').innerText = "🎤 Sending Audio... Vol: " + avg.toFixed(3);
                         socket.emit('audio_raw', data.buffer);
                    }
                };

                isRecording = true;
                btn.innerText = "⏹ Stop Microphone";
                btn.classList.add('recording');
            } catch(e) {
                alert("Mic Error: " + e.message);
                document.getElementById('connection-status').innerText = "❌ Mic Error: " + e.message;
            }
        }

        function stopMic() {
            isRecording = false;
            if (source) source.disconnect();
            if (processor) processor.disconnect();
            if (context) context.close();
            
            btn.innerText = "🎤 Start Microphone";
            btn.classList.remove('recording');
        }
    </script>
</body>
</html>
'''

# Import Decoder
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cpu_decoder import CPUPhonemeDecoder

# Import Confusion-Aware Decoder
from confusion_decoder import ConfusionAwareDecoder

# Import Batch Processor function
from batch_processor import process_all_files

# Initialize Flask App
app = Flask(__name__, template_folder='.')
app.config['SECRET_KEY'] = 'b2txt_secret'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

print("Initializing decoder model...")
decoder = CPUPhonemeDecoder()
confusion_decoder = ConfusionAwareDecoder()
print("✓ API Server ready!\n")

import subprocess
import re
from g2p_en import G2p

# Initialize G2P
g2p = G2p()

# whisper.cpp stream configuration
# whisper.cpp stream configuration
WHISPER_CPP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "whisper")
WHISPER_MODEL = os.path.join(WHISPER_CPP_PATH, "models", "ggml-base.en-q5_1.bin")
WHISPER_STREAM_CMD = [
    f"{WHISPER_CPP_PATH}/stream",
    "-m", WHISPER_MODEL,
    "--step", "200",        # 0.2s update interval (200ms)
    "--length", "3000",     # 3s audio window
    "--keep", "200",        # Keep 0.2s from previous chunk
    "-t", "4",              # 4 threads
    "-l", "en",             # English language
    "--vad-thold", "0.7",   # Higher VAD threshold = less hallucinations
    "--freq-thold", "200"   # Filter low-frequency noise
]

# Global subprocess handle
whisper_process = None
whisper_thread = None
model_lock = threading.Lock()

def start_whisper_stream():
    """Start whisper.cpp stream subprocess and process output."""
    global whisper_process
    
    print("Starting whisper.cpp stream subprocess...")
    print(f"Command: {' '.join(WHISPER_STREAM_CMD)}")
    
    try:
        whisper_process = subprocess.Popen(
            WHISPER_STREAM_CMD,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            text=True,
            bufsize=1
        )
        print(f"✓ whisper.cpp stream started (PID: {whisper_process.pid})")
    except Exception as e:
        print(f"✗ Failed to start whisper.cpp: {e}")
        return
    
    # Read stdout (including merged stderr) in a thread
    def read_output():
        import re
        # Regex to strip ANSI escape codes (e.g., [2K, [0m, etc.)
        ansi_escape = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]|\x1b\[[\?]?[0-9;]*[a-zA-Z]')
        
        last_text = ""
        try:
            for line in whisper_process.stdout:
                # Strip ANSI escape codes first
                line = ansi_escape.sub('', line)
                line = line.strip()
                
                if not line:
                    continue
                    
                # Debug: print all output from whisper.cpp
                print(f"[whisper.cpp] {line}")
                
                # Filter out whisper.cpp status messages
                if line.startswith('[') or line.startswith('init:') or line.startswith('whisper') or line.startswith('ggml') or line.startswith('main:'):
                    continue
                
                # Filter common Whisper hallucinations (appears during silence)
                hallucinations = [
                    'thank you', 'thanks for watching', 'thanks for listening',
                    'see you next time', 'goodbye', 'bye', 'subscribe',
                    'like and subscribe', 'please subscribe', 'thank you for watching',
                    'you', 'the', 'a', 'i', 'it', 'is', 'and', 'to', 'of',
                    '...', '..', '.', '-', '--', '♪', '♫', '(music)',
                    '(footsteps)', '(keyboard clicking)', '(fire crackling)',
                    '(silence)', '(coughing)', '(breathing)', '(laughing)'
                ]
                if line.lower().strip() in hallucinations or len(line.strip()) < 3:
                    continue
                    
                # Skip duplicate outputs
                if line == last_text:
                    continue
                last_text = line
                
                # Convert to phonemes and emit via Socket.IO
                print(f"DEBUG: Transcription: '{line}'")
                phonemes = text_to_arpabet(line)
                print(f"DEBUG: Phonemes: '{phonemes}'")
                
                # Emit to all connected clients
                socketio.emit('result', {'text': line, 'phonemes': phonemes})
        except Exception as e:
            print(f"Error in whisper output thread: {e}")
        
        # Check if process exited
        retcode = whisper_process.poll()
        if retcode is not None:
            print(f"✗ whisper.cpp stream exited with code {retcode}")
    
    import threading
    global whisper_thread
    whisper_thread = threading.Thread(target=read_output, daemon=True)
    whisper_thread.start()

def stop_whisper_stream():
    """Stop whisper.cpp stream subprocess."""
    global whisper_process
    if whisper_process:
        whisper_process.terminate()
        whisper_process.wait()
        print("✓ whisper.cpp stream stopped")
        whisper_process = None


# User-defined Phoneme Set (matches Model Output)
LOGIT_TO_PHONEME = [
    'BLANK', 'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 
    'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 
    'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 
    'V', 'W', 'Y', 'Z', 'ZH', ' | '
]
VALID_PHONEMES = set([p for p in LOGIT_TO_PHONEME if p != 'BLANK' and p != ' | '])

def text_to_arpabet(text):
    """
    Convert text to ARPAbet phonemes using G2P-en (High Quality).
    Normalized to LOGIT_TO_PHONEME.
    Strips stress digits (e.g. AY1 -> AY).
    Inserts ' | ' for silence between words.
    """
    if not text: return ""
    
    # G2P-en conversion
    # Returns list like ['HH', 'AH0', 'L', 'OW1', ' ', 'W', 'ER1', 'L', 'D']
    raw_seq = g2p(text)
    
    phoneme_sequence = []
    
    for p in raw_seq:
        if p == ' ':
            # Add silence if previous wasn't silence
            if not phoneme_sequence or phoneme_sequence[-1] != '|':
                phoneme_sequence.append('|')
            continue
            
        # Strip digits (stress)
        p_clean = ''.join([c for c in p if not c.isdigit()])
        
        if p_clean in VALID_PHONEMES:
            phoneme_sequence.append(p_clean)
            
    # Remove trailing silence if exists
    if phoneme_sequence and phoneme_sequence[-1] == '|':
        phoneme_sequence.pop()
        
    result_str = ' '.join([p if p != '|' else ' | ' for p in phoneme_sequence])
    print(f"DEBUG: G2P-en: '{text}' -> '{result_str}'")
    return result_str

def decode_ascii_transcription(ascii_codes):
    """Convert numpy array of ASCII codes to string."""
    try:
        valid_codes = ascii_codes[ascii_codes != 0]
        chars = [chr(int(c)) for c in valid_codes]
        return "".join(chars).strip()
    except:
        return ""

# Phoneme ID to string mapping (from cpu_decoder.py)
PHONEME_MAP = {
    0: '<blank>',
    1: 'AA', 2: 'AE', 3: 'AH', 4: 'AO', 5: 'AW',
    6: 'AY', 7: 'B', 8: 'CH', 9: 'D', 10: 'DH',
    11: 'EH', 12: 'ER', 13: 'EY', 14: 'F', 15: 'G',
    16: 'HH', 17: 'IH', 18: 'IY', 19: 'JH', 20: 'K',
    21: 'L', 22: 'M', 23: 'N', 24: 'NG', 25: 'OW',
    26: 'OY', 27: 'P', 28: 'R', 29: 'S', 30: 'SH',
    31: 'T', 32: 'TH', 33: 'UH', 34: 'UW', 35: 'V',
    36: 'W', 37: 'Y', 38: 'Z', 39: 'ZH', 40: 'SIL', 41: 'SP'
}

def decode_phoneme_gt(seq_class_ids):
    """Convert seq_class_ids to phoneme string."""
    try:
        valid_ids = seq_class_ids[seq_class_ids > 0]
        phonemes = [PHONEME_MAP.get(int(p), f'?{p}') for p in valid_ids]
        # Remove consecutive duplicates
        deduped = []
        prev = None
        for p in phonemes:
            if p != prev:
                deduped.append(p)
            prev = p
        return ' '.join(deduped)
    except:
        return ""

def preprocess_mat_brain2voice(mat_data, trial_idx=0):
    """
    Preprocess brain-to-voice MAT file format to phoneme decoder format.
    Uses V4 method: threshcross + spikepow, avg downsample, z-score normalize.
    """
    spikepow = mat_data['spikepow_trials'][0, trial_idx]      # (T, 256)
    threshcross = mat_data['threshcross_trials'][0, trial_idx]  # (T, 256)
    
    # Combine: threshcross first (0-255), spikepow second (256-511)
    combined = np.concatenate([threshcross, spikepow], axis=1)  # (T, 512)
    
    # Downsample: 10ms -> 20ms (average every 2 frames)
    T = combined.shape[0] // 2
    downsampled = np.zeros((T, 512), dtype=np.float32)
    for i in range(T):
        downsampled[i] = (combined[2*i] + combined[2*i + 1]) / 2
    
    # Z-score normalize per channel
    downsampled = (downsampled - downsampled.mean(axis=0)) / (downsampled.std(axis=0) + 1e-8)
    
    return downsampled

@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model': 'loaded'})

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to decode a single HDF5 file.
    Returns phonemes AND ground truth for each trial.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    try:
        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
            
        try:
            results = []
            
            # Check if HDF5 file with trials
            if tmp_path.endswith(('.h5', '.hdf5')):
                with h5py.File(tmp_path, 'r') as f:
                    trial_keys = sorted([k for k in f.keys() if k.startswith('trial_')])
                    
                    for trial_key in trial_keys:
                        trial_grp = f[trial_key]
                        
                        # Get neural data
                        if 'input_features' not in trial_grp:
                            continue
                        neural_data = trial_grp['input_features'][:]
                        if len(neural_data.shape) == 2:
                            if neural_data.shape[1] == 512:
                                neural_data = neural_data[np.newaxis, ...]
                            elif neural_data.shape[0] == 512:
                                neural_data = neural_data.T[np.newaxis, ...]
                        
                        
                        # Get Ground Truth
                        ground_truth = ""
                        if 'transcription' in trial_grp:
                            ground_truth = decode_ascii_transcription(trial_grp['transcription'][()])
                        
                        # Get raw neural data for brain activity visualization
                        raw_features = trial_grp['input_features'][:]  # [T, 512]
                        total_frames = raw_features.shape[0]
                        
                        # Group 512 channels into 4 brain regions (128 each)
                        regions = ["dorsal6v", "area4", "ventral6v", "area55b"]
                        
                        # Build timeline (sample every 5 frames to reduce data size)
                        timeline = []
                        for frame_idx in range(0, total_frames, 5):
                            frame_data = raw_features[frame_idx]
                            # Split 512 channels into 4 regions of 128 each
                            activations = {
                                regions[0]: float(np.mean(np.abs(frame_data[0:128]))),
                                regions[1]: float(np.mean(np.abs(frame_data[128:256]))),
                                regions[2]: float(np.mean(np.abs(frame_data[256:384]))),
                                regions[3]: float(np.mean(np.abs(frame_data[384:512])))
                            }
                            timeline.append({
                                "frame": frame_idx,
                                "time_ms": frame_idx * 20,  # 20ms per frame
                                "activations": activations
                            })
                        
                        brain_activity = {
                            "sample_rate_ms": 20,
                            "total_frames": total_frames,
                            "regions": regions,
                            "timeline": timeline
                        }
                        
                        # Decode phonemes
                        decoded = decoder.decode(neural_data)
                        phoneme_string = decoded[0]['phoneme_string']
                        
                        # Get Phoneme Ground Truth
                        phoneme_gt = ""
                        if 'seq_class_ids' in trial_grp:
                            phoneme_gt = decode_phoneme_gt(trial_grp['seq_class_ids'][()])
                        
                        # Get per-phoneme confidence and confusion analysis
                        phoneme_list = phoneme_string.split()
                        rescored = confusion_decoder.rescore_with_context(phoneme_list)
                        
                        # Build confidences array
                        confidences = [round(r['confidence'], 3) for r in rescored]
                        
                        # Build confusion_analysis in requested format
                        avg_confidence = np.mean([r['confidence'] for r in rescored if r['phoneme'] != 'SIL'])
                        
                        potential_corrections = []
                        for i, r in enumerate(rescored):
                            if r['confidence'] < 0.9 and r['phoneme'] != 'SIL':
                                # Build alternatives as [[phoneme, probability], ...]
                                alternatives = [[r['phoneme'], round(r['confidence'], 3)]]  # Current as first
                                for alt_phoneme, alt_prob in r['alternatives'][:3]:
                                    alternatives.append([alt_phoneme, round(alt_prob, 3)])
                                
                                potential_corrections.append({
                                    'position': i,
                                    'current': r['phoneme'],
                                    'confidence': round(r['confidence'], 3),
                                    'alternatives': alternatives
                                })
                        
                        confusion_analysis = {
                            'original': phoneme_string,
                            'avg_confidence': round(float(avg_confidence), 3),
                            'low_confidence_count': len(potential_corrections),
                            'potential_corrections': potential_corrections
                        }
                        
                        results.append({
                            'trial_id': trial_key,
                            # Sentence Ground Truth - multiple field names for compatibility
                            'ground_truth': ground_truth,
                            'ground_truth_sentence': ground_truth,
                            'gt_sentence': ground_truth,
                            'reference_sentence': ground_truth,
                            'transcript': ground_truth,
                            # Phoneme Ground Truth - multiple field names for compatibility
                            'phoneme_gt': phoneme_gt,
                            'gt_phonemes': phoneme_gt,
                            'ground_truth_phonemes': phoneme_gt,
                            'reference_phonemes': phoneme_gt,
                            # Predicted phonemes
                            'predicted_phonemes': phoneme_string,
                            'phonemes': phoneme_string,
                            # Per-phoneme confidences
                            'confidences': confidences,
                            # Confusion analysis in requested format
                            'confusion_analysis': confusion_analysis,
                            # Flags for UI
                            'has_ground_truth': bool(ground_truth),
                            'has_phoneme_gt': bool(phoneme_gt),
                            # Brain activity for animation
                            'brain_activity': brain_activity
                        })
            # Handle MAT files (brain-to-voice format)
            elif tmp_path.endswith('.mat'):
                try:
                    mat_data = sio.loadmat(tmp_path)
                    
                    # Check if it's brain-to-voice format
                    if 'spikepow_trials' in mat_data and 'threshcross_trials' in mat_data:
                        num_trials = mat_data['spikepow_trials'].shape[1]
                        sentences = mat_data.get('sentences', ['' for _ in range(num_trials)])
                        
                        for trial_idx in range(num_trials):
                            # Preprocess with V4 method
                            neural_data = preprocess_mat_brain2voice(mat_data, trial_idx)
                            input_data = neural_data[np.newaxis, ...]  # (1, T, 512)
                            
                            # Decode
                            decoded = decoder.decode(input_data)
                            phoneme_string = decoded[0]['phoneme_string']
                            
                            # Get sentence GT
                            sentence_gt = sentences[trial_idx].strip() if trial_idx < len(sentences) else ''
                            
                            results.append({
                                'trial_id': f'trial_{trial_idx:04d}',
                                'ground_truth': sentence_gt,
                                'gt_sentence': sentence_gt,
                                'predicted_phonemes': phoneme_string,
                                'phonemes': phoneme_string,
                                'has_ground_truth': bool(sentence_gt),
                                'has_phoneme_gt': False,  # MAT files don't have phoneme GT
                                'format': 'brain2voice_mat'
                            })
                    else:
                        # Regular MAT file - use original decoder
                        data = decoder.load_file(tmp_path)
                        decoded = decoder.decode(data)
                        results = decoded
                except Exception as e:
                    return jsonify({'error': f'MAT file error: {str(e)}'}), 500
            else:
                # Other file types - use original decoder
                data = decoder.load_file(tmp_path)
                decoded = decoder.decode(data)
                results = decoded
            
            return jsonify({
                'count': len(results),
                'results': results
            })
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch', methods=['POST'])
def run_batch():
    """
    Trigger batch processing in a background thread.
    Returns immediately.
    """
    def run_thread():
        print("Starting batch processing via API...")
        process_all_files() 
        print("Batch processing finished.")
        
    thread = threading.Thread(target=run_thread)
    thread.start()
    
    return jsonify({
        'message': 'Batch processing started', 
        'output_file': 'outputs/predictions_dataset.csv'
    })

@app.route('/generate_sentence', methods=['POST'])
def generate_sentence():
    """
    Generate sentence(s) from phoneme string using LLM.
    Can generate for primary phonemes and alternatives.
    
    Request body:
    {
        "phonemes": "AY SIL L AY K SIL DH AE T SIL",
        "alternatives": ["AY SIL L IY K SIL DH AE T SIL", ...]  // optional
    }
    
    Returns:
    {
        "primary_sentence": "I like that",
        "confidence": 0.92,
        "alternatives": [
            {"phonemes": "...", "sentence": "...", "score": 0.85}
        ]
    }
    """
    try:
        import anthropic
        
        data = request.get_json()
        if not data or 'phonemes' not in data:
            return jsonify({'error': 'Missing phonemes field'}), 400
        
        phonemes = data['phonemes']
        
        # Configure Claude
        api_key = os.environ.get('ANTHROPIC_API_KEY', 'INSERT_API_KEY')
        client = anthropic.Anthropic(api_key=api_key)
        
        # Get UI data with per-phoneme confidence
        ui_data = confusion_decoder.get_ui_response(phonemes, include_alternatives=True)
        
        # Build detailed prompt with confidence info
        phoneme_list = phonemes.split()
        rescored = confusion_decoder.rescore_with_context(phoneme_list)
        
        # Format phonemes with confidence
        phoneme_with_conf = []
        for i, r in enumerate(rescored):
            if r['phoneme'] == 'SIL':
                phoneme_with_conf.append('SIL')
            else:
                conf_pct = int(r['confidence'] * 100)
                if conf_pct < 90:
                    phoneme_with_conf.append(f"{r['phoneme']}*({conf_pct}%)")
                else:
                    phoneme_with_conf.append(f"{r['phoneme']}({conf_pct}%)")
        
        phonemes_formatted = ' '.join(phoneme_with_conf)
        
        # Build low-confidence details
        low_conf_details = []
        for lc in ui_data['low_confidence_phonemes']:
            alts = [f"{a['phoneme']}({int(a['probability']*100)}%)" for a in lc['alternatives'][:3]]
            low_conf_details.append(
                f"- Position {lc['position']}: {lc['phoneme']} ({int(lc['confidence']*100)}% confidence) - alternatives: {', '.join(alts)}"
            )
        
        confusion_patterns = """Known confusion patterns:
- D ↔ T, B ↔ P, G ↔ K, V ↔ F (voiced/unvoiced): 15% confusion
- IY ↔ IH, EH ↔ AH, OW ↔ UW (vowels): 10% confusion"""
        
        prompt = f'''You are helping decode speech from a brain-computer interface for an ALS patient.

PHONEME SEQUENCE (with confidence %):
{phonemes_formatted}

OVERALL CONFIDENCE: {int(ui_data['confidence'] * 100)}%

LOW-CONFIDENCE PHONEMES:
{chr(10).join(low_conf_details) if low_conf_details else "None - all phonemes are high confidence"}

{confusion_patterns}

INSTRUCTIONS:

STEP 1: GENERATE PRIMARY SENTENCE (Strict Acoustic Fidelity)
- Construct a sentence that matches the provided phoneme sequence EXACTLY.
- Do NOT try to "fix" grammar or context if it conflicts with the high-confidence phonemes.
- Prioritize acoustic accuracy over semantic meaning for this step.

STEP 2: GENERATE 10 ALTERNATIVE CANDIDATES (Context & Error Correction)
- Generate 10 different possible sentences considering:
  1. Acoustic Similarity: Swap low-confidence phonemes with likely alternatives (e.g., based on the confusion patterns).
  2. Language Probability: Use common phrases and colloquialisms that fit the partial phonemes.
  3. Contextual Coherence: Ensure the sentence makes grammatical and semantic sense.
- Explore different interpretations of the ambiguous parts.

STEP 3: SELECT TOP 3 ALTERNATIVES
- Review the 10 candidates from Step 2.
- Select the best 3 based on a balance of:
  - How well they match the phoneme backbone (especially high-confidence parts).
  - How natural and likely they are in daily conversation.
- Rank them as ALT1, ALT2, ALT3.

OUTPUT FORMAT:
Return ONLY the final result in this EXACT format (no reasoning text):
PRIMARY: [Strict acoustic match]
ALT1: [Best alternative]
ALT2: [Second best alternative]
ALT3: [Third best alternative]'''
        
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=300,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        response_text = message.content[0].text.strip()
        
        # Parse response
        lines = response_text.split('\n')
        primary_sentence = ""
        alternatives = []
        
        for line in lines:
            line = line.strip()
            if line.startswith('PRIMARY:'):
                primary_sentence = line.replace('PRIMARY:', '').strip().strip('"').strip("'")
            elif line.startswith('ALT'):
                alt_sentence = line.split(':', 1)[1].strip().strip('"').strip("'") if ':' in line else ""
                if alt_sentence:
                    alternatives.append(alt_sentence)  # Simple string, not object
        
        # Fallback if parsing fails
        if not primary_sentence:
            primary_sentence = response_text.split('\n')[0].strip()
        
        result = {
            'phonemes': phonemes,
            'phonemes_with_confidence': phonemes_formatted,
            'primary_sentence': primary_sentence,
            'confidence': ui_data['confidence'],
            'low_confidence_phonemes': ui_data['low_confidence_phonemes'],
            'alternatives': alternatives[:3]  # Simple string array
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Audio endpoint for demo mode
audio_converter = None

@app.route('/predict_audio', methods=['POST'])
def predict_audio():
    """
    Demo mode: Convert audio to phonemes and generate sentence.
    Accepts audio file upload (wav, mp3, etc.)
    
    Returns same format as /predict for UI compatibility.
    """
    global audio_converter
    
    if 'file' not in request.files:
        return jsonify({'error': 'No audio file'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Lazy load the audio converter (heavy model)
        if audio_converter is None:
            from audio_demo import AudioPhonemeConverter
            audio_converter = AudioPhonemeConverter()
        
        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
        
        try:
            import librosa
            audio, sr = librosa.load(tmp_path, sr=16000)
            
            # Convert to phonemes
            arpabet, ipa = audio_converter.audio_to_phonemes(audio)
            
            # Get confidence analysis
            phoneme_list = arpabet.split()
            rescored = confusion_decoder.rescore_with_context(phoneme_list)
            confidences = [round(r['confidence'], 3) for r in rescored]
            avg_confidence = np.mean([r['confidence'] for r in rescored if r['phoneme'] != 'SIL'])
            
            # Build confusion analysis
            potential_corrections = []
            for i, r in enumerate(rescored):
                if r['confidence'] < 0.9 and r['phoneme'] != 'SIL':
                    alternatives = [[r['phoneme'], round(r['confidence'], 3)]]
                    for alt_phoneme, alt_prob in r['alternatives'][:3]:
                        alternatives.append([alt_phoneme, round(alt_prob, 3)])
                    potential_corrections.append({
                        'position': i,
                        'current': r['phoneme'],
                        'confidence': round(r['confidence'], 3),
                        'alternatives': alternatives
                    })
            
            result = {
                'trial_id': 'audio_demo',
                'predicted_phonemes': arpabet,
                'phonemes': arpabet,
                'ipa_phonemes': ipa,
                'confidences': confidences,
                'confusion_analysis': {
                    'original': arpabet,
                    'avg_confidence': round(float(avg_confidence), 3),
                    'low_confidence_count': len(potential_corrections),
                    'potential_corrections': potential_corrections
                },
                'has_ground_truth': False,
                'has_phoneme_gt': False,
                'is_demo_mode': True
            }
            
            return jsonify({
                'count': 1,
                'results': [result],
                'mode': 'audio_demo'
            })
            
        finally:
            os.unlink(tmp_path)
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- Real-Time Streaming Socket Events ---

@socketio.on('connect')
def handle_connect():
    print('Client connected to real-time stream')
    emit('status', {'message': 'Connected to B2TXT Server'})

@socketio.on('audio_raw')
def handle_raw_audio(array_buffer):
    """Legacy handler - whisper.cpp now captures audio directly."""
    # Audio is now captured by whisper.cpp subprocess, not from browser
    pass

@socketio.on('start_stream')
def handle_start_stream():
    """Start whisper.cpp stream on client request."""
    print("Client requested stream start")
    start_whisper_stream()
    emit('status', {'message': 'whisper.cpp stream started'})

@socketio.on('stop_stream')
def handle_stop_stream():
    """Stop whisper.cpp stream on client request."""
    print("Client requested stream stop")
    stop_whisper_stream()
    emit('status', {'message': 'whisper.cpp stream stopped'})

# --- Head Tracking Socket Events ---

@socketio.on('gaze_data')
def handle_gaze_data(data):
    """Receive gaze coordinates from HeadTracker."""
    x = data.get('x')
    y = data.get('y')
    ts = data.get('timestamp')
    # Broadcast to all clients (for multi-device sync)
    socketio.emit('gaze_update', {'x': x, 'y': y, 'timestamp': ts})
    # Log for debugging (uncomment if needed)
    # print(f"[Gaze] x={x:.0f}, y={y:.0f}")

@socketio.on('mouth_click')
def handle_mouth_click(data):
    """Receive mouth-open click events from HeadTracker."""
    ts = data.get('timestamp')
    print(f"[MouthClick] ts={ts}")
    # Broadcast to all clients
    socketio.emit('mouth_click_event', {'timestamp': ts})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    print(f"Starting API Server on port {port}...")
    
    # Start whisper.cpp stream subprocess
    print("Starting whisper.cpp stream (Metal GPU)...")
    start_whisper_stream()
    
    # Use socketio.run instead of app.run
    try:
        socketio.run(app, host='0.0.0.0', port=port, debug=False, allow_unsafe_werkzeug=True)
    finally:
        stop_whisper_stream()
