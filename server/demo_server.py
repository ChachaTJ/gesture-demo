"""
Fast Local Audio Demo - Using Whisper Tiny
Fastest CPU inference for real-time demos
"""
from flask import Flask, render_template_string, request, jsonify
from flask_cors import CORS
import numpy as np
import tempfile
import os
import whisper
import pronouncing

app = Flask(__name__)
CORS(app)

# Load Whisper tiny model (fast on CPU)
model = None

def get_model():
    global model
    if model is None:
        print("Loading Whisper tiny model...")
        model = whisper.load_model("tiny")  # 39MB, very fast
        print("✓ Model loaded!")
    return model

def text_to_arpabet(text):
    """Convert text to ARPAbet phonemes."""
    words = text.lower().split()
    phonemes = []
    
    for word in words:
        phones = pronouncing.phones_for_word(word)
        if phones:
            phonemes.append(phones[0])
        else:
            for char in word:
                if char.isalpha():
                    phonemes.append(char.upper())
    
    return ' SIL '.join(phonemes) + ' SIL' if phonemes else 'SIL'

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>🎤 Fast Audio Demo</title>
    <meta charset="utf-8">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        .container {
            background: white;
            border-radius: 20px;
            padding: 40px;
            max-width: 800px;
            width: 100%;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        h1 {
            text-align: center;
            color: #667eea;
            margin-bottom: 10px;
            font-size: 32px;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
            font-size: 14px;
        }
        .record-btn {
            width: 200px;
            height: 200px;
            border-radius: 50%;
            border: none;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-size: 24px;
            font-weight: bold;
            cursor: pointer;
            margin: 0 auto;
            display: block;
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
            transition: all 0.3s ease;
        }
        .record-btn:hover {
            transform: scale(1.05);
        }
        .record-btn.recording {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.1); }
        }
        .status {
            text-align: center;
            margin: 20px 0;
            font-size: 18px;
            color: #666;
            min-height: 30px;
        }
        .results {
            margin-top: 30px;
            display: none;
        }
        .results.show { display: block; }
        .result-section {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 15px;
        }
        .result-label {
            font-weight: bold;
            color: #667eea;
            margin-bottom: 8px;
            font-size: 14px;
            text-transform: uppercase;
        }
        .result-content {
            font-size: 18px;
            color: #333;
            word-break: break-word;
        }
        .phoneme-box {
            display: inline-block;
            padding: 8px 12px;
            margin: 4px;
            background: white;
            border: 2px solid #667eea;
            border-radius: 8px;
            font-family: 'Courier New', monospace;
            font-weight: bold;
            color: #667eea;
        }
        .error {
            background: #fee;
            color: #c00;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>⚡ Fast Audio Demo</h1>
        <p class="subtitle">Whisper Tiny - Local CPU (~1-2s)</p>

        <button id="recordBtn" class="record-btn">
            🎙️<br>Record
        </button>

        <div id="status" class="status"></div>

        <div id="results" class="results">
            <div class="result-section">
                <div class="result-label">Recognized Text</div>
                <div id="text" class="result-content"></div>
            </div>
            
            <div class="result-section">
                <div class="result-label">Phonemes</div>
                <div id="phonemes" class="result-content"></div>
            </div>
        </div>

        <div id="error" class="error" style="display:none;"></div>
    </div>

    <script>
        let mediaRecorder;
        let audioChunks = [];
        const recordBtn = document.getElementById('recordBtn');
        const status = document.getElementById('status');
        const results = document.getElementById('results');
        const errorDiv = document.getElementById('error');

        recordBtn.addEventListener('click', async () => {
            if (!mediaRecorder || mediaRecorder.state === 'inactive') {
                startRecording();
            }
        });

        async function startRecording() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ 
                    audio: {
                        channelCount: 1,
                        sampleRate: 16000
                    }
                });
                
                mediaRecorder = new MediaRecorder(stream, {
                    mimeType: 'audio/webm;codecs=opus'
                });
                audioChunks = [];

                mediaRecorder.ondataavailable = (event) => {
                    audioChunks.push(event.data);
                };

                mediaRecorder.onstop = async () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                    await processAudio(audioBlob);
                    stream.getTracks().forEach(track => track.stop());
                };

                mediaRecorder.start();
                recordBtn.classList.add('recording');
                recordBtn.innerHTML = '⏸️<br>Recording...';
                status.textContent = '🔴 Recording for 3 seconds...';
                results.classList.remove('show');
                errorDiv.style.display = 'none';

                setTimeout(() => {
                    if (mediaRecorder && mediaRecorder.state === 'recording') {
                        mediaRecorder.stop();
                        recordBtn.classList.remove('recording');
                        recordBtn.innerHTML = '🎙️<br>Record';
                        status.textContent = '⚡ Processing...';
                    }
                }, 3000);

            } catch (err) {
                showError('Microphone access denied: ' + err.message);
            }
        }

        async function processAudio(audioBlob) {
            const formData = new FormData();
            formData.append('audio', audioBlob, 'recording.webm');

            try {
                const response = await fetch('/convert', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (data.error) {
                    showError(data.error);
                    return;
                }

                // Display results
                status.textContent = '✅ Done!';
                results.classList.add('show');

                document.getElementById('text').textContent = data.text;
                
                const phonemes = data.arpabet.split(' ');
                document.getElementById('phonemes').innerHTML = 
                    phonemes.map(p => `<span class="phoneme-box">${p}</span>`).join('');

            } catch (err) {
                showError('Error: ' + err.message);
            }
        }

        function showError(message) {
            errorDiv.textContent = '❌ ' + message;
            errorDiv.style.display = 'block';
            status.textContent = '';
            recordBtn.innerHTML = '🎙️<br>Record';
            recordBtn.classList.remove('recording');
        }
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/convert', methods=['POST'])
def convert():
    """Convert audio to text using Whisper, then to phonemes."""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file'}), 400
    
    audio_file = request.files['audio']
    
    try:
        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmp:
            audio_file.save(tmp.name)
            tmp_path = tmp.name
        
        try:
            # Load model
            whisper_model = get_model()
            
            # Transcribe (Whisper handles format conversion internally)
            print("Transcribing...")
            result = whisper_model.transcribe(tmp_path, language='en', fp16=False)
            text = result['text'].strip()
            print(f"Text: {text}")
            
            if not text:
                return jsonify({
                    'text': '(no speech)',
                    'arpabet': 'SIL'
                })
            
            # Convert to phonemes
            arpabet = text_to_arpabet(text)
            
            return jsonify({
                'text': text,
                'arpabet': arpabet,
                'model': 'whisper-tiny'
            })
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'text': 'ERROR',
            'arpabet': 'ERROR'
        }), 500

if __name__ == '__main__':
    print("=" * 60)
    print("⚡ Fast Audio Demo - Whisper Tiny (Local)")
    print("=" * 60)
    print("\nLoading model...")
    get_model()  # Preload
    print("\n📍 http://localhost:5002")
    print("⚡ CPU inference: ~1-2 seconds")
    print("\nPress Ctrl+C to stop\n")
    
    app.run(host='0.0.0.0', port=5002, debug=False)
