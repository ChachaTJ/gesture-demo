"""Web UI for CPU Phoneme Decoder"""
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import os, sys, tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cpu_decoder import CPUPhonemeDecoder

app = Flask(__name__)
CORS(app)

print("Initializing decoder...")
decoder = CPUPhonemeDecoder()
print("✓ Web UI ready!\n")

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Neural Phoneme Decoder (CPU)</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; padding: 20px; }
.container { max-width: 900px; margin: 0 auto; }
.header { text-align: center; color: white; margin-bottom: 40px; }
.header h1 { font-size: 3em; margin-bottom: 10px; }
.badge { background: #4CAF50; color: white; padding: 8px 16px; border-radius: 20px; display: inline-block; margin-top: 10px; }
.card { background: white; border-radius: 20px; padding: 40px; box-shadow: 0 20px 60px rgba(0,0,0,0.3); margin-bottom: 20px; }
.upload-zone { border: 3px dashed #667eea; border-radius: 15px; padding: 60px 20px; text-align: center; cursor: pointer; transition: all 0.3s; background: #f8f9ff; }
.upload-zone:hover { background: #f0f2ff; border-color: #764ba2; }
.upload-zone.dragover { background: #e8ebff; transform: scale(1.02); }
.upload-icon { font-size: 4em; margin-bottom: 20px; color: #667eea; }
.processing { text-align: center; padding: 40px; display: none; }
.spinner { border: 4px solid #f3f3f3; border-top: 4px solid #667eea; border-radius: 50%; width: 60px; height: 60px; animation: spin 1s linear infinite; margin: 0 auto 20px; }
@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
.results { display: none; }
.result-item { background: #f8f9ff; border-radius: 10px; padding: 20px; margin-bottom: 15px; border-left: 4px solid #667eea; }
.sample-id { font-weight: bold; color: #667eea; font-size: 1.1em; margin-bottom: 15px; }
.phoneme-sequence { font-family: 'Courier New', monospace; font-size: 1.1em; padding: 15px; background: white; border-radius: 8px; word-wrap: break-word; }
.stats { margin-top: 10px; color: #666; font-size: 0.9em; }
</style>
</head>
<body>
<div class="container">
<div class="header">
<h1>🧠 Neural Phoneme Decoder</h1>
<p>Your trained 92% model • CPU Mode</p>
<div class="badge">✓ No GPU Required</div>
</div>
<div class="card">
<div class="upload-zone" id="uploadZone">
<div class="upload-icon">📁</div>
<div style="font-size:1.3em;">Drag & Drop your file</div>
<div style="color:#666;margin-top:10px;">CSV, HDF5 (.h5), MATLAB (.mat)</div>
</div>
<input type="file" id="fileInput" style="display:none" accept=".csv,.h5,.hdf5,.mat">
<div class="processing" id="processing">
<div class="spinner"></div>
<div style="color:#667eea;font-size:1.2em;">Decoding on CPU...</div>
</div>
</div>
<div class="card results" id="results">
<h2 style="margin-bottom:20px;color:#667eea;">Results</h2>
<div id="resultsList"></div>
</div>
</div>
<script>
const uploadZone=document.getElementById('uploadZone'),fileInput=document.getElementById('fileInput'),processing=document.getElementById('processing'),results=document.getElementById('results'),resultsList=document.getElementById('resultsList');
uploadZone.addEventListener('click',()=>fileInput.click());
fileInput.addEventListener('change',e=>{if(e.target.files.length>0)uploadFile(e.target.files[0])});
uploadZone.addEventListener('dragover',e=>{e.preventDefault();uploadZone.classList.add('dragover')});
uploadZone.addEventListener('dragleave',()=>uploadZone.classList.remove('dragover'));
uploadZone.addEventListener('drop',e=>{e.preventDefault();uploadZone.classList.remove('dragover');if(e.dataTransfer.files.length>0)uploadFile(e.dataTransfer.files[0])});
async function uploadFile(file){results.style.display='none';uploadZone.style.display='none';processing.style.display='block';const formData=new FormData();formData.append('file',file);try{const response=await fetch('/decode',{method:'POST',body:formData}),data=await response.json();data.success?displayResults(data.results):alert('Error: '+data.error)}catch(error){alert('Error: '+error.message)}processing.style.display='none';uploadZone.style.display='block'}
function displayResults(data){resultsList.innerHTML='';data.forEach(r=>{const item=document.createElement('div');item.className='result-item';item.innerHTML=`<div class="sample-id">Sample ${r.sample_id+1}</div><div class="phoneme-sequence">${r.phoneme_string}</div><div class="stats">${r.num_phonemes} phonemes</div>`;resultsList.appendChild(item)});results.style.display='block'}
</script>
</body>
</html>"""

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/decode', methods=['POST'])
def decode():
    try:
        file = request.files['file']
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
        try:
            data = decoder.load_file(tmp_path)
            results = decoder.decode(data)
            return jsonify({'success': True, 'results': results})
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("="*80)
    print("WEB UI RUNNING ON http://localhost:5000")
    print("="*80 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
