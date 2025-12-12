print("Start debug_batch.py")
import os
import sys
print("Imports os/sys done")
import google.generativeai as genai
print("Import genai done")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from cpu_decoder import CPUPhonemeDecoder
    print("Import CPUPhonemeDecoder done")
except Exception as e:
    print(f"Import CPUPhonemeDecoder failed: {e}")

print("End debug_batch.py")
