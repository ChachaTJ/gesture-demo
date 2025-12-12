
import sys
import os

print("Testing imports...")
try:
    import google.generativeai as genai
    print("SUCCESS: google-generativeai imported")
except ImportError as e:
    print(f"FAILURE: google-generativeai failed: {e}")
except Exception as e:
    print(f"FAILURE: google-generativeai unexpected error: {e}")

try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from cpu_decoder import CPUPhonemeDecoder
    print("SUCCESS: cpu_decoder imported")
except ImportError as e:
    print(f"FAILURE: cpu_decoder failed: {e}")
except Exception as e:
    print(f"FAILURE: cpu_decoder unexpected error: {e}")
