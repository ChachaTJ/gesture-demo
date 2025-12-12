from RealtimeSTT import AudioToTextRecorder
import numpy as np

try:
    print("Initializing Recorder...")
    # Initialize with input_device_index=None or similar if attempting to avoid mic? 
    # Or just check if method 'feed_audio' exists.
    recorder = AudioToTextRecorder(model="tiny", language="en", spinner=False)
    
    if hasattr(recorder, "feed_audio"):
        print("Has feed_audio method.")
    else:
        print("No feed_audio method found.")

    print("Test Complete.")
except Exception as e:
    print(f"Error: {e}")
