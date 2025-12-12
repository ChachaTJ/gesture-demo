print("Start debug_batch_full.py")
import os
import sys
import glob
print("glob imported")
import h5py
print("h5py imported")
import numpy as np
print("numpy imported")
import pandas as pd
print("pandas imported")
from pathlib import Path
from tqdm import tqdm
print("tqdm imported")
import google.generativeai as genai
print("genai imported")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cpu_decoder import CPUPhonemeDecoder
print("decoder imported")

print("End debug_batch_full.py")
