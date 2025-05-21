#!/usr/bin/env python3
"""
Script to verify fast.ai installation
"""

import sys
import fastai
import torch
import torchvision

print(f"Python version: {sys.version}")
print(f"fast.ai version: {fastai.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")

# Check we can import all the modules we need
try:
    from fastai.vision.all import *
    print("Successfully imported fastai.vision.all")
    
    from fastai.text.all import *
    print("Successfully imported fastai.text.all")
    
    from fastai.tabular.all import *
    print("Successfully imported fastai.tabular.all")
    
    print("\nAll fast.ai modules imported successfully!")
    print("Installation looks good!")
except Exception as e:
    print(f"Error importing modules: {e}")