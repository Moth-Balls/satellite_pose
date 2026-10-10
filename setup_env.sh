#!/bin/bash

# Create enviroment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install opencv-python ultralytics

