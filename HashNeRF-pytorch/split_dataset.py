"""
This script can split the dataset into Train, Test, Validation
"""
import os
import json

folder = 'data/lab'
images = os.path.join(folder, 'images')
transform = os.path.join(folder, 'transform.json')
