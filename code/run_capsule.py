""" top level run script """
import sys
import os 
from pathlib import Path
import subprocess


def run():

    # define the path to the data
    data_path = os.path.abspath("/data")
    data_folders = os.listdir(data_path)

    # preprocess the data
    for folder in data_folders:
        subprocess.run(['python', 'preprocess_1.py', '--folder', str(folder)])
if __name__ == "__main__": 
    run()
