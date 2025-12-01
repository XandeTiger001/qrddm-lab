#!/usr/bin/env python3
import subprocess
import sys
import os
from threading import Thread
import time

def start_ml_server():
    subprocess.run([sys.executable, "src/local_server.py"], cwd=os.path.dirname(__file__))

def start_honeypot():
    subprocess.run([sys.executable, "src/honeypot.py"], cwd=os.path.dirname(__file__))

if __name__ == "__main__":
    print("Starting ML Detection Server and Honeypot...")
    
    ml_thread = Thread(target=start_ml_server, daemon=True)
    honeypot_thread = Thread(target=start_honeypot, daemon=True)
    
    ml_thread.start()
    honeypot_thread.start()
    
    print("Servers starting... Press Ctrl+C to stop")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down servers...")