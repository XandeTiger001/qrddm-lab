#!/usr/bin/env python3
import subprocess
import time
import sys
import os
from threading import Thread

def start_ml_server():
    """Start the ML detection server in background"""
    try:
        subprocess.run([sys.executable, "src/local_server.py"], 
                      cwd=os.path.dirname(__file__), check=True)
    except KeyboardInterrupt:
        pass

def start_honeypot_server():
    """Start the honeypot server in background"""
    try:
        subprocess.run([sys.executable, "src/honeypot.py"], 
                      cwd=os.path.dirname(__file__), check=True)
    except KeyboardInterrupt:
        pass

def run_tests():
    """Run the test suite"""
    print("Starting servers...")
    ml_server_thread = Thread(target=start_ml_server, daemon=True)
    honeypot_thread = Thread(target=start_honeypot_server, daemon=True)
    
    ml_server_thread.start()
    honeypot_thread.start()
    
    # Wait for servers to start
    time.sleep(3)
    
    try:
        print("Running ML detection tests...")
        subprocess.run([sys.executable, "tests/test_attacks.py"], check=True)
        
        print("\nRunning honeypot tests...")
        subprocess.run([sys.executable, "tests/test_honeypot.py"], check=True)
        
        print("\n✅ All tests completed!")
        print("\nPress Ctrl+C to stop servers...")
        
        # Keep main thread alive to allow servers to run
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Test error: {e}")

if __name__ == "__main__":
    run_tests()