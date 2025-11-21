#!/usr/bin/env python3
"""
Simple script to monitor training progress by checking model file updates
"""
import os
import time
from datetime import datetime

model_path = "mini_suno.pth"
check_interval = 30  # seconds

print("Monitoring training progress...")
print("=" * 60)

last_modified = None
check_count = 0

try:
    while True:
        if os.path.exists(model_path):
            current_modified = os.path.getmtime(model_path)
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            mod_time = datetime.fromtimestamp(current_modified).strftime('%Y-%m-%d %H:%M:%S')
            
            if last_modified is None:
                print(f"Initial model state:")
                print(f"  Last modified: {mod_time}")
                print(f"  Size: {size_mb:.2f} MB")
                last_modified = current_modified
            elif current_modified > last_modified:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Model updated!")
                print(f"  Last modified: {mod_time}")
                print(f"  Size: {size_mb:.2f} MB")
                print(f"  -> Training is making progress (better loss achieved)")
                last_modified = current_modified
            else:
                check_count += 1
                if check_count % 10 == 0:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Still training... (checked {check_count} times)")
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Model file not found yet...")
        
        time.sleep(check_interval)
        
except KeyboardInterrupt:
    print("\n\nMonitoring stopped.")
    print(f"Total checks: {check_count}")



