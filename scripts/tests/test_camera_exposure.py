#!/usr/bin/env python3
"""
Test script to verify camera exposure settings are properly applied and persist.
"""

import time
import sys
sys.path.insert(0, '/mnt/data2/droid/droid')

try:
    import pyzed.sl as sl
except ModuleNotFoundError:
    print("ERROR: pyzed not installed")
    sys.exit(1)

from droid.camera_utils.wrappers.multi_camera_wrapper import MultiCameraWrapper

def test_camera_exposure():
    print("=" * 60)
    print("Testing Camera Exposure Settings")
    print("=" * 60)
    
    # Create camera wrapper
    print("\n1. Initializing camera wrapper...")
    camera_wrapper = MultiCameraWrapper()
    
    # Check if wrist camera exists
    if '18482824' not in camera_wrapper.camera_dict:
        print("ERROR: Wrist camera 18482824 not found!")
        print(f"Available cameras: {list(camera_wrapper.camera_dict.keys())}")
        return
    
    print("\n2. Verifying initial settings...")
    settings = camera_wrapper.verify_camera_settings('18482824')
    
    # Read a few frames to ensure settings persist
    print("\n3. Capturing 5 frames to verify settings persist...")
    for i in range(5):
        obs, timestamps = camera_wrapper.read_cameras()
        time.sleep(0.1)
        if i % 2 == 0:
            print(f"   Frame {i+1}/5 captured")
    
    print("\n4. Re-verifying settings after frame capture...")
    settings_after = camera_wrapper.verify_camera_settings('18482824')
    
    # Check if settings changed
    if settings['exposure'] == settings_after['exposure'] and settings['aec_agc'] == 0:
        print("\n✓ SUCCESS: Exposure settings are stable!")
        print(f"   - AEC_AGC remains: {settings_after['aec_agc']} (manual mode)")
        print(f"   - Exposure remains: {settings_after['exposure']}")
    else:
        print("\n✗ FAILURE: Exposure settings changed!")
        print(f"   - Initial: AEC_AGC={settings['aec_agc']}, Exposure={settings['exposure']}")
        print(f"   - After: AEC_AGC={settings_after['aec_agc']}, Exposure={settings_after['exposure']}")
    
    # Test reconfiguration persistence
    print("\n5. Testing reconfiguration (calling set_trajectory_mode again)...")
    camera_wrapper.set_trajectory_mode()
    
    print("\n6. Verifying settings after reconfiguration...")
    settings_reconfig = camera_wrapper.verify_camera_settings('18482824')
    
    if settings_reconfig['exposure'] == settings['exposure'] and settings_reconfig['aec_agc'] == 0:
        print("\n✓ SUCCESS: Settings persist through reconfiguration!")
    else:
        print("\n✗ FAILURE: Settings lost after reconfiguration!")
        print(f"   - Expected: AEC_AGC=0, Exposure={settings['exposure']}")
        print(f"   - Got: AEC_AGC={settings_reconfig['aec_agc']}, Exposure={settings_reconfig['exposure']}")
    
    # Cleanup
    print("\n7. Cleaning up...")
    camera_wrapper.disable_cameras()
    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)

if __name__ == "__main__":
    test_camera_exposure()

