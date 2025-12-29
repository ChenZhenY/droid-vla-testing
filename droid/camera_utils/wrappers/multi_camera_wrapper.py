import os
import random
from collections import defaultdict

from droid.camera_utils.camera_readers.zed_camera import gather_zed_cameras
from droid.camera_utils.info import get_camera_type

class MultiCameraWrapper:
    def __init__(self, camera_kwargs={}):
        # Open Cameras #
        zed_cameras = gather_zed_cameras()
        self.camera_dict = {cam.serial_number: cam for cam in zed_cameras}

        # Set Correct Parameters #
        for cam_id in self.camera_dict.keys():
            cam_type = get_camera_type(cam_id)
            curr_cam_kwargs = camera_kwargs.get(cam_type, {})
            self.camera_dict[cam_id].set_reading_parameters(**curr_cam_kwargs)

        # Launch Camera #
        self.set_trajectory_mode()

        # ### TODO: hack for debugging - Set persistent manual exposure for wrist camera ###
        try:
            import pyzed.sl as sl
        except ModuleNotFoundError:
            print("WARNING: You have not setup the ZED cameras, and currently cannot use them")
            return
        # Configure persistent manual exposure for wrist camera (survives reconfiguration)
        if '18482824' in self.camera_dict:
            # NOTE: if you want fixed exposure
            self.camera_dict['18482824'].configure_persistent_settings({
                sl.VIDEO_SETTINGS.AEC_AGC: 0,      # Disable auto-exposure (CRITICAL!)
                sl.VIDEO_SETTINGS.EXPOSURE: 10,    # Set manual exposure value
            })
            # self.camera_dict['18482824'].configure_persistent_settings({
            #     sl.VIDEO_SETTINGS.AEC_AGC: 1
            # })
            # Verify settings
            if self.camera_dict['18482824'].is_running():
                aec_agc = self.camera_dict['18482824'].get_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC)
                exposure = self.camera_dict['18482824'].get_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE)
                print(f"✓ Camera 18482824 - AEC_AGC: {aec_agc}, Exposure: {exposure}")
        # ##############

    ### Calibration Functions ###
    def get_camera(self, camera_id):
        return self.camera_dict[camera_id]

    def enable_advanced_calibration(self):
        for cam in self.camera_dict.values():
            cam.enable_advanced_calibration()

    def disable_advanced_calibration(self):
        for cam in self.camera_dict.values():
            cam.disable_advanced_calibration()

    def set_calibration_mode(self, cam_id):
        # If High Res Calibration, Only One Can Run #
        close_all = any([cam.high_res_calibration for cam in self.camera_dict.values()])

        if close_all:
            for curr_cam_id in self.camera_dict:
                if curr_cam_id != cam_id:
                    self.camera_dict[curr_cam_id].disable_camera()

        self.camera_dict[cam_id].set_calibration_mode()

    def set_trajectory_mode(self):
        # If High Res Calibration, Close All #
        close_all = any(
            [cam.high_res_calibration and cam.current_mode == "calibration" for cam in self.camera_dict.values()]
        )

        if close_all:
            for cam in self.camera_dict.values():
                cam.disable_camera()

        # Put All Cameras In Trajectory Mode #
        for cam in self.camera_dict.values():
            cam.set_trajectory_mode()

    ### Data Storing Functions ###
    def start_recording(self, recording_folderpath):
        subdir = os.path.join(recording_folderpath, "SVO")
        if not os.path.isdir(subdir):
            os.makedirs(subdir)
        for cam in self.camera_dict.values():
            filepath = os.path.join(subdir, cam.serial_number + ".svo")
            cam.start_recording(filepath)

    def stop_recording(self):
        for cam in self.camera_dict.values():
            cam.stop_recording()

    ### Basic Camera Functions ###
    def read_cameras(self):
        full_obs_dict = defaultdict(dict)
        full_timestamp_dict = {}

        # Read Cameras In Randomized Order #
        all_cam_ids = list(self.camera_dict.keys())
        random.shuffle(all_cam_ids)

        for cam_id in all_cam_ids:
            if not self.camera_dict[cam_id].is_running():
                continue

            # try:
            data_dict, timestamp_dict = self.camera_dict[cam_id].read_camera()
            

            for key in data_dict:
                full_obs_dict[key].update(data_dict[key])
            full_timestamp_dict.update(timestamp_dict)

        return full_obs_dict, full_timestamp_dict

    def disable_cameras(self):
        for camera in self.camera_dict.values():
            camera.disable_camera()

    def verify_camera_settings(self, cam_id):
        """Verify current camera settings for debugging."""
        try:
            import pyzed.sl as sl
        except ModuleNotFoundError:
            print("WARNING: pyzed not available")
            return
        
        if cam_id not in self.camera_dict:
            print(f"Camera {cam_id} not found")
            return
        
        cam = self.camera_dict[cam_id]
        if not cam.is_running():
            print(f"Camera {cam_id} is not running")
            return
        
        aec_agc = cam.get_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC)
        exposure = cam.get_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE)
        gain = cam.get_camera_settings(sl.VIDEO_SETTINGS.GAIN)
        brightness = cam.get_camera_settings(sl.VIDEO_SETTINGS.BRIGHTNESS)
        
        print(f"\n=== Camera {cam_id} Settings ===")
        print(f"AEC_AGC (auto-exposure): {aec_agc} (0=manual, 1=auto)")
        print(f"Exposure: {exposure}")
        print(f"Gain: {gain}")
        print(f"Brightness: {brightness}")
        print("================================\n")
        
        return {
            'aec_agc': aec_agc,
            'exposure': exposure,
            'gain': gain,
            'brightness': brightness
        }
