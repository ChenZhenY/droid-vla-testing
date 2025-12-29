# ruff: noqa

"""
Dry run inference script that replays dataset observations through the policy server
and compares predicted actions with ground truth actions.

Usage:
python scripts/dry_run_inference.py \
    --dataset_path data/success/2025-12-24/Wed_Dec_24_15:34:57_2025 \
    --external_camera left \
    --remote_host 0.0.0.0 \
    --remote_port 8000
"""

import contextlib
import copy
import dataclasses
import datetime
import enum
import glob
import json
import os
import signal
import time
from collections import defaultdict
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
from moviepy.editor import ImageSequenceClip
from openpi_client import image_tools
from openpi_client import websocket_client_policy
from PIL import Image
import tqdm
import tyro


# DROID data collection frequency
DROID_CONTROL_FREQUENCY = 15


##########################################################################################################
################ Data loading functions copied from convert_droid_data_to_lerobot.py ####################
##########################################################################################################

camera_type_dict = {
    "hand_camera_id": 0,
    "varied_camera_1_id": 1,
    "varied_camera_2_id": 1,
}

camera_type_to_string_dict = {
    0: "hand_camera",
    1: "varied_camera",
    2: "fixed_camera",
}


def get_camera_type(cam_id):
    if cam_id not in camera_type_dict:
        return None
    type_int = camera_type_dict[cam_id]
    return camera_type_to_string_dict[type_int]


class MP4Reader:
    def __init__(self, filepath, serial_number):
        self.serial_number = serial_number
        self._index = 0
        self._mp4_reader = cv2.VideoCapture(filepath)
        if not self._mp4_reader.isOpened():
            raise RuntimeError("Corrupted MP4 File")

    def set_reading_parameters(
        self,
        image=True,
        concatenate_images=False,
        resolution=(0, 0),
        resize_func=None,
    ):
        self.image = image
        self.concatenate_images = concatenate_images
        self.resolution = resolution
        self.resize_func = cv2.resize
        self.skip_reading = not image
        if self.skip_reading:
            return

    def get_frame_count(self):
        if self.skip_reading:
            return 0
        return int(self._mp4_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    def set_frame_index(self, index):
        if self.skip_reading:
            return

        if index < self._index:
            self._mp4_reader.set(cv2.CAP_PROP_POS_FRAMES, index - 1)
            self._index = index

        while self._index < index:
            self.read_camera(ignore_data=True)

    def _process_frame(self, frame):
        frame = copy.deepcopy(frame)
        if self.resolution == (0, 0):
            return frame
        return self.resize_func(frame, self.resolution)

    def read_camera(self, ignore_data=False, correct_timestamp=None):
        if self.skip_reading:
            return {}

        success, frame = self._mp4_reader.read()

        self._index += 1
        if not success:
            return None
        if ignore_data:
            return None

        data_dict = {}

        if self.concatenate_images or "stereo" not in self.serial_number:
            data_dict["image"] = {self.serial_number: self._process_frame(frame)}
        else:
            single_width = frame.shape[1] // 2
            data_dict["image"] = {
                self.serial_number + "_left": self._process_frame(frame[:, :single_width, :]),
                self.serial_number + "_right": self._process_frame(frame[:, single_width:, :]),
            }

        return data_dict

    def disable_camera(self):
        if hasattr(self, "_mp4_reader"):
            self._mp4_reader.release()


class RecordedMultiCameraWrapper:
    def __init__(self, recording_folderpath, camera_kwargs={}):
        self.camera_kwargs = camera_kwargs

        mp4_filepaths = glob.glob(recording_folderpath + "/*.mp4")
        all_filepaths = mp4_filepaths

        self.camera_dict = {}
        for f in all_filepaths:
            serial_number = f.split("/")[-1][:-4]
            cam_type = get_camera_type(serial_number)
            camera_kwargs.get(cam_type, {})

            if f.endswith(".mp4"):
                Reader = MP4Reader
            else:
                raise ValueError

            self.camera_dict[serial_number] = Reader(f, serial_number)

    def read_cameras(self, index=None, camera_type_dict={}, timestamp_dict={}):
        full_obs_dict = defaultdict(dict)

        all_cam_ids = list(self.camera_dict.keys())

        for cam_id in all_cam_ids:
            if "stereo" in cam_id:
                continue
            try:
                cam_type = camera_type_dict[cam_id]
            except KeyError:
                print(f"{self.camera_dict} -- {camera_type_dict}")
                raise ValueError(f"Camera type {cam_id} not found in camera_type_dict")
            curr_cam_kwargs = self.camera_kwargs.get(cam_type, {})
            self.camera_dict[cam_id].set_reading_parameters(**curr_cam_kwargs)

            timestamp = timestamp_dict.get(cam_id + "_frame_received", None)
            if index is not None:
                self.camera_dict[cam_id].set_frame_index(index)

            data_dict = self.camera_dict[cam_id].read_camera(correct_timestamp=timestamp)

            if data_dict is None:
                return None
            for key in data_dict:
                full_obs_dict[key].update(data_dict[key])

        return full_obs_dict


def get_hdf5_length(hdf5_file, keys_to_ignore=[]):
    length = None

    for key in hdf5_file:
        if key in keys_to_ignore:
            continue

        curr_data = hdf5_file[key]
        if isinstance(curr_data, h5py.Group):
            curr_length = get_hdf5_length(curr_data, keys_to_ignore=keys_to_ignore)
        elif isinstance(curr_data, h5py.Dataset):
            curr_length = len(curr_data)
        else:
            raise ValueError

        if length is None:
            length = curr_length
        assert curr_length == length

    return length


def load_hdf5_to_dict(hdf5_file, index, keys_to_ignore=[]):
    data_dict = {}

    for key in hdf5_file:
        if key in keys_to_ignore:
            continue

        curr_data = hdf5_file[key]
        if isinstance(curr_data, h5py.Group):
            data_dict[key] = load_hdf5_to_dict(curr_data, index, keys_to_ignore=keys_to_ignore)
        elif isinstance(curr_data, h5py.Dataset):
            data_dict[key] = curr_data[index]
        else:
            raise ValueError

    return data_dict


class TrajectoryReader:
    def __init__(self, filepath, read_images=True):
        self._hdf5_file = h5py.File(filepath, "r")
        is_video_folder = "observations/videos" in self._hdf5_file
        self._read_images = read_images and is_video_folder
        self._length = get_hdf5_length(self._hdf5_file)
        self._video_readers = {}
        self._index = 0

    def length(self):
        return self._length

    def read_timestep(self, index=None, keys_to_ignore=[]):
        if index is None:
            index = self._index
        else:
            assert not self._read_images
            self._index = index
        assert index < self._length

        keys_to_ignore = [*keys_to_ignore.copy(), "videos"]
        timestep = load_hdf5_to_dict(self._hdf5_file, self._index, keys_to_ignore=keys_to_ignore)

        self._index += 1

        return timestep

    def close(self):
        self._hdf5_file.close()


def load_trajectory(
    filepath=None,
    read_cameras=True,
    recording_folderpath=None,
    camera_kwargs={},
    remove_skipped_steps=False,
    num_samples_per_traj=None,
    num_samples_per_traj_coeff=1.5,
):
    read_recording_folderpath = read_cameras and (recording_folderpath is not None)

    traj_reader = TrajectoryReader(filepath)
    if read_recording_folderpath:
        camera_reader = RecordedMultiCameraWrapper(recording_folderpath, camera_kwargs)

    horizon = traj_reader.length()
    timestep_list = []

    if num_samples_per_traj:
        num_to_save = num_samples_per_traj
        if remove_skipped_steps:
            num_to_save = int(num_to_save * num_samples_per_traj_coeff)
        max_size = min(num_to_save, horizon)
        indices_to_save = np.sort(np.random.choice(horizon, size=max_size, replace=False))
    else:
        indices_to_save = np.arange(horizon)

    for i in indices_to_save:
        timestep = traj_reader.read_timestep(index=i)

        if read_recording_folderpath:
            timestamp_dict = timestep["observation"]["timestamp"]["cameras"]
            camera_type_dict = {
                k: camera_type_to_string_dict[v] for k, v in timestep["observation"]["camera_type"].items()
            }
            camera_obs = camera_reader.read_cameras(
                index=i, camera_type_dict=camera_type_dict, timestamp_dict=timestamp_dict
            )
            camera_failed = camera_obs is None

            if camera_failed:
                break
            timestep["observation"].update(camera_obs)

        step_skipped = not timestep["observation"]["controller_info"].get("movement_enabled", True)
        delete_skipped_step = step_skipped and remove_skipped_steps

        if delete_skipped_step:
            del timestep
        else:
            timestep_list.append(timestep)

    timestep_list = np.array(timestep_list)
    if (num_samples_per_traj is not None) and (len(timestep_list) > num_samples_per_traj):
        ind_to_keep = np.random.choice(len(timestep_list), size=num_samples_per_traj, replace=False)
        timestep_list = timestep_list[ind_to_keep]

    traj_reader.close()

    return timestep_list


##########################################################################################################

class ExternalCamera(enum.Enum):
    LEFT = 0
    RIGHT = 1

@dataclasses.dataclass
class Args:
    # Dataset parameters
    dataset_path: str = "/mnt/data2/droid/droid/data/success/2025-12-24/Wed_Dec_24_15:39:08_2025"  # Path to the dataset folder containing trajectory.h5 and recordings/
    
    # Policy parameters
    # Choose which external camera to use: 0 for first exterior camera, 1 for second exterior camera
    external_camera: ExternalCamera = ExternalCamera.RIGHT
    
    # Rollout parameters
    max_timesteps: int = None  # Maximum timesteps to process (None for all)
    open_loop_horizon: int = 8  # How many actions from chunk before querying again
    
    # Remote server parameters
    remote_host: str = "0.0.0.0"
    remote_port: int = 8000
    
    # Output parameters
    save_video: bool = True
    save_results: bool = True
    output_dir: str = "results/dry_run"


def draw_prompt_overlay(
    img: np.ndarray,
    prompt: str,
    *,
    panel_pos: str = "top",
    max_width_ratio: float = 0.90,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 2.8,
    thickness: int = 2,
    line_spacing: float = 1.35,
    pad: int = 12,
    alpha: float = 0.55,
    add_timestamp: bool = False,
) -> np.ndarray:
    """Draw a nice prompt overlay on BGR uint8 image."""
    if img is None:
        return img
    out = img.copy()
    
    prompt = "" if prompt is None else str(prompt)
    prompt = prompt.strip()
    if add_timestamp:
        prompt = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {prompt}"
    
    H, W = out.shape[:2]
    if prompt == "":
        return out
    
    # Estimate wrap width in characters
    import textwrap
    sample = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    (sample_w, sample_h), _ = cv2.getTextSize(sample, font, font_scale, thickness)
    avg_char_w = max(6, int(sample_w / max(1, len(sample))))
    max_text_px = int(W * max_width_ratio) - 2 * pad
    max_chars = max(10, max_text_px // avg_char_w)
    
    lines = textwrap.wrap(prompt, width=max_chars, break_long_words=False, break_on_hyphens=False)
    lines = lines[:6]
    
    # Compute text block size
    line_sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    text_w = max(w for (w, h) in line_sizes) if line_sizes else 0
    text_h_single = max(h for (w, h) in line_sizes) if line_sizes else 0
    block_h = int(len(lines) * text_h_single * line_spacing)
    
    panel_w = min(W - 2 * pad, text_w + 2 * pad)
    panel_h = min(H - 2 * pad, block_h + 2 * pad)
    
    x0 = (W - panel_w) // 2
    y0 = pad if panel_pos == "top" else (H - panel_h - pad)
    x1, y1 = x0 + panel_w, y0 + panel_h
    
    # Semi-transparent panel
    overlay = out.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), thickness=-1)
    out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)
    cv2.rectangle(out, (x0, y0), (x1, y1), (255, 255, 255), thickness=1)
    
    # Draw text with outline
    y = y0 + pad + text_h_single
    for line in lines:
        (lw, lh), _ = cv2.getTextSize(line, font, font_scale, thickness)
        x = x0 + (panel_w - lw) // 2
        cv2.putText(out, line, (x, y), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(out, line, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y += int(lh * line_spacing)
    
    return out


@contextlib.contextmanager
def prevent_keyboard_interrupt():
    """Temporarily prevent keyboard interrupts by delaying them until after the protected code."""
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)
    
    def handler(signum, frame):
        nonlocal interrupted
        interrupted = True
    
    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt


def load_language_instruction(dataset_path: Path):
    """Load language instruction from metadata and aggregated_instructions.json."""
    metadata_files = list(dataset_path.glob("metadata_*.json"))
    if not metadata_files:
        return "No instruction found"
    
    metadata_file = metadata_files[0]
    episode_id = metadata_file.name.split(".")[0].split("_", 1)[-1]
    
    parent_dir = dataset_path.parent
    agg_instructions_path = parent_dir / "aggregated_instructions.json"
    
    if agg_instructions_path.exists():
        with open(agg_instructions_path, "r") as f:
            annotations = json.load(f)
            if episode_id in annotations:
                return annotations[episode_id].get("language_instruction1", "No instruction")
    
    return "No instruction found"


def main(args: Args):
    # Validate parameters
    assert args.external_camera in [ExternalCamera.LEFT, ExternalCamera.RIGHT], \
        f"external_camera must be {ExternalCamera.LEFT} or {ExternalCamera.RIGHT}, got {args.external_camera}"
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
    
    dataset_path = Path(args.dataset_path)
    
    # Load language instruction
    language_instruction = load_language_instruction(dataset_path)
    print(f"Language instruction: {language_instruction}")
    
    # Load trajectory data using the proven load_trajectory function
    print(f"Loading trajectory from {dataset_path}...")
    trajectory_filepath = str(dataset_path / "trajectory.h5")
    recording_folderpath = str(dataset_path / "recordings" / "MP4")
    
    trajectory = load_trajectory(
        filepath=trajectory_filepath,
        read_cameras=True,
        recording_folderpath=recording_folderpath,
        camera_kwargs={},
        remove_skipped_steps=True,
    )
    
    print(f"Loaded {len(trajectory)} timesteps from dataset")
    
    # Connect to policy server
    print(f"Connecting to policy server at {args.remote_host}:{args.remote_port}")
    policy_client = websocket_client_policy.WebsocketClientPolicy(args.remote_host, args.remote_port)
    print("Connected to policy server!")
    
    # Prepare tracking variables
    action_errors = []
    video_frames = []
    
    # Rollout parameters
    actions_from_chunk_completed = 0
    pred_action_chunk = None
    
    # Determine how many timesteps to process
    num_timesteps = min(args.max_timesteps, len(trajectory)) if args.max_timesteps else len(trajectory)
    
    # Main inference loop
    print(f"\nRunning dry run inference on {num_timesteps} timesteps...")
    print("Press Ctrl+C to stop early.\n")
    
    bar = tqdm.tqdm(range(num_timesteps))
    for t_step in bar:
        try:
            start_time = time.time()
            
            # Get timestep data
            step = trajectory[t_step]
            
            # Identify camera types (0=wrist, 1=exterior)
            camera_type_dict = step["observation"]["camera_type"]
            wrist_ids = [k for k, v in camera_type_dict.items() if v == 0]
            exterior_ids = [k for k, v in camera_type_dict.items() if v != 0]
            
            if len(exterior_ids) < 2:
                print(f"Warning: Expected 2 exterior cameras, found {len(exterior_ids)}")
            if len(wrist_ids) < 1:
                print(f"Warning: Expected 1 wrist camera, found {len(wrist_ids)}")
            
            # Select which exterior camera to use
            selected_exterior_id = exterior_ids[args.external_camera.value]
            wrist_id = wrist_ids[0]
            
            # Extract images (BGR format from MP4, left stereo camera only)
            external_image = step["observation"]["image"][selected_exterior_id]  # BGR
            wrist_image = step["observation"]["image"][wrist_id]  # BGR

            # Convert BGR to RGB for policy
            external_image_rgb = external_image[..., ::-1]
            wrist_image_rgb = wrist_image[..., ::-1]
            
            # Create visualization frame (keep BGR for OpenCV)
            if args.save_video:
                frame = np.concatenate([external_image_rgb, wrist_image_rgb], axis=1)
                frame = draw_prompt_overlay(frame, language_instruction, panel_pos="top", alpha=0.55)
                video_frames.append(frame)
            
            # Extract robot state (proprioception)
            joint_position = np.array(step["observation"]["robot_state"]["joint_positions"])
            gripper_position = np.array([step["observation"]["robot_state"]["gripper_position"]])
            
            # Query policy if needed
            if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= args.open_loop_horizon:
                actions_from_chunk_completed = 0
                
                # Prepare request (resize images to 224x224)
                request_data = {
                    "observation/exterior_image_1_left": image_tools.resize_with_pad(
                        external_image_rgb, 224, 224
                    ),
                    "observation/wrist_image_left": image_tools.resize_with_pad(wrist_image_rgb, 224, 224),
                    "observation/joint_position": joint_position,
                    "observation/gripper_position": gripper_position,
                    "prompt": language_instruction,
                }
                
                # Query policy server
                with prevent_keyboard_interrupt():
                    response = policy_client.infer(request_data)
                    pred_action_chunk = response["actions"][:10, :]  # [10, 8]
                
                assert pred_action_chunk.shape == (10, 8), f"Expected (10, 8), got {pred_action_chunk.shape}"
            
            # Get current predicted action from chunk
            pred_action = pred_action_chunk[actions_from_chunk_completed]
            actions_from_chunk_completed += 1
            
            # Extract ground truth action
            gt_joint_velocity = np.array(step["action"]["joint_velocity"])
            gt_gripper_position = np.array([step["action"]["gripper_position"]])
            gt_action_vec = np.concatenate([gt_joint_velocity, gt_gripper_position])
            
            # Calculate error
            action_error = np.abs(pred_action - gt_action_vec)
            action_errors.append({
                "timestep": t_step,
                "joint_velocity_error": action_error[:7],
                "gripper_position_error": action_error[7],
                "joint_velocity_mse": np.mean(action_error[:7] ** 2),
                "total_mse": np.mean(action_error ** 2),
            })
            
            # Update progress bar with current error
            bar.set_postfix({
                "jv_mse": f"{action_errors[-1]['joint_velocity_mse']:.4f}",
                "total_mse": f"{action_errors[-1]['total_mse']:.4f}",
            })
            
            # Sleep to match data collection frequency
            elapsed_time = time.time() - start_time
            if elapsed_time < 1 / DROID_CONTROL_FREQUENCY:
                time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed_time)
        
        except KeyboardInterrupt:
            print("\nStopped early by user.")
            break
        except Exception as e:
            print(f"\nError at timestep {t_step}: {e}")
            import traceback
            traceback.print_exc()
            break
    
    # Calculate statistics
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    if len(action_errors) == 0:
        print("No timesteps were processed!")
        return
    
    joint_velocity_errors = [e["joint_velocity_error"] for e in action_errors]
    gripper_errors = [e["gripper_position_error"] for e in action_errors]
    joint_velocity_mse = [e["joint_velocity_mse"] for e in action_errors]
    total_mse = [e["total_mse"] for e in action_errors]
    
    print(f"Total timesteps processed: {len(action_errors)}")
    print(f"Language instruction: {language_instruction}")
    print()
    print(f"Joint Velocity Error (MAE):")
    print(f"  Mean: {np.mean(joint_velocity_errors):.6f}")
    print(f"  Std:  {np.std(joint_velocity_errors):.6f}")
    print(f"  Max:  {np.max(joint_velocity_errors):.6f}")
    print()
    print(f"Joint Velocity MSE:")
    print(f"  Mean: {np.mean(joint_velocity_mse):.6f}")
    print(f"  Std:  {np.std(joint_velocity_mse):.6f}")
    print()
    print(f"Gripper Position Error (MAE):")
    print(f"  Mean: {np.mean(gripper_errors):.6f}")
    print(f"  Std:  {np.std(gripper_errors):.6f}")
    print(f"  Max:  {np.max(gripper_errors):.6f}")
    print()
    print(f"Total MSE (all actions):")
    print(f"  Mean: {np.mean(total_mse):.6f}")
    print(f"  Std:  {np.std(total_mse):.6f}")
    print()
    
    # Save results
    if args.save_results:
        # Save detailed results to CSV
        results_df = pd.DataFrame(action_errors)
        csv_filename = os.path.join(args.output_dir, f"dry_run_{timestamp}.csv")
        results_df.to_csv(csv_filename, index=False)
        print(f"Saved detailed results to {csv_filename}")
        
        # Save summary statistics
        summary = {
            "dataset_path": args.dataset_path,
            "language_instruction": language_instruction,
            "num_timesteps": len(action_errors),
            "joint_velocity_mae_mean": float(np.mean(joint_velocity_errors)),
            "joint_velocity_mae_std": float(np.std(joint_velocity_errors)),
            "joint_velocity_mse_mean": float(np.mean(joint_velocity_mse)),
            "gripper_mae_mean": float(np.mean(gripper_errors)),
            "gripper_mae_std": float(np.std(gripper_errors)),
            "total_mse_mean": float(np.mean(total_mse)),
            "total_mse_std": float(np.std(total_mse)),
            "timestamp": timestamp,
        }
        
        summary_filename = os.path.join(args.output_dir, f"summary_{timestamp}.json")
        with open(summary_filename, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary to {summary_filename}")
    
    # Save video
    if args.save_video and len(video_frames) > 0:
        video_filename = os.path.join(args.output_dir, f"dry_run_{timestamp}.mp4")
        video_array = np.stack(video_frames)
        ImageSequenceClip(list(video_array), fps=10).write_videofile(
            video_filename, codec="libx264", logger=None
        )
        print(f"Saved video to {video_filename}")
    
    print("\nDone!")

    import threading

    # Clear all threads except the main thread (if any non-daemon remain, warn or forcibly exit, as appropriate).
    for t in threading.enumerate():
        if t is threading.main_thread():
            continue
        if t.is_alive():
            if t.daemon:
                continue
            print(f"Warning: Non-daemon thread {t.name} is still alive at end of script.")


if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    main(args)

