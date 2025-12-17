# ruff: noqa

import contextlib
import dataclasses
import datetime
import faulthandler
import os
import signal
import time
from moviepy.editor import ImageSequenceClip
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
import pandas as pd
from PIL import Image
from droid.robot_env import RobotEnv
import tqdm
import tyro
import h5py

faulthandler.enable()

import sys, select, termios, tty

import cv2
import numpy as np
import textwrap

# DROID data collection frequency -- we slow down execution to match this frequency
DROID_CONTROL_FREQUENCY = 15
WRIST_TORQUE_BIAS = 0.18 # NOTE: hack for RL2 Franka droid

@dataclasses.dataclass
class Args:
    # Hardware parameters
    left_camera_id: str = "21497414" # "33087938"  # e.g., "24259877"
    # right_camera_id: str = "33087938"  # e.g., "24514023"
    wrist_camera_id: str = "18482824"  # e.g., "13062452"

    # Policy parameters
    external_camera: str = (
        None  # which external camera should be fed to the policy, choose from ["left", "right"]
    )

    # Rollout parameters
    max_timesteps: int = 800
    # How many actions to execute from a predicted action chunk before querying policy server again
    # 8 is usually a good default (equals 0.5 seconds of action execution).
    open_loop_horizon: int = 8

    # Remote server parameters
    remote_host: str = "0.0.0.0"  # point this to the IP address of the policy server, e.g., "192.168.1.100"
    remote_port: int = (
        8000  # point this to the port of the policy server, default server port for openpi servers is 8000
    )

    # Aux tool
    save_rollout_to_hdf5: bool = False

    # dirty fix
    torque_bias: bool = False # hack for RL2 Franka droid


def draw_prompt_overlay(
    img: np.ndarray,
    prompt: str,
    *,
    panel_pos: str = "top",          # "top" or "bottom"
    max_width_ratio: float = 0.90,   # fraction of image width usable for text
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 2.8,
    thickness: int = 2,
    line_spacing: float = 1.35,
    pad: int = 12,
    alpha: float = 0.55,             # panel transparency
    add_timestamp: bool = False,
) -> np.ndarray:
    """
    Draw a nice prompt overlay on BGR uint8 image.
    Returns a new image (does not modify input).
    """
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

    # --- Estimate wrap width in characters based on pixel width ---
    # Measure average char width using a representative string.
    sample = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    (sample_w, sample_h), _ = cv2.getTextSize(sample, font, font_scale, thickness)
    avg_char_w = max(6, int(sample_w / max(1, len(sample))))
    max_text_px = int(W * max_width_ratio) - 2 * pad
    max_chars = max(10, max_text_px // avg_char_w)

    lines = textwrap.wrap(prompt, width=max_chars, break_long_words=False, break_on_hyphens=False)
    lines = lines[:6]  # hard cap to avoid covering the whole frame

    # --- Compute text block size ---
    line_sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    text_w = max(w for (w, h) in line_sizes) if line_sizes else 0
    text_h_single = max(h for (w, h) in line_sizes) if line_sizes else 0
    block_h = int(len(lines) * text_h_single * line_spacing)

    panel_w = min(W - 2 * pad, text_w + 2 * pad)
    panel_h = min(H - 2 * pad, block_h + 2 * pad)

    x0 = (W - panel_w) // 2
    y0 = pad if panel_pos == "top" else (H - panel_h - pad)
    x1, y1 = x0 + panel_w, y0 + panel_h

    # --- Semi-transparent rounded-ish panel (rectangle + border) ---
    overlay = out.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), thickness=-1)
    out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)

    cv2.rectangle(out, (x0, y0), (x1, y1), (255, 255, 255), thickness=1)

    # --- Draw text with outline for readability ---
    y = y0 + pad + text_h_single
    for line in lines:
        (lw, lh), _ = cv2.getTextSize(line, font, font_scale, thickness)
        x = x0 + (panel_w - lw) // 2

        # Outline
        cv2.putText(out, line, (x, y), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        # Foreground
        cv2.putText(out, line, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        y += int(lh * line_spacing)

    return out

def poll_for_prompt_update(current_prompt: str) -> str:
    """
    Non-blocking check for a full line on stdin. If user typed 'i' (and pressed Enter),
    pause the loop and synchronously read a new prompt with input(), then return it.
    Otherwise return the original prompt unchanged.
    """
    # Is there a line waiting on stdin?
    ready, _, _ = select.select([sys.stdin], [], [], 0)
    if ready:
        line = sys.stdin.readline().strip()
        if line.lower() == 'i':
            # Block here: this will PAUSE the control loop until user enters new prompt
            try:
                new_prompt = input("\n[update] Enter new instruction: ").strip()
                if new_prompt:
                    print("[update] Prompt updated.")
                    return new_prompt
                else:
                    print("[update] Empty input, keeping previous prompt.")
            except KeyboardInterrupt:
                print("\n[update] Canceled; keeping previous prompt.")
    return current_prompt

def save_rollout_to_hdf5(rollout_data_list, filename):
    """
    Save a list of rollout data (dicts) to an HDF5 file.
    Each key in the dict will be a dataset; values must be numpy arrays or convertible.
    """
    with h5py.File(filename, 'w') as f:
        if not rollout_data_list:
            raise ValueError("rollout_data_list is empty")
        keys = rollout_data_list[0].keys()
        for key in keys:
            data = np.array([d[key] for d in rollout_data_list])
            # Handle string/object dtype
            if data.dtype.kind in {'U', 'O'}:
                dt = h5py.string_dtype(encoding='utf-8')
                f.create_dataset(key, data=data.astype('U'), dtype=dt)
            else:
                f.create_dataset(key, data=data)
    print(f"Saved rollout data to {filename}")

# We are using Ctrl+C to optionally terminate rollouts early -- however, if we press Ctrl+C while the policy server is
# waiting for a new action chunk, it will raise an exception and the server connection dies.
# This context manager temporarily prevents Ctrl+C and delays it after the server call is complete.
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


def main(args: Args):
    # Make sure external camera is specified by user -- we only use one external camera for the policy
    assert (
        args.external_camera is not None and args.external_camera in ["left", "right"]
    ), f"Please specify an external camera to use for the policy, choose from ['left', 'right'], but got {args.external_camera}"

    # Initialize the Panda environment. Using joint velocity action space and gripper position action space is very important.
    env = RobotEnv(action_space="joint_velocity", gripper_action_space="position")
    print("Created the droid env!")
    env.reset()
    time.sleep(1.0)

    # Connect to the policy server
    policy_client = websocket_client_policy.WebsocketClientPolicy(args.remote_host, args.remote_port)

    df = pd.DataFrame(columns=["success", "duration", "video_filename"])

    while True:
        instruction = None
        instruction_prev = None
        while instruction is None:
            instruction = input("Enter instruction: ")
            if instruction == 'reset':
                env.reset()
                time.sleep(1.0)
                instruction = None

        # Rollout parameters
        actions_from_chunk_completed = 0
        pred_action_chunk = None

        # Prepare to save video of rollout
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
        save_rollout_data_list = []
        video = []
        bar = tqdm.tqdm(range(args.max_timesteps))
        print("Running rollout... press Ctrl+C to stop early.")
        for t_step in bar:
            start_time = time.time()
            try:

                #### Update new instruction if 'i' is pressed ####
                # instruction = maybe_update_instruction(instruction)
                instruction = poll_for_prompt_update(instruction)

                # clean action buffer when instruction changes
                if instruction != instruction_prev:
                    instruction_prev = instruction
                    actions_from_chunk_completed = 0
                    pred_action_chunk = None
                    # print(f"[info] Instruction changed to: {instruction} prev instruction: {instruction_prev}")

                if instruction == "reset":
                    env.reset()
                    time.sleep(1.0)
                    break

                # Get the current observation
                curr_obs = _extract_observation(
                    args,
                    env.get_observation(),
                    # Save the first observation to disk
                    save_to_disk=t_step == 0,
                )

                left_frame = curr_obs[f"{args.external_camera}_image"]  # BGR uint8 assumed
                wrist_frame = curr_obs["wrist_image"]
                frame = np.concatenate([left_frame, wrist_frame], axis=1)
                frame = draw_prompt_overlay(frame, instruction, panel_pos="top", alpha=0.55)
                video.append(frame)

                ###################
                # Send websocket request to policy server if it's time to predict a new chunk
                if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= args.open_loop_horizon:
                    actions_from_chunk_completed = 0

                    # We resize images on the robot laptop to minimize the amount of data sent to the policy server
                    # and improve latency.
                    request_data = {
                        "observation/exterior_image_1_left": image_tools.resize_with_pad(
                            curr_obs[f"{args.external_camera}_image"], 224, 224
                        ),
                        "observation/wrist_image_left": image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224),
                        "observation/joint_position": curr_obs["joint_position"],
                        "observation/gripper_position": curr_obs["gripper_position"],
                        "prompt": instruction,
                    }
                    save_rollout_data_list.append(request_data)

                    # Wrap the server call in a context manager to prevent Ctrl+C from interrupting it
                    # Ctrl+C will be handled after the server call is complete
                    with prevent_keyboard_interrupt():
                        # this returns action chunk [10, 8] of 10 joint velocity actions (7) + gripper position (1)
                        pred_action_chunk = policy_client.infer(request_data)["actions"]

                    # print(f"Predicted action chunk with shape: {pred_action_chunk.shape}")
                    pred_action_chunk = pred_action_chunk[:10, :] # TODO: hack
                    assert pred_action_chunk.shape == (10, 8)
                #####################

                # #### TODO: hack for testing
                # if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= args.open_loop_horizon:
                #     actions_from_chunk_completed = 0
                #     pred_action_chunk = np.zeros((10, 8))  # TODO: hack
                #     pred_action_chunk[:, 5] = 0.18  # keep gripper open
                # ####

                # Select current action to execute from chunk
                action = pred_action_chunk[actions_from_chunk_completed]
                actions_from_chunk_completed += 1

                # Binarize gripper action
                if action[-1].item() > 0.5:
                    # action[-1] = 1.0
                    action = np.concatenate([action[:-1], np.ones((1,))])
                else:
                    # action[-1] = 0.0
                    action = np.concatenate([action[:-1], np.zeros((1,))])

                # hack to add torque bias
                if args.torque_bias:
                    action[5] += WRIST_TORQUE_BIAS # NOTE: hack for RL2 Franka droid
                # clip all dimensions of action to [-1, 1]
                action = np.clip(action, -1, 1)

                # print(f"action is {action}")
                env.step(action)

                instruction_prev = instruction

                # Sleep to match DROID data collection frequency
                elapsed_time = time.time() - start_time
                if elapsed_time < 1 / DROID_CONTROL_FREQUENCY:
                    time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed_time)
            except KeyboardInterrupt:
                break

        video = np.stack(video)
        save_filename = "videos/video_" + timestamp
        ImageSequenceClip(list(video), fps=10).write_videofile(save_filename + ".mp4", codec="libx264")

        ### Save all data ###
        if args.save_rollout_to_hdf5:
            # import pdb; pdb.set_trace()
            save_rollout_to_hdf5(save_rollout_data_list, save_filename + ".hdf5")

        success: str | float | None = None
        while not isinstance(success, float):
            success = input(
                "Did the rollout succeed? (enter y for 100%, n for 0%), or a numeric value 0-100 based on the evaluation spec"
            )
            if success == "y":
                success = 1.0
            elif success == "n":
                success = 0.0

            success = float(success) / 100
            if not (0 <= success <= 1):
                print(f"Success must be a number in [0, 100] but got: {success * 100}")

        df = df.append(
            {
                "success": success,
                "duration": t_step,
                "video_filename": save_filename,
            },
            ignore_index=True,
        )

        if input("Do one more eval? (enter y or n) ").lower() != "y":
            break
        env.reset()
        time.sleep(1.0)

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%I:%M%p_%B_%d_%Y")
    csv_filename = os.path.join("results", f"eval_{timestamp}.csv")
    df.to_csv(csv_filename)
    print(f"Results saved to {csv_filename}")


def _extract_observation(args: Args, obs_dict, *, save_to_disk=False):
    image_observations = obs_dict["image"]
    left_image, right_image, wrist_image = None, None, None
    for key in image_observations:
        # Note the "left" below refers to the left camera in the stereo pair.
        # The model is only trained on left stereo cams, so we only feed those.
        if args.left_camera_id in key and "left" in key:
            left_image = image_observations[key]
        # elif args.right_camera_id in key and "left" in key:
        #     right_image = image_observations[key]
        elif args.wrist_camera_id in key and "left" in key:
            wrist_image = image_observations[key]

    # Drop the alpha dimension
    left_image = left_image[..., :3]
    # right_image = right_image[..., :3]
    wrist_image = wrist_image[..., :3]

    # Convert to RGB
    left_image = left_image[..., ::-1]
    # right_image = right_image[..., ::-1]
    wrist_image = wrist_image[..., ::-1]

    # In addition to image observations, also capture the proprioceptive state
    robot_state = obs_dict["robot_state"]
    cartesian_position = np.array(robot_state["cartesian_position"])
    joint_position = np.array(robot_state["joint_positions"])
    gripper_position = np.array([robot_state["gripper_position"]])

    # Save the images to disk so that they can be viewed live while the robot is running
    # Create one combined image to make live viewing easy
    if save_to_disk:
        combined_image = np.concatenate([left_image, wrist_image], axis=1)
        combined_image = Image.fromarray(combined_image)
        combined_image.save("robot_camera_views.png")

    return {
        "left_image": left_image,
        # "right_image": right_image,
        "wrist_image": wrist_image,
        "cartesian_position": cartesian_position,
        "joint_position": joint_position,
        "gripper_position": gripper_position,
    }


if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    main(args)
