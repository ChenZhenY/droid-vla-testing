#!/bin/bash

# Script to generate batch config and collect interpolated trajectories
# Usage: ./run_batch_interpolation.sh [options]

# TODO: calcuate delta based on the average delta speed of the dataset

set -e  # Exit on error

# Default values
DATASET_DIR="/mnt/data2/droid/droid/data/success/2025-12-24"
# TASKS="put the white bowl on the countertop shelf,put the white bowl in the sink"
TASKS="put the white bowl on the countertop shelf,put the tea bag in the yellow drawer"
STAGE=0
NUM_DEMOS=50
STEP_DELTA=0.005 # important to set a small value to match the control frequency of the robot
BIDIRECTION="True"  # generate bidirectional trajectory pairs
OUTPUT_YAML="data/batch_config_interpolate.yaml"
OUTPUT_DIR="data/interpolation"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

python "$SCRIPT_DIR/generate_interpolate_batch_config.py" \
    --dataset-dir "$DATASET_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --step-delta $STEP_DELTA \
    --tasks "$TASKS" \
    --stage $STAGE \
    --num-demos $NUM_DEMOS \
    --output-yaml "$OUTPUT_YAML" \
    $( [[ "$BIDIRECTION" == "True" ]] && echo "--bidirection" )

python $SCRIPT_DIR/collect_interpolated_trajectory.py --config-file $OUTPUT_YAML

