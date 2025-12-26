#!/bin/bash
export HF_LEROBOT_HOME=/mnt/data2/droid/lerobot_dataset
export PATH=/mnt/data2/dexmimic/miniconda3/envs/robot/bin:$PATH

DATA_ROOT=/mnt/data2/droid/droid/data
DATASET=2025-12-24
TARGET_NAME=RL2_Klaus_4tasks_1224
USER=Daniel233

# # convert SVO to MP4 TODO: check whether need to use start data, used when only converting data of that day
# see config in the script, by default convert all data under data/ folder
# cd /mnt/data2/droid/droid
# python scripts/convert/svo_to_mp4.py --lab RL2 --start_date ${DATASET}
# # python scripts/convert/svo_to_depth.py --lab RL2 --start_date ${DATASET} --lab_agnostic False

# # get the language annoations
# cd /mnt/data2/droid/droid
# python ./scripts/convert/aggregate_instruction.py --root $DATA_ROOT/success/$DATASET \
#                                                    --out $DATA_ROOT/success/$DATASET/aggregated_instructions.json

# merge to the LeRobot format
# cd /mnt/data2/droid/openpi
# uv run examples/droid/convert_droid_data_to_lerobot.py --data-dir $DATA_ROOT/success/$DATASET \
#                                                         --target_dir $TARGET_NAME

# upload to Cedar
# rsync -avz --progress $HF_LEROBOT_HOME/$USER/$TARGET_NAME zhenyang-ice:/storage/cedar/cedar0/cedarp-dxu345-0/zhenyang/datasets
rsync -avz --progress /mnt/data2/droid/droid/data/success/2025-12-24 zhenyang-ice:/storage/cedar/cedar0/cedarp-dxu345-0/zhenyang/datasets