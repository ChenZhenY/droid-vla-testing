################ Get the ckpt #################
CONFIG_NAME=pi05_droid_finetune
CKPT_NAME=pi05_DROID_4tasks_1225

# # downloaded from scratch folder
rsync -avz --progress zhenyang-ice:/home/hice1/zchen927/scratch/openpi/checkpoints/$CONFIG_NAME/$CKPT_NAME /mnt/data2/droid/openpi/checkpoints

# downloaded from cedar
# rsync -avz --progress zhenyang-ice:/storage/cedar/cedar0/cedarp-dxu345-0/zhenyang/checkpoints/$CONFIG_NAME/$CKPT_NAME /mnt/data2/droid/openpi/checkpoints
