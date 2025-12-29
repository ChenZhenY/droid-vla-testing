################ Get the ckpt #################
CONFIG_NAME=pi05_droid_finetune
CKPT_NAME=pi05_DROID_whitebowl_noops_1228
EPOCH=9999

# # downloaded from scratch folder
rsync -avz --progress zhenyang-ice:/home/hice1/zchen927/scratch/openpi/checkpoints/$CONFIG_NAME/$CKPT_NAME/$EPOCH /mnt/data2/droid/openpi/checkpoints/$CKPT_NAME

# downloaded from cedar
# rsync -avz --progress zhenyang-ice:/storage/cedar/cedar0/cedarp-dxu345-0/zhenyang/checkpoints/$CONFIG_NAME/$CKPT_NAME /mnt/data2/droid/openpi/checkpoints
    