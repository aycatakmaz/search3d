#!/bin/bash
export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine

# SEARCH3D MULTISCAN OBJECT MASK COMPUTATION
# This script computes and saves the instance-level object masks for MultiScan

# NOTE TO USER: SET THE FOLLOWING PATHS: "SCANS_PATH" AND "OUTPUT_DIRECTORY"! Other paths are automatically set.
# Optionally, you can also update the experiment name, but ensure that you use the same experiment name in the segmentation and feature computation scripts.
SCANS_PATH="/media/ayca/Elements/SEARCH3D/multiscan_processed_search3d"
OUTPUT_DIRECTORY="/media/ayca/Elements/search3d_unified_experiments_MULTISCAN/fusion_exp_3d_segment"
EXPERIMENT_NAME="fusion_3d_seg_MULTISCAN"
#OUTPUT_FOLDER_DIRECTORY="${OUTPUT_DIRECTORY}/${EXPERIMENT_NAME}"
TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S")
TIMESTAMP="EXP"
OUTPUT_FOLDER_DIRECTORY="${OUTPUT_DIRECTORY}/${TIMESTAMP}-${EXPERIMENT_NAME}"
echo "[INFO] Output folder: ${OUTPUT_FOLDER_DIRECTORY}"

MASKS_SAVE_DIR="${OUTPUT_FOLDER_DIRECTORY}/masks" 
MASK_MODULE_CKPT_PATH="../resources/scannet200_model.ckpt"

cd search3d

python object_and_part_computation/object_instance_computation/get_masks_multiscan.py \
general.experiment_name="multiscan" \
general.checkpoint=${MASK_MODULE_CKPT_PATH} \
general.train_mode=false \
data.test_mode=test \
model.num_queries=150 \
general.use_dbscan=true \
general.dbscan_eps=0.95 \
general.mask_save_dir=${MASKS_SAVE_DIR} \
general.scan_dir=${SCANS_PATH} \
hydra.run.dir="${OUTPUT_FOLDER_DIRECTORY}/hydra_outputs/mask_computation"