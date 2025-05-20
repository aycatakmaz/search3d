#!/bin/bash
export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine
set -e

# SEARCH3D MULTISCAN OBJECT FEATURE COMPUTATION
# This script computes and saves the object features for MultiScan, given that object masks are already 
# extracted at the previous step (using the run_search3d_multiscan_obj_masks.sh script)

# --------
# NOTE TO USER: SET THE FOLLOWING PATHS "SCANS_PATH" AND "OUTPUT_DIRECTORY"! Other paths are automatically set.
# Optionally, you can also update the experiment name, but it should be the same as the one used in the mask computation script.

SCANS_PATH="/media/ayca/Elements/SEARCH3D/multiscan_processed_search3d"
OUTPUT_DIRECTORY="/media/ayca/Elements/search3d_unified_experiments_MULTISCAN/fusion_exp_3d_segment"
EXPERIMENT_NAME="fusion_3d_seg_MULTISCAN"
TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S")
TIMESTAMP="EXP"
OUTPUT_FOLDER_DIRECTORY="${OUTPUT_DIRECTORY}/${TIMESTAMP}-${EXPERIMENT_NAME}"

echo "[INFO] Output folder: ${OUTPUT_FOLDER_DIRECTORY}"

# output directories to save masks and mask features
MASKS_SAVE_DIR="${OUTPUT_FOLDER_DIRECTORY}/masks" #/media/ayca/Elements/ayca/OpenMask3D/mask3d_predictions/scannet200_val/scannet200/validation_query_150_topk_750_dbscan_0.95_scores_threshold_0.0_filter_out_False_iou_threshold_1.0
MASK_FEATURES_SAVE_DIR="${OUTPUT_FOLDER_DIRECTORY}/mask_features"

OPTIMIZE_GPU_USAGE=true
SAVE_CROPS=false

cd search3d

# 2. Compute mask features
echo "[INFO] STEP 2 - Starting mask feature computation..."
echo "[INFO] Using masks from ${MASKS_SAVE_DIR}."
python mask_feature_computation/compute_mask_features.py \
--config-name="search3d_multiscan_eval" \
data.scans_path=${SCANS_PATH} \
data.masks.masks_dir=${MASKS_SAVE_DIR} \
output.mask_features_save_dir=${MASK_FEATURES_SAVE_DIR} \
output.save_crops=${SAVE_CROPS} \
search3d.mask_features.frequency=${MASK_FEATS_FREQUENCY} \
gpu.optimize_gpu_usage=${OPTIMIZE_GPU_USAGE} \
hydra.run.dir="${OUTPUT_FOLDER_DIRECTORY}/hydra_outputs/mask_feature_computation"
echo "[INFO] Mask feature computation done!"
echo "[INFO] Mask features saved to ${MASK_FEATURES_SAVE_DIR}."
