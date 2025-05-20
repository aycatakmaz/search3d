#!/bin/bash
export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine
set -e

# SEARCH3D MULTISCAN SEGMENT FEATURE COMPUTATION SCRIPT
# This script performs the following in order to compute segment-level features for the MultiScan dataset:
# 1. Compute segments and save them
# 2. Compute segment features for each segment in each instance and save them
# This script assumes that the object masks have already been computed and saved using the script run_search3d_multiscan_obj_masks.sh.

# --------
# NOTE TO USER: SET THE FOLLOWING PATHS: "SCANS_PATH" AND "OUTPUT_DIRECTORY"! Other paths are automatically set.
# Optionally, you can also update the experiment name, but ensure that you use the same experiment name in the segmentation and feature computation scripts.
# --------
SCANS_PATH="/media/ayca/Elements/SEARCH3D/multiscan_processed_search3d"
OUTPUT_DIRECTORY="/media/ayca/Elements/search3d_unified_experiments_MULTISCAN/fusion_exp_3d_segment"
EXPERIMENT_NAME="fusion_3d_seg_MULTISCAN"

TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S")
TIMESTAMP="EXP"
OUTPUT_FOLDER_DIRECTORY="${OUTPUT_DIRECTORY}/${TIMESTAMP}-${EXPERIMENT_NAME}"

echo "[INFO] Output folder: ${OUTPUT_FOLDER_DIRECTORY}"

MASKS_SAVE_DIR="${OUTPUT_FOLDER_DIRECTORY}/masks" # read masks from this directory - masks were saved in the previous step
SEGMENTS_SAVE_DIR="${OUTPUT_FOLDER_DIRECTORY}/segments"
SEGMENT_FEATURES_SAVE_DIR="${OUTPUT_FOLDER_DIRECTORY}/segment_features"

echo "[INFO] Masks folder: ${MASKS_SAVE_DIR}"
echo "[INFO] Segments folder: ${SEGMENTS_SAVE_DIR}"
echo "[INFO] Segment features folder: ${SEGMENT_FEATURES_SAVE_DIR}"

OPTIMIZE_GPU_USAGE=true
echo "OPTIMIZE GPU USAGE: $OPTIMIZE_GPU_USAGE"

# Segment computation parameters
K_THRESH=0.05 
SEG_MIN_VERTS=100

cd search3d
DENSE_FEATS_EXP_RATIO=0.1
DENSE_FEATS_FREQUENCY=5

# 1. Compute the segments and save them
echo "[INFO] STEP 2 - Starting segment computation..."
python object_and_part_computation/get_segments.py \
--config-name="search3d_multiscan_eval" \
search3d.segment_computation.kThresh=${K_THRESH} \
search3d.segment_computation.segMinVerts=${SEG_MIN_VERTS} \
data.masks.masks_dir=${MASKS_SAVE_DIR} \
data.segments.segments_dir=${SEGMENTS_SAVE_DIR} \
gpu.optimize_gpu_usage=${OPTIMIZE_GPU_USAGE} \
hydra.run.dir="${OUTPUT_FOLDER_DIRECTORY}/hydra_outputs/segment_computation"
echo "[INFO] Segment computation done!"
echo "[INFO] Segments saved to ${SEGMENTS_SAVE_DIR}."

# 2. Compute segment features for each segment within each instance and save them
echo "[INFO] STEP 4 - Starting segment feature computation..."
python dense_feature_computation/fuse_dense_features_per_3d_segment.py \
--config-name="search3d_multiscan_eval" \
search3d.dense_features.single_level_expansion_ratio=${DENSE_FEATS_EXP_RATIO} \
search3d.dense_features.frequency=${DENSE_FEATS_FREQUENCY} \
data.scans_path=${SCANS_PATH} \
data.segments.segments_dir=${SEGMENTS_SAVE_DIR} \
output.segment_features_save_dir=${SEGMENT_FEATURES_SAVE_DIR} \
gpu.optimize_gpu_usage=${OPTIMIZE_GPU_USAGE} \
hydra.run.dir="${OUTPUT_FOLDER_DIRECTORY}/hydra_outputs/segment_feature_computation"
echo "[INFO] Segment feature computation done!"
echo "[INFO] Segment features saved to ${SEGMENT_FEATURES_SAVE_DIR}."

