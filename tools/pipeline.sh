#!/usr/bin/env bash

if [[ $# -lt 1 || -z "$1" ]]; then
  echo "Usage: bash tools/pipeline.sh <CASE>" >&2
  exit 2
fi

cd /nvme0/zhangruiying/PhysCoRe
conda activate /nvme0/zhangruiying/anaconda3/envs/physcore

export CASE="$1"
export RAW_ROOT=$PWD/data/data/physcore/different_types
export EPISODE_ROOT=/data5/public_data/PhysCoRe/medium
export OUTPUT_ROOT=/data5/public_data/PhysCoRe/result

# CASE 中包含 cloth 时使用布料粒子预设，其他 case 使用通用预设。
PARTICLE_VOXEL_SIZE=0.008
TARGET_TOTAL_PARTICLES=12000
INTERIOR_TO_SHELL_ARGS=()
if [[ "$CASE" == *cloth* ]]; then
  PARTICLE_VOXEL_SIZE=0.004
  TARGET_TOTAL_PARTICLES=16000
  INTERIOR_TO_SHELL_ARGS=(--interior_to_shell_max_distance 0.004)
fi

python datagen/convert3d/convert_to_episode.py \
  --source_dir "$RAW_ROOT/$CASE" \
  --output_dir "$EPISODE_ROOT/$CASE" \
  --controller_mask_label hand \
  --no-fill_interior \
  --particle_voxel_size "$PARTICLE_VOXEL_SIZE" \
  --target_total_particles "$TARGET_TOTAL_PARTICLES" \
  --mask_erode_pixels 2 \
  "${INTERIOR_TO_SHELL_ARGS[@]}"

python validate_MfM.py \
  --config data/configs/validate_MfM.yaml \
  --checkpoint data/checkpoints/MfM_checkpoint.pt \
  --root "$EPISODE_ROOT/$CASE/episode_0000" \
  --out "$OUTPUT_ROOT/MfM/$CASE/validation.json"

export RFD_RUN=$OUTPUT_ROOT/RfD/released
mkdir -p "$RFD_RUN/checkpoints"
cp data/checkpoints/RfD_checkpoint.pt "$RFD_RUN/checkpoints/gvc_epoch_0001.pt"

python validate_RfD.py \
  --config data/configs/validate_RfD.yaml \
  --refiner-checkpoint data/checkpoints/MfM_checkpoint.pt \
  --epoch 1 \
  --save-trajectory \
  "dataset.validation_roots=[$EPISODE_ROOT/$CASE/episode_0000]" \
  "train.output_dir=$RFD_RUN"

export GS_RUN='init=pcd_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0'

python render_MfM_confidence_3dgs.py \
  --config configs/render_MfM_confidence_3dgs.yaml \
  --mode appearance \
  --episode-root "$EPISODE_ROOT/$CASE/episode_0000" \
  --gs-dir "$PWD/data/gaussian_output/physcore/$CASE/$GS_RUN" \
  --traj-path "$OUTPUT_ROOT/MfM/$CASE/validation_episode_0000_trajectory.pt" \
  --out-dir "$OUTPUT_ROOT/MfM_appearance/$CASE" \
  --name "$CASE"

python render_MfM_confidence_3dgs.py \
  --config configs/render_MfM_confidence_3dgs.yaml \
  --mode appearance \
  --episode-root "$EPISODE_ROOT/$CASE/episode_0000" \
  --gs-dir "$PWD/data/gaussian_output/physcore/$CASE/$GS_RUN" \
  --traj-path "$OUTPUT_ROOT/RfD/released/trajectories/epoch_0001/val00_${CASE}_episode_0000_trajectory.pt"  \
  --out-dir "$OUTPUT_ROOT/MfM_appearance/$CASE/RfD" \
  --name "$CASE"

python tools/visualize_particle_trajectory.py \
  --input /data5/public_data/PhysCoRe/result/MfM/$CASE/validation_episode_0000_trajectory.pt \
  --output /tmp/${CASE}_particles.mp4 \
  --fps 30 \
  --width 480 \
  --height 360 \
  --point-radius 1 \
  --azimuth -45 \
  --elevation 25

python tools/visualize_sampled_tracks.py \
  --input data/data/physcore/different_types/$CASE/sampled_tracks.pkl \
  --output /tmp/${CASE}_tracks.mp4 \
  --color-mode track \
  --azimuth -45 \
  --elevation 20 \
  --point-radius 2