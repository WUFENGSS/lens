#!/usr/bin/env bash
set -euo pipefail

ckpt=/root/GLASS-main/results/models/backbone_0/mvtec_0227/ckpt_v3_20.pth
test_dir=/root/autodl-tmp/lens/OK_ROI0227/0227/test
mask_dir=/root/autodl-tmp/lens/OK_ROI0227/0227/pseudo_ground_truth
vis_dir=/root/autodl-tmp/lens/OK_ROI0227/0227/pseudo_vis

cd ..
python pseudo_label_infer.py \
  --ckpt_path "$ckpt" \
  --test_dir "$test_dir" \
  --output_mask_dir "$mask_dir" \
  --output_vis_dir "$vis_dir" \
  --gpu 0 \
  --backbone wideresnet50 \
  --layers layer2 layer3 \
  --pretrain_embed_dimension 1536 \
  --target_embed_dimension 1536 \
  --patchsize 3 \
  --dsc_layers 2 \
  --dsc_hidden 1024 \
  --pre_proj 1 \
  --resize 512 \
  --imagesize 512 \
  --use_polar true \
  --polar_max_radius_ratio 1.0 \
  --save_vis_compare true \
  --vis_save_size 1024 \
  --threshold_mode percentile \
  --threshold_percentile 99.7 \
  --min_area 20
