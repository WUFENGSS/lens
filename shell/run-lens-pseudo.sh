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
  --vis_save_size 256 \
  --threshold_mode percentile \
  --threshold_percentile 99.7 \
  --min_area 20

# 注意：pseudo_label_infer.py 当前用于导出伪标签，不计算 image/pixel/pro 指标。
# 因此这里不添加 --region_split_eval / --aperture_ratio 等评估参数（该脚本尚未实现这些入参）。
# 若需要 Housing/Aperture 分区指标，请在 main.py 评估命令中使用：
# --region_split_eval 1 --aperture_ratio 0.25 --region_center_x -1.0 --region_center_y -1.0
