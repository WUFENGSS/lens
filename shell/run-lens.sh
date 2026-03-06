#!/bin/bash
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

datapath=/root/autodl-tmp/lens/OK_ROI0227
augpath=/root/autodl-tmp/dtd/images
realbank=/root/autodl-tmp/lens_semi_anomaly
realimg=/root/autodl-tmp/lens_semi_anomaly
classes=('utd_data')
flags=($(for class in "${classes[@]}"; do echo '-d '"${class}"; done))

cd ..
python main.py \
    --gpu 0 \
    --seed 0 \
    --test ckpt \
  net \
    -b wideresnet50 \
    -le layer2 \
    -le layer3 \
    --pretrain_embed_dimension 1536 \
    --target_embed_dimension 1536 \
    --patchsize 3 \
    --meta_epochs 100 \
    --eval_epochs 10 \
    --dsc_layers 2 \
    --dsc_hidden 1024 \
    --pre_proj 1 \
    --mining 1 \
    --noise 0.015 \
    --radius 0.75 \
    --p 0.5 \
    --step 20 \
    --limit 392 \
    --use_costfilter 1 \
    --cf_kernel_size 3 \
    --cf_base_channels 32 \
    --cf_lambda 0.2 \
    --cf_weight 0.3 \
    --real_feat_guidance 1 \
    --real_bank_path $realbank \
    --real_mode hybrid \
    --real_lambda 0.1 \
    --real_warmup_epochs 0 \
    --real_bank_max_samples 2048 \
    --real_mix_prob_min 0.15 \
    --real_mix_prob_max 0.30 \
    --real_curriculum_ratio 0.30 \
  dataset \
    --distribution 2 \
    --mean 0.5 \
    --std 0.1 \
    --fg 0 \
    --use_real_in_image_synth 1 \
    --real_anomaly_source_path $realimg \
    --real_anomaly_prob 0.5 \
    --use_polar 1 \
    --synth_in_cartesian 1 \
    --polar_inner_ratio 0.0 \
    --polar_outer_ratio 1.0 \
    --polar_ring_constraint 1 \
    --region_split_eval 1 \
    --aperture_ratio 0.25 \
    --region_center_x -1.0 \
    --region_center_y -1.0 \
    --vis_save_size 256 \
    --fpr_target_tpr 0.98 \
    --rand_aug 1 \
    --batch_size 4 \
    --resize 512 \
    --imagesize 512 "${flags[@]}" mvtec $datapath $augpath

# use_costfilter 1  开启代价滤波
# use_real_in_image_synth 1 开启“DTD+真实缺陷”图像级异常源混合
# real_anomaly_source_path：真实缺陷图像源根目录（递归读取 jpg/jpeg/png/bmp/tif/tiff）
# real_anomaly_prob 0.5：图像级合成时采样真实缺陷概率（与 DTD 约 1:1）
# synth_in_cartesian 1 在笛卡尔域做异常合成（DTD/真实），再映射到极坐标
# cf_base_channels：网络容量（默认 32）
# cf_kernel_size：代价体邻域（默认 3）
# cf_lambda：raw 与 MR-CFN 融合权重（默认 0.2）
# cf_weight：训练中 L_cf 权重（默认 0.3）

# real_feat_guidance：开启真实缺陷特征校准（无位置先验）
# real_bank_path：真实缺陷库路径，优先读取 *_crop.*
# real_mode：真实分布校准方式 cosine/mahalanobis/hybrid
# real_lambda：真实校准损失权重
# real_mix_prob_min/max：后期每iter注入真实校准概率（建议 0.15~0.30）
# real_curriculum_ratio：前30% epoch 纯合成，后70%逐步注入真实校准


# 1做极坐标展开
# 环带内半径比例
# 环带外半径比例
# 1: 开启环带约束（异常合成仅在环带有效区）基于几何先验的前景掩码提取
# 可视化输出高度（拼图宽度=3倍）

# region_split_eval 1：开启评估分区统计（全图指标仍保留，用于与历史结果可比）
# aperture_ratio 0.25：Aperture 半径阈值比例；Housing 自动定义为其外环带
# region_center_x / region_center_y：分区圆心（-1 表示使用图像中心/极坐标中心）
# 分区结果会额外写入 results.csv，字段前缀为 aperture_* 与 housing_*