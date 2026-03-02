datapath=/root/autodl-tmp/lens/OK_ROI0227
augpath=/root/autodl-tmp/dtd/images
classes=('0227')
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
  dataset \
    --distribution 2 \
    --mean 0.5 \
    --std 0.1 \
    --fg 0 \
    --use_polar 1 \
    --polar_inner_ratio 0.0 \
    --polar_outer_ratio 1.0 \
    --polar_ring_constraint 1 \
    --vis_save_size 512 \
    --rand_aug 1 \
    --batch_size 4 \
    --resize 512 \
    --imagesize 512 "${flags[@]}" mvtec $datapath $augpath

# polar_ring_constraint 1  开启代价滤波
# cf_base_channels：网络容量（默认 32）
# cf_kernel_size：代价体邻域（默认 3）
# cf_lambda：raw 与 MR-CFN 融合权重（默认 0.2）
# cf_weight：训练中 L_cf 权重（默认 0.3）


# 1做极坐标展开
# 环带内半径比例
# 环带外半径比例
# 1: 开启环带约束（异常合成仅在环带有效区）基于几何先验的前景掩码提取
# 可视化输出高度（拼图宽度=3倍）