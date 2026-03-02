from loss import FocalLoss
from collections import OrderedDict
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from model import Discriminator, ProjectionMap
from costfilter import CostFilterLite

import numpy as np
import pandas as pd
import torch.nn.functional as F

import logging
import os
import math
import torch
import tqdm
import common
import metrics
import cv2
import utils
import glob
import shutil

LOGGER = logging.getLogger(__name__)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class TBWrapper:
    def __init__(self, log_dir):
        self.g_iter = 0
        self.logger = SummaryWriter(log_dir=log_dir)

    def step(self):
        self.g_iter += 1


class GLASS(torch.nn.Module):
    def __init__(self, device):
        super(GLASS, self).__init__()
        self.device = device

    def load(
            self,
            backbone,
            layers_to_extract_from,
            device,
            input_shape,
            pretrain_embed_dimension,
            target_embed_dimension,
            patchsize=3,
            patchstride=1,
            meta_epochs=640,
            eval_epochs=1,
            dsc_layers=2,
            dsc_hidden=1024,
            dsc_margin=0.5,
            train_backbone=False,
            pre_proj=1,
            mining=1,
            noise=0.015,
            radius=0.75,
            p=0.5,
            lr=0.0001,
            svd=0,
            step=20,
            limit=392,
                use_costfilter=0,
                cf_kernel_size=3,
                    cf_base_channels=32,
                cf_lambda=0.2,
                cf_weight=0.3,
            **kwargs,
    ):

        self.backbone = backbone.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape
        self.device = device

        self.forward_modules = torch.nn.ModuleDict({})
        feature_aggregator = common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device, train_backbone
        )
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator"] = feature_aggregator

        self.feature_dim = int(np.sum(feature_dimensions))
        self.target_embed_dimension = target_embed_dimension
        self.map_pool = torch.nn.AvgPool2d(
            kernel_size=patchsize,
            stride=1,
            padding=patchsize // 2,
        )
        self.map_pool.to(self.device)
        self.feature_adaptor = None
        if self.feature_dim != self.target_embed_dimension:
            self.feature_adaptor = torch.nn.Conv2d(
                in_channels=self.feature_dim,
                out_channels=self.target_embed_dimension,
                kernel_size=1,
                stride=1,
            ).to(self.device)

        self.meta_epochs = meta_epochs
        self.lr = lr
        self.train_backbone = train_backbone
        if self.train_backbone:
            self.backbone_opt = torch.optim.AdamW(self.forward_modules["feature_aggregator"].backbone.parameters(), lr)

        self.pre_proj = pre_proj
        if self.pre_proj > 0:
            self.pre_projection = ProjectionMap(self.target_embed_dimension, self.target_embed_dimension)
            self.pre_projection.to(self.device)
            self.proj_opt = torch.optim.Adam(self.pre_projection.parameters(), lr, weight_decay=1e-5)

        self.eval_epochs = eval_epochs
        self.dsc_layers = dsc_layers
        self.dsc_hidden = dsc_hidden
        self.discriminator = Discriminator(self.target_embed_dimension, n_layers=dsc_layers, hidden=dsc_hidden)
        self.discriminator.to(self.device)
        self.dsc_opt = torch.optim.AdamW(self.discriminator.parameters(), lr=lr * 2)
        self.dsc_margin = dsc_margin

        self.use_costfilter = bool(use_costfilter)
        self.cf_kernel_size = max(1, int(cf_kernel_size))
        if self.cf_kernel_size % 2 == 0:
            self.cf_kernel_size += 1
        self.cf_base_channels = int(max(16, cf_base_channels))
        self.cf_lambda = float(np.clip(cf_lambda, 0.0, 1.0))
        self.cf_weight = float(max(0.0, cf_weight))
        self.cost_filter = None
        self.cf_opt = None
        self.cf_bce = torch.nn.BCEWithLogitsLoss()
        if self.use_costfilter:
            cf_in_channels = self.cf_kernel_size * self.cf_kernel_size + 1
            self.cost_filter = CostFilterLite(
                in_channels=cf_in_channels,
                hidden_channels=self.cf_base_channels,
            ).to(self.device)
            self.cf_opt = torch.optim.AdamW(self.cost_filter.parameters(), lr=lr, weight_decay=1e-5)

        self.c = torch.tensor(0)
        self.c_ = torch.tensor(0)
        self.p = p
        self.radius = radius
        self.mining = mining
        self.noise = noise
        self.svd = svd
        self.step = step
        self.limit = limit
        self.distribution = 0
        self.focal_loss = FocalLoss()

        self.anomaly_segmentor = common.RescaleSegmentor(device=self.device, target_size=input_shape[-2:])
        self.model_dir = ""
        self.dataset_name = ""
        self.logger = None

    def set_model_dir(self, model_dir, dataset_name):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.ckpt_dir = os.path.join(self.model_dir, dataset_name)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.tb_dir = os.path.join(self.ckpt_dir, "tb")
        os.makedirs(self.tb_dir, exist_ok=True)
        self.logger = TBWrapper(self.tb_dir)

    def _embed(self, images, detach=True, provide_patch_shapes=False, evaluation=False):
        """Returns fused feature maps for images."""
        if not evaluation and self.train_backbone:
            self.forward_modules["feature_aggregator"].train()
            features = self.forward_modules["feature_aggregator"](images, eval=evaluation)
        else:
            self.forward_modules["feature_aggregator"].eval()
            with torch.no_grad():
                features = self.forward_modules["feature_aggregator"](images)

        features = [features[layer] for layer in self.layers_to_extract_from]

        for i, feat in enumerate(features):
            if len(feat.shape) == 3:
                B, L, C = feat.shape
                features[i] = feat.reshape(B, int(math.sqrt(L)), int(math.sqrt(L)), C).permute(0, 3, 1, 2)

        ref_h, ref_w = features[0].shape[-2:]
        resized_features = [features[0]]
        for i in range(1, len(features)):
            resized = F.interpolate(
                features[i], size=(ref_h, ref_w), mode="bilinear", align_corners=False
            )
            resized_features.append(resized)

        feature_map = torch.cat(resized_features, dim=1)
        feature_map = self.map_pool(feature_map)
        if self.feature_adaptor is not None:
            feature_map = self.feature_adaptor(feature_map)

        if provide_patch_shapes:
            return feature_map, (ref_h, ref_w)
        return feature_map, None

    @staticmethod
    def _flatten_hw(feat_map):
        return feat_map.permute(0, 2, 3, 1).reshape(-1, feat_map.shape[1])

    @staticmethod
    def _flatten_mask(mask_map):
        return mask_map.reshape(-1, 1)

    @staticmethod
    def _get_polar_center(dataset_obj, h, w):
        cx_cfg = float(getattr(dataset_obj, "polar_center_x", -1.0))
        cy_cfg = float(getattr(dataset_obj, "polar_center_y", -1.0))
        cx = (w - 1) / 2.0 if cx_cfg < 0 else cx_cfg
        cy = (h - 1) / 2.0 if cy_cfg < 0 else cy_cfg
        cx = float(np.clip(cx, 0.0, w - 1))
        cy = float(np.clip(cy, 0.0, h - 1))
        return cx, cy

    @staticmethod
    def _inverse_polar_image(polar_img, out_h, out_w, dataset_obj, interpolation=cv2.INTER_LINEAR):
        cx, cy = GLASS._get_polar_center(dataset_obj, out_h, out_w)
        max_radius_ratio = float(getattr(dataset_obj, "polar_max_radius_ratio", 1.0))
        max_radius = min(cx, cy, (out_w - 1) - cx, (out_h - 1) - cy)
        max_radius = max(1.0, max_radius * max_radius_ratio)

        flags = cv2.WARP_POLAR_LINEAR + cv2.WARP_FILL_OUTLIERS + cv2.WARP_INVERSE_MAP + interpolation
        return cv2.warpPolar(polar_img, (out_w, out_h), (cx, cy), max_radius, flags)

    def _build_local_cost_volume(self, feat_map, ref_map):
        """Build a local matching-cost volume using neighborhood patches from ref_map."""
        k = self.cf_kernel_size
        b, c, h, w = ref_map.shape
        ref_unfold = F.unfold(ref_map, kernel_size=k, padding=k // 2)
        ref_unfold = ref_unfold.view(b, c, k * k, h, w).permute(0, 2, 1, 3, 4)
        cost_volume = (feat_map.unsqueeze(1) - ref_unfold).pow(2).mean(dim=2)
        return cost_volume

    def _refine_scores(self, raw_score_map, feat_map, ref_map, detach_input=False):
        if not self.use_costfilter or self.cost_filter is None:
            return raw_score_map, None

        if detach_input:
            feat_in = feat_map.detach()
            ref_in = ref_map.detach()
            raw_in = raw_score_map.detach()
        else:
            feat_in = feat_map
            ref_in = ref_map
            raw_in = raw_score_map

        cost_volume = self._build_local_cost_volume(feat_in, ref_in)
        cf_in = torch.cat([cost_volume, raw_in], dim=1)
        cf_logits = self.cost_filter(cf_in)
        cf_score = torch.sigmoid(cf_logits)
        refined_score = self.cf_lambda * raw_score_map + (1.0 - self.cf_lambda) * cf_score
        return refined_score, cf_logits

    def trainer(self, training_data, val_data, name):
        state_dict = {}
        ckpt_path = glob.glob(self.ckpt_dir + '/ckpt_best*')
        ckpt_path_save = os.path.join(self.ckpt_dir, "ckpt.pth")
        if len(ckpt_path) != 0:
            LOGGER.info("Start testing, ckpt file found!")
            return 0., 0., 0., 0., 0., -1.

        def update_state_dict():
            state_dict["discriminator"] = OrderedDict({
                k: v.detach().cpu()
                for k, v in self.discriminator.state_dict().items()})
            if self.pre_proj > 0:
                state_dict["pre_projection"] = OrderedDict({
                    k: v.detach().cpu()
                    for k, v in self.pre_projection.state_dict().items()})
            if self.use_costfilter and self.cost_filter is not None:
                state_dict["cost_filter"] = OrderedDict({
                    k: v.detach().cpu()
                    for k, v in self.cost_filter.state_dict().items()})
            if isinstance(self.c, torch.Tensor):
                state_dict["center"] = self.c.detach().cpu()

        self.distribution = training_data.dataset.distribution
        xlsx_path = './datasets/excel/' + name.split('_')[0] + '_distribution.xlsx'
        try:
            if self.distribution == 1:  # rejudge by image-level spectrogram analysis
                self.distribution = 1
                self.svd = 1
            elif self.distribution == 2:  # manifold
                self.distribution = 0
                self.svd = 0
            elif self.distribution == 3:  # hypersphere
                self.distribution = 0
                self.svd = 1
            elif self.distribution == 4:  # opposite choose by file
                self.distribution = 0
                df = pd.read_excel(xlsx_path)
                self.svd = 1 - df.loc[df['Class'] == name, 'Distribution'].values[0]
            else:  # choose by file
                self.distribution = 0
                df = pd.read_excel(xlsx_path)
                self.svd = df.loc[df['Class'] == name, 'Distribution'].values[0]
        except:
            self.distribution = 1
            self.svd = 1

        # judge by image-level spectrogram analysis
        if self.distribution == 1:
            self.forward_modules.eval()
            with torch.no_grad():
                for i, data in enumerate(training_data):
                    img = data["image"]
                    img = img.to(torch.float).to(self.device)
                    batch_mean = torch.mean(img, dim=0)
                    if i == 0:
                        self.c = batch_mean
                    else:
                        self.c += batch_mean
                self.c /= len(training_data)

            avg_img = utils.torch_format_2_numpy_img(self.c.detach().cpu().numpy())
            self.svd = utils.distribution_judge(avg_img, name)
            os.makedirs(f'./results/judge/avg/{self.svd}', exist_ok=True)
            cv2.imwrite(f'./results/judge/avg/{self.svd}/{name}.png', avg_img)
            return self.svd

        pbar = tqdm.tqdm(range(self.meta_epochs), unit='epoch')
        pbar_str1 = ""
        best_record = None
        for i_epoch in pbar:
            self.forward_modules.eval()
            with torch.no_grad():  # compute center
                for i, data in enumerate(training_data):
                    img = data["image"]
                    img = img.to(torch.float).to(self.device)
                    if self.pre_proj > 0:
                        outputs = self.pre_projection(self._embed(img, evaluation=False)[0])
                    else:
                        outputs = self._embed(img, evaluation=False)[0]

                    batch_mean = torch.mean(outputs, dim=0)
                    if i == 0:
                        self.c = batch_mean
                    else:
                        self.c += batch_mean
                self.c /= len(training_data)

            pbar_str, pt, pf = self._train_discriminator(training_data, i_epoch, pbar, pbar_str1)
            update_state_dict()

            if (i_epoch + 1) % self.eval_epochs == 0:
                images, scores, segmentations, labels_gt, masks_gt, img_paths = self.predict(val_data)
                image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro = self._evaluate(images, scores, segmentations,
                                                                                         labels_gt, masks_gt, name,
                                                                                         img_paths=img_paths,
                                                                                         dataset_obj=val_data.dataset)

                self.logger.logger.add_scalar("i-auroc", image_auroc, i_epoch)
                self.logger.logger.add_scalar("i-ap", image_ap, i_epoch)
                self.logger.logger.add_scalar("p-auroc", pixel_auroc, i_epoch)
                self.logger.logger.add_scalar("p-ap", pixel_ap, i_epoch)
                self.logger.logger.add_scalar("p-pro", pixel_pro, i_epoch)

                eval_path = './results/eval/' + name + '/'
                train_path = './results/training/' + name + '/'
                if best_record is None:
                    best_record = [image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro, i_epoch]
                    ckpt_path_best = os.path.join(self.ckpt_dir, "ckpt_best_{}.pth".format(i_epoch))
                    torch.save(state_dict, ckpt_path_best)
                    shutil.rmtree(eval_path, ignore_errors=True)
                    shutil.copytree(train_path, eval_path)

                elif image_auroc + pixel_auroc > best_record[0] + best_record[2]:
                    best_record = [image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro, i_epoch]
                    os.remove(ckpt_path_best)
                    ckpt_path_best = os.path.join(self.ckpt_dir, "ckpt_best_{}.pth".format(i_epoch))
                    torch.save(state_dict, ckpt_path_best)
                    shutil.rmtree(eval_path, ignore_errors=True)
                    shutil.copytree(train_path, eval_path)

                pbar_str1 = f" IAUC:{round(image_auroc * 100, 2)}({round(best_record[0] * 100, 2)})" \
                            f" IAP:{round(image_ap * 100, 2)}({round(best_record[1] * 100, 2)})" \
                            f" PAUC:{round(pixel_auroc * 100, 2)}({round(best_record[2] * 100, 2)})" \
                            f" PAP:{round(pixel_ap * 100, 2)}({round(best_record[3] * 100, 2)})" \
                            f" PRO:{round(pixel_pro * 100, 2)}({round(best_record[4] * 100, 2)})" \
                            f" E:{i_epoch}({best_record[-1]})"
                pbar_str += pbar_str1
                pbar.set_description_str(pbar_str)

            torch.save(state_dict, ckpt_path_save)
        return best_record

    def _train_discriminator(self, input_data, cur_epoch, pbar, pbar_str1):
        self.forward_modules.eval()
        if self.pre_proj > 0:
            self.pre_projection.train()
        self.discriminator.train()
        if self.use_costfilter and self.cost_filter is not None:
            self.cost_filter.train()

        all_loss, all_p_true, all_p_fake, all_r_t, all_r_g, all_r_f = [], [], [], [], [], []
        sample_num = 0
        for i_iter, data_item in enumerate(input_data):
            self.dsc_opt.zero_grad()
            if self.pre_proj > 0:
                self.proj_opt.zero_grad()
            if self.use_costfilter and self.cf_opt is not None:
                self.cf_opt.zero_grad()

            aug = data_item["aug"]
            aug = aug.to(torch.float).to(self.device)
            img = data_item["image"]
            img = img.to(torch.float).to(self.device)
            if self.pre_proj > 0:
                fake_feats = self.pre_projection(self._embed(aug, evaluation=False)[0])
                true_feats = self.pre_projection(self._embed(img, evaluation=False)[0])
            else:
                fake_feats = self._embed(aug, evaluation=False)[0]
                fake_feats.requires_grad = True
                true_feats = self._embed(img, evaluation=False)[0]
                true_feats.requires_grad = True

            _, _, feat_h, feat_w = true_feats.shape
            mask_s_gt_map = data_item["mask_s"].to(torch.float).unsqueeze(1).to(self.device)
            mask_s_gt_map = F.interpolate(mask_s_gt_map, size=(feat_h, feat_w), mode="nearest")
            mask_s_gt = self._flatten_mask(mask_s_gt_map)

            noise = torch.normal(0, self.noise, true_feats.shape).to(self.device)
            gaus_feats = true_feats + noise

            center = self.c.unsqueeze(0).repeat(img.shape[0], 1, 1, 1)
            fake_points_vec = self._flatten_hw(fake_feats)
            true_points_vec = self._flatten_hw(true_feats)
            center_vec = self._flatten_hw(center)

            true_points = torch.concat([fake_points_vec[mask_s_gt[:, 0] == 0], true_points_vec], dim=0)
            c_t_points = torch.concat([center_vec[mask_s_gt[:, 0] == 0], center_vec], dim=0)
            dist_t = torch.norm(true_points - c_t_points, dim=1)
            r_t = torch.tensor([torch.quantile(dist_t, q=self.radius)]).to(self.device)
            r_g = r_t

            for step in range(self.step + 1):
                scores = self.discriminator(torch.cat([true_feats, gaus_feats], dim=0))
                scores = self._flatten_mask(scores)
                split_idx = true_feats.shape[0] * feat_h * feat_w
                true_scores = scores[:split_idx]
                gaus_scores = scores[split_idx:]
                true_loss = torch.nn.BCELoss()(true_scores, torch.zeros_like(true_scores))
                gaus_loss = torch.nn.BCELoss()(gaus_scores, torch.ones_like(gaus_scores))
                bce_loss = true_loss + gaus_loss

                if step == self.step:
                    break
                elif self.mining == 0:
                    dist_g = torch.norm(gaus_feats - center, dim=1)
                    r_g = torch.tensor([torch.quantile(dist_g, q=self.radius)]).to(self.device)
                    break

                grad = torch.autograd.grad(gaus_loss, [gaus_feats])[0]
                grad_norm = torch.norm(grad, dim=1, keepdim=True)
                grad_normalized = grad / (grad_norm + 1e-10)

                with torch.no_grad():
                    gaus_feats.add_(0.001 * grad_normalized)

                if (step + 1) % 5 == 0:
                    dist_g = torch.norm(gaus_feats - center, dim=1)
                    r_g = torch.tensor([torch.quantile(dist_g, q=self.radius)]).to(self.device)
                    proj_feats = center if self.svd == 1 else true_feats
                    r = r_t if self.svd == 1 else 0.5

                    h = gaus_feats - proj_feats
                    h_norm = dist_g if self.svd == 1 else torch.norm(h, dim=1)
                    alpha = torch.clamp(h_norm, r, 2 * r)
                    proj = (alpha / (h_norm + 1e-10)).unsqueeze(1)
                    h = proj * h
                    gaus_feats = proj_feats + h

            fake_points_vec = self._flatten_hw(fake_feats)
            true_points_vec = self._flatten_hw(true_feats)
            center_vec = self._flatten_hw(center)

            fake_points = fake_points_vec[mask_s_gt[:, 0] == 1]
            true_points = true_points_vec[mask_s_gt[:, 0] == 1]
            c_f_points = center_vec[mask_s_gt[:, 0] == 1]
            dist_f = torch.norm(fake_points - c_f_points, dim=1)
            r_f = torch.tensor([torch.quantile(dist_f, q=self.radius)]).to(self.device)
            proj_feats = c_f_points if self.svd == 1 else true_points
            r = r_t if self.svd == 1 else 1

            if self.svd == 1:
                h = fake_points - proj_feats
                h_norm = dist_f if self.svd == 1 else torch.norm(h, dim=1)
                alpha = torch.clamp(h_norm, 2 * r, 4 * r)
                proj = (alpha / (h_norm + 1e-10)).view(-1, 1)
                h = proj * h
                fake_points = proj_feats + h
                fake_points_vec[mask_s_gt[:, 0] == 1] = fake_points
                fake_feats = fake_points_vec.reshape(img.shape[0], feat_h, feat_w, -1).permute(0, 3, 1, 2)

            fake_score_map = self.discriminator(fake_feats)
            fake_scores = self._flatten_mask(fake_score_map)
            if self.p > 0:
                fake_dist = (fake_scores - mask_s_gt) ** 2
                d_hard = torch.quantile(fake_dist, q=self.p)
                fake_scores_ = fake_scores[fake_dist >= d_hard].unsqueeze(1)
                mask_ = mask_s_gt[fake_dist >= d_hard].unsqueeze(1)
            else:
                fake_scores_ = fake_scores
                mask_ = mask_s_gt
            output = torch.cat([1 - fake_scores_, fake_scores_], dim=1)
            focal_loss = self.focal_loss(output, mask_)

            cf_loss = torch.tensor(0.0, device=self.device)
            if self.use_costfilter and self.cost_filter is not None:
                refined_score_map, cf_logits = self._refine_scores(
                    fake_score_map,
                    fake_feats,
                    center,
                    detach_input=True,
                )
                cf_loss = self.cf_bce(cf_logits, mask_s_gt_map)
            else:
                refined_score_map = fake_score_map

            loss = bce_loss + focal_loss + self.cf_weight * cf_loss
            loss.backward()
            if self.pre_proj > 0:
                self.proj_opt.step()
            if self.train_backbone:
                self.backbone_opt.step()
            self.dsc_opt.step()
            if self.use_costfilter and self.cf_opt is not None:
                self.cf_opt.step()

            pix_true = torch.concat([fake_scores.detach() * (1 - mask_s_gt), true_scores.detach()])
            pix_fake = torch.concat([fake_scores.detach() * mask_s_gt, gaus_scores.detach()])
            p_true = ((pix_true < self.dsc_margin).sum() - (pix_true == 0).sum()) / ((mask_s_gt == 0).sum() + true_scores.shape[0])
            p_fake = (pix_fake >= self.dsc_margin).sum() / ((mask_s_gt == 1).sum() + gaus_scores.shape[0])

            self.logger.logger.add_scalar(f"p_true", p_true, self.logger.g_iter)
            self.logger.logger.add_scalar(f"p_fake", p_fake, self.logger.g_iter)
            self.logger.logger.add_scalar(f"r_t", r_t, self.logger.g_iter)
            self.logger.logger.add_scalar(f"r_g", r_g, self.logger.g_iter)
            self.logger.logger.add_scalar(f"r_f", r_f, self.logger.g_iter)
            self.logger.logger.add_scalar("loss", loss, self.logger.g_iter)
            if self.use_costfilter and self.cost_filter is not None:
                self.logger.logger.add_scalar("cf_loss", cf_loss, self.logger.g_iter)
                self.logger.logger.add_scalar("cf_score_mean", refined_score_map.mean(), self.logger.g_iter)
            self.logger.step()

            all_loss.append(loss.detach().cpu().item())
            all_p_true.append(p_true.cpu().item())
            all_p_fake.append(p_fake.cpu().item())
            all_r_t.append(r_t.cpu().item())
            all_r_g.append(r_g.cpu().item())
            all_r_f.append(r_f.cpu().item())

            all_loss_ = np.mean(all_loss)
            all_p_true_ = np.mean(all_p_true)
            all_p_fake_ = np.mean(all_p_fake)
            all_r_t_ = np.mean(all_r_t)
            all_r_g_ = np.mean(all_r_g)
            all_r_f_ = np.mean(all_r_f)
            sample_num = sample_num + img.shape[0]

            pbar_str = f"epoch:{cur_epoch} loss:{all_loss_:.2e}"
            pbar_str += f" pt:{all_p_true_ * 100:.2f}"
            pbar_str += f" pf:{all_p_fake_ * 100:.2f}"
            pbar_str += f" rt:{all_r_t_:.2f}"
            pbar_str += f" rg:{all_r_g_:.2f}"
            pbar_str += f" rf:{all_r_f_:.2f}"
            pbar_str += f" svd:{self.svd}"
            pbar_str += f" sample:{sample_num}"
            pbar_str2 = pbar_str
            pbar_str += pbar_str1
            pbar.set_description_str(pbar_str)

            if sample_num > self.limit:
                break

        return pbar_str2, all_p_true_, all_p_fake_

    def tester(self, test_data, name):
        ckpt_path = glob.glob(self.ckpt_dir + '/ckpt_best*')
        if len(ckpt_path) != 0:
            state_dict = torch.load(ckpt_path[0], map_location=self.device)
            if 'discriminator' in state_dict:
                self.discriminator.load_state_dict(state_dict['discriminator'])
                if "pre_projection" in state_dict:
                    self.pre_projection.load_state_dict(state_dict["pre_projection"])
                if self.use_costfilter and self.cost_filter is not None and "cost_filter" in state_dict:
                    try:
                        self.cost_filter.load_state_dict(state_dict["cost_filter"])
                    except Exception as e:
                        LOGGER.warning(f"Skip loading cost_filter due to mismatch: {e}")
                if "center" in state_dict:
                    self.c = state_dict["center"].to(self.device)
            else:
                self.load_state_dict(state_dict, strict=False)

            images, scores, segmentations, labels_gt, masks_gt, img_paths = self.predict(test_data)
            image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro = self._evaluate(images, scores, segmentations,
                                                                                     labels_gt, masks_gt, name, path='eval',
                                                                                     img_paths=img_paths,
                                                                                     dataset_obj=test_data.dataset)
            epoch = int(ckpt_path[0].split('_')[-1].split('.')[0])
        else:
            image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro, epoch = 0., 0., 0., 0., 0., -1.
            LOGGER.info("No ckpt file found!")

        return image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro, epoch

    def _evaluate(self, images, scores, segmentations, labels_gt, masks_gt, name, path='training',
                  img_paths=None, dataset_obj=None):
        scores = np.squeeze(np.array(scores))
        image_scores = metrics.compute_imagewise_retrieval_metrics(scores, labels_gt, path)
        image_auroc = image_scores["auroc"]
        image_ap = image_scores["ap"]

        if len(masks_gt) > 0:
            segmentations = np.array(segmentations)
            pixel_scores = metrics.compute_pixelwise_retrieval_metrics(segmentations, masks_gt, path)
            pixel_auroc = pixel_scores["auroc"]
            pixel_ap = pixel_scores["ap"]
            if path == 'eval':
                try:
                    pixel_pro = metrics.compute_pro(np.squeeze(np.array(masks_gt)), segmentations)
                except:
                    pixel_pro = 0.
            else:
                pixel_pro = 0.
        else:
            pixel_auroc = -1.
            pixel_ap = -1.
            pixel_pro = -1.
            return image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro

        defects = np.array(images)
        targets = np.array(masks_gt)
        use_polar = bool(getattr(dataset_obj, "use_polar", False)) if dataset_obj is not None else False
        vis_save_size = int(getattr(dataset_obj, "vis_save_size", 256)) if dataset_obj is not None else 256
        for i in range(len(defects)):
            defect = utils.torch_format_2_numpy_img(defects[i])
            target = utils.torch_format_2_numpy_img(targets[i])
            seg_map = segmentations[i]

            if use_polar and img_paths is not None and i < len(img_paths):
                raw_bgr = cv2.imread(img_paths[i], cv2.IMREAD_COLOR)
                if raw_bgr is not None:
                    defect = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
                    out_h, out_w = defect.shape[:2]

                    if target.ndim == 3:
                        target_gray = target[:, :, 0]
                    else:
                        target_gray = target
                    target_cart = self._inverse_polar_image(
                        target_gray.astype(np.float32), out_h, out_w, dataset_obj, interpolation=cv2.INTER_NEAREST
                    )
                    target = np.repeat(target_cart[:, :, None], 3, axis=2)

                    seg_map = self._inverse_polar_image(
                        seg_map.astype(np.float32), out_h, out_w, dataset_obj, interpolation=cv2.INTER_LINEAR
                    )

            mask = cv2.cvtColor(cv2.resize(seg_map, (defect.shape[1], defect.shape[0])),
                                cv2.COLOR_GRAY2BGR)
            mask = (mask * 255).astype('uint8')
            mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)

            img_up = np.hstack([defect, target, mask])
            if vis_save_size > 0:
                img_up = cv2.resize(img_up, (vis_save_size * 3, vis_save_size))
            full_path = './results/' + path + '/' + name + '/'
            utils.del_remake_dir(full_path, del_flag=False)
            cv2.imwrite(full_path + str(i + 1).zfill(3) + '.png', img_up)

        return image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro

    def predict(self, test_dataloader):
        """This function provides anomaly scores/maps for full dataloaders."""
        self.forward_modules.eval()

        img_paths = []
        images = []
        scores = []
        masks = []
        labels_gt = []
        masks_gt = []

        with tqdm.tqdm(test_dataloader, desc="Inferring...", leave=False, unit='batch') as data_iterator:
            for data in data_iterator:
                if isinstance(data, dict):
                    labels_gt.extend(data["is_anomaly"].numpy().tolist())
                    if data.get("mask_gt", None) is not None:
                        masks_gt.extend(data["mask_gt"].numpy().tolist())
                    image = data["image"]
                    images.extend(image.numpy().tolist())
                    img_paths.extend(data["image_path"])
                _scores, _masks = self._predict(image)
                for score, mask in zip(_scores, _masks):
                    scores.append(score)
                    masks.append(mask)

        return images, scores, masks, labels_gt, masks_gt, img_paths

    def _predict(self, img):
        """Infer score and mask for a batch of images."""
        img = img.to(torch.float).to(self.device)
        self.forward_modules.eval()

        if self.pre_proj > 0:
            self.pre_projection.eval()
        self.discriminator.eval()
        if self.use_costfilter and self.cost_filter is not None:
            self.cost_filter.eval()

        with torch.no_grad():

            patch_features, _ = self._embed(img, provide_patch_shapes=True, evaluation=True)
            if self.pre_proj > 0:
                patch_features = self.pre_projection(patch_features)

            patch_score_map = self.discriminator(patch_features)
            if self.use_costfilter and self.cost_filter is not None and isinstance(self.c, torch.Tensor) and self.c.ndim == 3:
                center = self.c.unsqueeze(0).repeat(img.shape[0], 1, 1, 1).to(self.device)
                if center.shape[-2:] != patch_features.shape[-2:]:
                    center = F.interpolate(center, size=patch_features.shape[-2:], mode="bilinear", align_corners=False)
                patch_score_map, _ = self._refine_scores(patch_score_map, patch_features, center, detach_input=False)

            patch_scores = patch_score_map.squeeze(1)
            masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)

            image_scores = patch_scores.reshape(img.shape[0], -1).max(dim=1).values
            if isinstance(image_scores, torch.Tensor):
                image_scores = image_scores.cpu().numpy()

        return list(image_scores), list(masks)
