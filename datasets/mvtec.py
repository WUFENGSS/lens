from torchvision import transforms
from perlin import perlin_mask
from enum import Enum

import numpy as np
import pandas as pd

import PIL
import torch
import os
import glob
import cv2

_CLASSNAMES = [
    "carpet",
    "grid",
    "leather",
    "tile",
    "wood",
    "bottle",
    "cable",
    "capsule",
    "hazelnut",
    "metal_nut",
    "pill",
    "screw",
    "toothbrush",
    "transistor",
    "zipper",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DatasetSplit(Enum):
    TRAIN = "train"
    TEST = "test"


class MVTecDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for MVTec.
    """

    def __init__(
            self,
            source,
            anomaly_source_path='/root/dataset/dtd/images',
            real_anomaly_source_path='',
            dataset_name='mvtec',
            classname='leather',
            resize=288,
            imagesize=288,
            split=DatasetSplit.TRAIN,
            rotate_degrees=0,
            translate=0,
            brightness_factor=0,
            contrast_factor=0,
            saturation_factor=0,
            gray_p=0,
            h_flip_p=0,
            v_flip_p=0,
            distribution=0,
            mean=0.5,
            std=0.1,
            fg=0,
            rand_aug=1,
            downsampling=8,
            scale=0,
            batch_size=8,
                use_real_in_image_synth=0,
                real_anomaly_prob=0.5,
                use_polar=0,
                synth_in_cartesian=0,
                polar_inner_ratio=0.00,
                polar_outer_ratio=1.0,
                polar_max_radius_ratio=1.0,
                polar_center_x=-1.0,
                polar_center_y=-1.0,
                polar_ring_constraint=1,
                region_split_eval=0,
                aperture_ratio=0.25,
                region_center_x=-1.0,
                region_center_y=-1.0,
                    vis_save_size=256,
            **kwargs,
    ):
        """
        Args:
            source: [str]. Path to the MVTec data folder.
            classname: [str or None]. Name of MVTec class that should be
                       provided in this dataset. If None, the datasets
                       iterates over all available images.
            resize: [int]. (Square) Size the loaded image initially gets
                    resized to.
            imagesize: [int]. (Square) Size the resized loaded image gets
                       (center-)cropped to.
            split: [enum-option]. Indicates if training or test split of the
                   data should be used. Has to be an option taken from
                   DatasetSplit, e.g. mvtec.DatasetSplit.TRAIN. Note that
                   mvtec.DatasetSplit.TEST will also load mask data.
        """
        super().__init__()
        self.source = source
        self.split = split
        self.batch_size = batch_size
        self.distribution = distribution
        self.mean = mean
        self.std = std
        self.fg = fg
        self.rand_aug = rand_aug
        self.downsampling = downsampling
        self.resize = resize if self.distribution != 1 else [resize, resize]
        self.imgsize = imagesize
        self.imagesize = (3, self.imgsize, self.imgsize)
        self.classname = classname
        self.dataset_name = dataset_name
        self.use_real_in_image_synth = bool(use_real_in_image_synth)
        self.real_anomaly_prob = float(np.clip(real_anomaly_prob, 0.0, 1.0))
        self.use_polar = bool(use_polar)
        self.synth_in_cartesian = bool(synth_in_cartesian)
        self.polar_inner_ratio = float(polar_inner_ratio)
        self.polar_outer_ratio = float(polar_outer_ratio)
        self.polar_max_radius_ratio = float(polar_max_radius_ratio)
        self.polar_center_x = float(polar_center_x)
        self.polar_center_y = float(polar_center_y)
        self.polar_ring_constraint = bool(polar_ring_constraint)
        self.region_split_eval = bool(region_split_eval)
        self.aperture_ratio = float(np.clip(aperture_ratio, 0.0, 1.0))
        self.region_center_x = float(region_center_x)
        self.region_center_y = float(region_center_y)
        self.vis_save_size = int(vis_save_size)
        self.fpr_target_tpr = float(kwargs.get('fpr_target_tpr', 0.98))
        self._polar_ring_cache = {}

        if self.distribution != 1 and (self.classname == 'toothbrush' or self.classname == 'wood'):
            self.resize = round(self.imgsize * 329 / 288)

        xlsx_path = './datasets/excel/' + self.dataset_name + '_distribution.xlsx'
        if self.fg == 2:  # choose by file
            try:
                df = pd.read_excel(xlsx_path)
                self.class_fg = df.loc[df['Class'] == self.dataset_name + '_' + classname, 'Foreground'].values[0]
            except:
                self.class_fg = 1
        elif self.fg == 1:  # with foreground mask
            self.class_fg = 1
        else:  # without foreground mask
            self.class_fg = 0

        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()
        self.anomaly_source_paths = sorted(1 * glob.glob(anomaly_source_path + "/*/*.jpg") +
                                           0 * list(next(iter(self.imgpaths_per_class.values())).values())[0])
        self.real_anomaly_source_paths = self._collect_real_anomaly_paths(real_anomaly_source_path)

        self.transform_img = [
            transforms.Resize(self.resize),
            transforms.ColorJitter(brightness_factor, contrast_factor, saturation_factor),
            transforms.RandomHorizontalFlip(h_flip_p),
            transforms.RandomVerticalFlip(v_flip_p),
            transforms.RandomGrayscale(gray_p),
            transforms.RandomAffine(rotate_degrees,
                                    translate=(translate, translate),
                                    scale=(1.0 - scale, 1.0 + scale),
                                    interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.imgsize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)

        self.transform_mask = [
            transforms.Resize(self.resize),
            transforms.CenterCrop(self.imgsize),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)

    def _collect_real_anomaly_paths(self, real_anomaly_source_path):
        if not real_anomaly_source_path:
            return []

        patterns = ["**/*.jpg", "**/*.jpeg", "**/*.png", "**/*.bmp", "**/*.tif", "**/*.tiff"]
        file_paths = []
        for pattern in patterns:
            file_paths.extend(glob.glob(os.path.join(real_anomaly_source_path, pattern), recursive=True))

        file_paths = sorted([path for path in file_paths if os.path.isfile(path)])
        return file_paths

    def _sample_anomaly_source_image(self):
        use_real_source = (
                self.use_real_in_image_synth
                and len(self.real_anomaly_source_paths) > 0
                and np.random.rand() < self.real_anomaly_prob
        )
        source_paths = self.real_anomaly_source_paths if use_real_source else self.anomaly_source_paths
        if len(source_paths) == 0:
            source_paths = self.anomaly_source_paths if len(self.anomaly_source_paths) > 0 else self.real_anomaly_source_paths
        if len(source_paths) == 0:
            raise RuntimeError("No anomaly source images found for image-level synthesis.")

        aug = PIL.Image.open(np.random.choice(source_paths)).convert("RGB")
        if self.rand_aug:
            transform_aug = self.rand_augmenter()
            aug = transform_aug(aug)
        else:
            aug = self.transform_img(aug)

        return aug

    def _get_polar_center(self, h, w):
        cx = (w - 1) / 2.0 if self.polar_center_x < 0 else self.polar_center_x
        cy = (h - 1) / 2.0 if self.polar_center_y < 0 else self.polar_center_y
        cx = float(np.clip(cx, 0.0, w - 1))
        cy = float(np.clip(cy, 0.0, h - 1))
        return cx, cy

    def _polar_warp_tensor(self, tensor, interpolation=cv2.INTER_LINEAR, binarize=False):
        """Warp CHW or HW tensor to polar coordinates with fixed output size."""
        if not self.use_polar:
            return tensor

        if tensor.ndim == 3:
            src = tensor.permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
            h, w = src.shape[:2]
        else:
            src = tensor.detach().cpu().numpy().astype(np.float32)
            h, w = src.shape[:2]

        cx, cy = self._get_polar_center(h, w)
        max_radius = min(cx, cy, (w - 1) - cx, (h - 1) - cy)
        max_radius = max(1.0, max_radius * self.polar_max_radius_ratio)

        flags = cv2.WARP_POLAR_LINEAR + cv2.WARP_FILL_OUTLIERS + interpolation
        dst = cv2.warpPolar(src, (w, h), (cx, cy), max_radius, flags)

        if tensor.ndim == 3:
            dst = torch.from_numpy(dst).permute(2, 0, 1).to(tensor.dtype)
        else:
            dst = torch.from_numpy(dst).to(tensor.dtype)

        if binarize:
            dst = torch.where(dst > 0.5, torch.ones_like(dst), torch.zeros_like(dst))
        return dst

    def _get_polar_ring_mask(self, h, w):
        """Create annulus mask in Cartesian space and warp it to polar space."""
        key = (
            h,
            w,
            round(self.polar_center_x, 4),
            round(self.polar_center_y, 4),
            round(self.polar_inner_ratio, 4),
            round(self.polar_outer_ratio, 4),
            round(self.polar_max_radius_ratio, 4),
        )
        if key in self._polar_ring_cache:
            return self._polar_ring_cache[key].clone()

        cx, cy = self._get_polar_center(h, w)
        max_radius = min(cx, cy, (w - 1) - cx, (h - 1) - cy)
        max_radius = max(1.0, max_radius * self.polar_max_radius_ratio)

        r_in = max(0.0, self.polar_inner_ratio * max_radius)
        r_out = max(r_in + 1e-6, self.polar_outer_ratio * max_radius)

        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        annulus = ((rr >= r_in) & (rr <= r_out)).astype(np.float32)

        annulus_tensor = torch.from_numpy(annulus)
        polar_annulus = self._polar_warp_tensor(
            annulus_tensor, interpolation=cv2.INTER_NEAREST, binarize=True
        )
        polar_annulus = torch.where(
            polar_annulus > 0.5,
            torch.ones_like(polar_annulus),
            torch.zeros_like(polar_annulus),
        )

        self._polar_ring_cache[key] = polar_annulus
        return polar_annulus.clone()

    def _get_cartesian_ring_mask(self, h, w):
        """Create annulus mask in Cartesian space."""
        cx, cy = self._get_polar_center(h, w)
        max_radius = min(cx, cy, (w - 1) - cx, (h - 1) - cy)
        max_radius = max(1.0, max_radius * self.polar_max_radius_ratio)

        r_in = max(0.0, self.polar_inner_ratio * max_radius)
        r_out = max(r_in + 1e-6, self.polar_outer_ratio * max_radius)

        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        annulus = ((rr >= r_in) & (rr <= r_out)).astype(np.float32)
        return torch.from_numpy(annulus)

    def rand_augmenter(self):
        list_aug = [
            transforms.ColorJitter(contrast=(0.8, 1.2)),
            transforms.ColorJitter(brightness=(0.8, 1.2)),
            transforms.ColorJitter(saturation=(0.8, 1.2), hue=(-0.2, 0.2)),
            transforms.RandomHorizontalFlip(p=1),
            transforms.RandomVerticalFlip(p=1),
            transforms.RandomGrayscale(p=1),
            transforms.RandomAutocontrast(p=1),
            transforms.RandomEqualize(p=1),
            transforms.RandomAffine(degrees=(-45, 45)),
        ]
        aug_idx = np.random.choice(np.arange(len(list_aug)), 3, replace=False)

        transform_aug = [
            transforms.Resize(self.resize),
            list_aug[aug_idx[0]],
            list_aug[aug_idx[1]],
            list_aug[aug_idx[2]],
            transforms.CenterCrop(self.imgsize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]

        transform_aug = transforms.Compose(transform_aug)
        return transform_aug

    def __getitem__(self, idx):
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)
        if self.use_polar and not self.synth_in_cartesian:
            image = self._polar_warp_tensor(image, interpolation=cv2.INTER_LINEAR)

        mask_fg = mask_s = aug_image = torch.tensor([1])
        if self.split == DatasetSplit.TRAIN:
            aug = self._sample_anomaly_source_image()

            if self.use_polar and not self.synth_in_cartesian:
                aug = self._polar_warp_tensor(aug, interpolation=cv2.INTER_LINEAR)

            if self.class_fg:
                fgmask_path = image_path.split(classname)[0] + 'fg_mask/' + classname + '/' + os.path.split(image_path)[-1]
                mask_fg = PIL.Image.open(fgmask_path)
                mask_fg = torch.ceil(self.transform_mask(mask_fg)[0])
                if self.use_polar and not self.synth_in_cartesian:
                    mask_fg = self._polar_warp_tensor(
                        mask_fg, interpolation=cv2.INTER_NEAREST, binarize=True
                    )
            else:
                mask_fg = torch.ones(image.shape[-2:], dtype=torch.float32)

            if self.use_polar and self.polar_ring_constraint:
                if self.synth_in_cartesian:
                    ring_mask = self._get_cartesian_ring_mask(image.shape[-2], image.shape[-1])
                else:
                    ring_mask = self._get_polar_ring_mask(image.shape[-2], image.shape[-1])
                mask_fg = mask_fg * ring_mask
                mask_fg = torch.where(mask_fg > 0.5, 1.0, 0.0)

            mask_all = perlin_mask(image.shape, self.imgsize // self.downsampling, 0, 6, mask_fg, 1)
            mask_s = torch.from_numpy(mask_all[0])
            mask_l = torch.from_numpy(mask_all[1])

            beta = np.random.normal(loc=self.mean, scale=self.std)
            beta = np.clip(beta, .2, .8)
            aug_image = image * (1 - mask_l) + (1 - beta) * aug * mask_l + beta * image * mask_l

            if self.use_polar and self.synth_in_cartesian:
                image = self._polar_warp_tensor(image, interpolation=cv2.INTER_LINEAR)
                aug_image = self._polar_warp_tensor(aug_image, interpolation=cv2.INTER_LINEAR)
                mask_s = self._polar_warp_tensor(mask_s, interpolation=cv2.INTER_NEAREST, binarize=True)

        if self.split == DatasetSplit.TEST and mask_path is not None:
            mask_gt = PIL.Image.open(mask_path).convert('L')
            mask_gt = self.transform_mask(mask_gt)
            if self.use_polar:
                mask_gt = self._polar_warp_tensor(
                    mask_gt[0], interpolation=cv2.INTER_NEAREST, binarize=True
                ).unsqueeze(0)
        else:
            mask_gt = torch.zeros([1, *image.size()[1:]])

        if self.split == DatasetSplit.TEST and self.use_polar and self.synth_in_cartesian:
            image = self._polar_warp_tensor(image, interpolation=cv2.INTER_LINEAR)

        return {
            "image": image,
            "aug": aug_image,
            "mask_s": mask_s,
            "mask_gt": mask_gt,
            "is_anomaly": int(anomaly != "good"),
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        imgpaths_per_class = {}
        maskpaths_per_class = {}

        classpath = os.path.join(self.source, self.classname, self.split.value)
        maskpath = os.path.join(self.source, self.classname, "ground_truth")
        anomaly_types = os.listdir(classpath)

        imgpaths_per_class[self.classname] = {}
        maskpaths_per_class[self.classname] = {}

        for anomaly in anomaly_types:
            anomaly_path = os.path.join(classpath, anomaly)
            anomaly_files = sorted(os.listdir(anomaly_path))
            imgpaths_per_class[self.classname][anomaly] = [os.path.join(anomaly_path, x) for x in anomaly_files]

            if self.split == DatasetSplit.TEST and anomaly != "good":
                anomaly_mask_path = os.path.join(maskpath, anomaly)
                anomaly_mask_files = sorted(os.listdir(anomaly_mask_path))
                maskpaths_per_class[self.classname][anomaly] = [os.path.join(anomaly_mask_path, x) for x in anomaly_mask_files]
            else:
                maskpaths_per_class[self.classname]["good"] = None

        data_to_iterate = []
        for classname in sorted(imgpaths_per_class.keys()):
            for anomaly in sorted(imgpaths_per_class[classname].keys()):
                for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
                    data_tuple = [classname, anomaly, image_path]
                    if self.split == DatasetSplit.TEST and anomaly != "good":
                        data_tuple.append(maskpaths_per_class[classname][anomaly][i])
                    else:
                        data_tuple.append(None)
                    data_to_iterate.append(data_tuple)

        return imgpaths_per_class, data_to_iterate
