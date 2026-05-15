
import os.path
import cv2
from PIL import Image
from .base_dataset import BaseDataset
import torchvision.transforms as transforms
from .image_folder import make_dataset
from .utils import MaskToTensor
import numpy as np
import torch
from tqdm import tqdm
import random  # 新增：用于随机选择增强方式


class CrackDataset(BaseDataset):
    """A dataset class for crack dataset."""

    def __init__(self, args):
        """Initialize this dataset class.

        Parameters:
            args (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, args)
        self.img_paths = make_dataset(os.path.join(args.dataset_path, '{}_img'.format(args.phase)))
        self.lab_dir = os.path.join(args.dataset_path, '{}_lab'.format(args.phase))
        self.img_transforms = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.5, 0.5, 0.5),
                                                                       (0.5, 0.5, 0.5))])
        self.lab_transform = MaskToTensor()
        self.phase = args.phase
        self.augment = True  # 是否启用数据增强

    def augment_single_sample(self, image, mask):
        h, w = image.shape[:2]
        original_size = (w, h)

        angle = random.choice([0, 45, 90, 135, 180, 225, 270, 315])

        if angle != 0:
            angle_rad = np.deg2rad(angle)
            cos_a = abs(np.cos(angle_rad))
            sin_a = abs(np.sin(angle_rad))
            new_w = int(w * cos_a + h * sin_a)
            new_h = int(h * cos_a + w * sin_a)
            new_center = (new_w // 2, new_h // 2)
            rotation_matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            rotation_matrix[0, 2] += (new_w - w) / 2
            rotation_matrix[1, 2] += (new_h - h) / 2
            rotated_image = cv2.warpAffine(
                image, rotation_matrix, (new_w, new_h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0)
            )
            rotated_mask = cv2.warpAffine(
                mask, rotation_matrix, (new_w, new_h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )

            start_x = (new_w - w) // 2
            start_y = (new_h - h) // 2
            end_x = min(start_x + w, new_w)
            end_y = min(start_y + h, new_h)
            start_x = max(0, end_x - w)
            start_y = max(0, end_y - h)

            aug_image = rotated_image[start_y:end_y, start_x:end_x]
            aug_mask = rotated_mask[start_y:end_y, start_x:end_x]

            if aug_image.shape[:2] != (h, w):
                aug_image = cv2.resize(aug_image, original_size, interpolation=cv2.INTER_CUBIC)
                aug_mask = cv2.resize(aug_mask, original_size, interpolation=cv2.INTER_NEAREST)
        else:
            aug_image = image.copy()
            aug_mask = mask.copy()

        if random.random() < 0.5:
            aug_image = np.fliplr(aug_image).copy()
            aug_mask = np.fliplr(aug_mask).copy()

        assert aug_image.shape[:2] == (h, w), f"图像尺寸错误：{aug_image.shape[:2]} != ({h}, {w})"
        assert aug_mask.shape[:2] == (h, w), f"掩码尺寸错误：{aug_mask.shape[:2]} != ({h}, {w})"

        return aug_image, aug_mask

    def __getitem__(self, index):
        """
        Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            image (tensor) - - an image (shape: [3, H, W])
            label (tensor) - - its corresponding segmentation (shape: [1, H, W])
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        img_path = self.img_paths[index]
        lab_path = os.path.join(self.lab_dir, os.path.basename(img_path).split('.')[0] + '.png')

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        lab = cv2.imread(lab_path, cv2.IMREAD_UNCHANGED)

        if len(lab.shape) == 3:
            lab = cv2.cvtColor(lab, cv2.COLOR_BGR2GRAY)

        # adjust the image size
        w, h = self.args.load_width, self.args.load_height
        if w > 0 or h > 0:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
            lab = cv2.resize(lab, (w, h), interpolation=cv2.INTER_CUBIC)

        _, lab = cv2.threshold(lab, 127, 255, cv2.THRESH_BINARY)
        _, lab = cv2.threshold(lab, 127, 1, cv2.THRESH_BINARY)

        if self.augment:
            aug_img, aug_lab = self.augment_single_sample(img, lab)
            img = self.img_transforms(Image.fromarray(aug_img))
            lab = self.lab_transform(aug_lab).unsqueeze(0)
        else:
            img = self.img_transforms(Image.fromarray(img))
            lab = self.lab_transform(lab).unsqueeze(0)

        return {'image': img, 'label': lab, 'A_paths': img_path, 'B_paths': lab_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.img_paths)