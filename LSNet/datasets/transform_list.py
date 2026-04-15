import numpy as np
import random
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from PIL import Image
import numbers
import collections


class RandomCropNumpy(object):
    """随机裁剪numpy数组（H×W×C格式）"""

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, imgs):
        h, w = imgs[0].shape[:2]
        th, tw = self.size

        if w == tw and h == th:
            return imgs

        # 随机选择裁剪位置
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        results = []
        for img in imgs:
            if img is not None:
                if len(img.shape) == 3:
                    img = img[y1:y1 + th, x1:x1 + tw, :]
                else:
                    img = img[y1:y1 + th, x1:x1 + tw]
            results.append(img)

        return results


class RandomHorizontalFlip(object):
    """随机水平翻转numpy数组"""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, imgs):
        if random.random() < self.p:
            results = []
            for img in imgs:
                if img is not None:
                    img = np.fliplr(img).copy()
                results.append(img)
            return results
        return imgs


class RandomColor(object):
    """随机调整颜色（亮度、对比度和饱和度）"""

    def __init__(self, multiplier_range=(0.8, 1.2), brightness_mult_range=(0.75, 1.25)):
        self.multiplier_range = multiplier_range
        self.brightness_mult_range = brightness_mult_range

    def __call__(self, img):
        if img is None:
            return None

        # 转换为PIL图像进行处理
        img_pil = Image.fromarray((img * 255).astype(np.uint8))

        # 随机亮度调整
        brightness_multiplier = random.uniform(*self.brightness_mult_range)
        img_pil = F.adjust_brightness(img_pil, brightness_multiplier)

        # 随机对比度调整
        contrast_multiplier = random.uniform(*self.multiplier_range)
        img_pil = F.adjust_contrast(img_pil, contrast_multiplier)

        # 随机饱和度调整
        saturation_multiplier = random.uniform(*self.multiplier_range)
        img_pil = F.adjust_saturation(img_pil, saturation_multiplier)

        # 转换回numpy数组
        img = np.array(img_pil).astype(np.float32) / 255.0
        return img


class ArrayToTensorNumpy(object):
    """将numpy数组转换为张量"""

    def __call__(self, imgs):
        results = []
        for img in imgs:
            if img is not None:
                # 处理不同维度的数组
                if len(img.shape) == 3:
                    img = torch.from_numpy(img.transpose((2, 0, 1)))
                else:
                    img = torch.from_numpy(img[None, :, :])
            results.append(img)
        return results


# 从torchvision.transforms导入Normalize
Normalize = transforms.Normalize


class EnhancedCompose(object):
    """增强的组合变换，可以处理多个输入并应用不同的变换"""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, imgs):
        for t in self.transforms:
            # 处理列表中的每个变换
            if isinstance(t, collections.Sequence):
                # 如果变换是一个序列，则对每个输入应用对应的变换
                assert len(t) == len(imgs), "变换序列长度必须与输入数量相同"
                new_imgs = []
                for i, transform in enumerate(t):
                    if transform is not None:
                        new_imgs.append(transform(imgs[i]))
                    else:
                        new_imgs.append(imgs[i])
                imgs = new_imgs
            else:
                # 如果变换不是序列，则对所有输入应用相同的变换
                imgs = t(imgs)
        return imgs