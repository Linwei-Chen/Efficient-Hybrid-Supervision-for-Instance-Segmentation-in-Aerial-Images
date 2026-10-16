from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import torch.nn.functional as func
import math
import os
from os import path as osp
import numbers
import random
import torchvision
from torchvision import transforms
import torch
import numpy as np
from PIL import Image
import collections

# Mean and std from
# https://github.com/zijundeng/pytorch-semantic-segmentation/blob/master/train/voc-fcn/train.py
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
}


# NEAREST：最近滤波。从输入图像中选取最近的像素作为输出像素。它忽略了所有其他的像素。
# BILINEAR：双线性滤波。在输入图像的2x2矩阵上进行线性插值。注意：PIL的当前版本，做下采样时该滤波器使用了固定输入模板。
# BICUBIC：双立方滤波。在输入图像的4x4矩阵上进行立方插值。注意：PIL的当前版本，做下采样时该滤波器使用了固定输入模板。
# ANTIALIAS：平滑滤波。这是PIL 1.1.3版本中新的滤波器。对所有可以影响输出像素的输入像素进行高质量的重采样滤波，以计算输出像素值。在当前的PIL版本中，这个滤波器只用于改变尺寸和缩略图方法。


class CropRandomObject:

    def __init__(self, keys=[], object_key="instance", size=100):
        self.keys = keys
        self.object_key = object_key
        self.size = size

    def __call__(self, sample):
        """
        sample by label uniquely
        :param sample:
        :return:
        """

        object_map = np.array(sample[self.object_key], copy=False)
        h, w = object_map.shape

        unique_objects = np.unique(object_map)
        unique_objects = unique_objects[unique_objects != 0]

        label = np.array(sample['label'])
        unique_labels = np.unique(label)
        unique_labels = unique_labels[unique_labels != 0]
        instance_to_label = {int(i): int(np.unique(label[object_map == i])) for i in unique_objects}
        # print('instance_to_label',instance_to_label)
        label_to_instance = {}
        for i in unique_objects:
            i = int(i)
            if label_to_instance.get(instance_to_label[i]) is None:
                label_to_instance[instance_to_label[i]] = []
                label_to_instance[instance_to_label[i]].append(i)
            else:
                label_to_instance[instance_to_label[i]].append(i)
        # print('label_to_instance', label_to_instance)

        if unique_labels.size > 0:
            random_label = np.random.choice(unique_labels, 1)
            random_id = np.random.choice(label_to_instance[int(random_label)], 1)

            y, x = np.where(object_map == random_id)
            ym, xm = np.mean(y), np.mean(x)

            i = int(np.clip(ym - self.size[1] / 2, 0, h - self.size[1]))
            j = int(np.clip(xm - self.size[0] / 2, 0, w - self.size[0]))

        else:
            i = random.randint(0, h - self.size[1])
            j = random.randint(0, w - self.size[0])

        for k in self.keys:
            assert (k in sample)

            sample[k] = F.crop(sample[k], i, j, self.size[1], self.size[0])

        return sample


class RandomResizedCrop(object):
    """Crop the given PIL Image to random size and aspect ratio.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, scale=(0.125, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        self.size = (size, size)
        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback
        w = min(img.size[0], img.size[1])
        i = (img.size[1] - w) // 2
        j = (img.size[0] - w) // 2
        return i, j, w, w

    def __call__(self, img, target, bbox_target):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        img = F.resized_crop(img, i, j, h, w, self.size, self.interpolation)
        target = F.resized_crop(target, i, j, h, w, self.size, Image.NEAREST)
        if isinstance(bbox_target, torch.Tensor):
            bbox_target = func.interpolate(bbox_target[:, i:i + h, j:j + w].unsqueeze(dim=0),
                                           size=self.size, mode='nearest').squeeze(0)
        else:
            bbox_target = F.resized_crop(bbox_target, i, j, h, w, self.size, Image.NEAREST)
        # img.show()
        # target.show()
        return img, target, bbox_target

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string


class Resize(object):
    """Resize the input PIL Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, stride=32, resize_both=True, down_sample=None, interpolation=Image.BILINEAR):
        self.stride = stride
        self.interpolation = interpolation
        self.down_sample = down_sample
        self.resize_target = resize_both

    def __call__(self, img, target, bbox_target):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
        w, h = img.size
        if self.down_sample is not None:
            w, h = w / self.down_sample, h / self.down_sample
        w_stride, h_stride = math.floor(w / self.stride), math.floor(h / self.stride)
        h_w_resize = (int(h_stride * self.stride), int(w_stride * self.stride))
        img = F.resize(img, h_w_resize, self.interpolation)
        if isinstance(bbox_target, torch.Tensor):
            bbox_target = func.interpolate(bbox_target.unsqueeze(dim=0), h_w_resize, mode='nearest').squeeze(0)
        else:
            bbox_target = F.resize(bbox_target, h_w_resize, Image.NEAREST)

        if self.resize_target:
            target = F.resize(target, h_w_resize, Image.NEAREST)
        # img_resized.show()
        return img, target, bbox_target

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)


class RandomRotation(object):
    """Rotate the image by angle.

    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter.
            See http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#filters
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
    """

    def __init__(self, degrees=10, resample=Image.BILINEAR, expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, img, target, bbox_target):
        """
            img (PIL Image): Image to be rotated.

        Returns:
            PIL Image: Rotated image.
        """

        angle = self.get_params(self.degrees)
        img = F.rotate(img, angle, self.resample, self.expand, self.center)
        target = F.rotate(target, angle, Image.NEAREST, self.expand, self.center)
        if isinstance(bbox_target, torch.Tensor):
            w, h = target.size
            res = torch.zeros((bbox_target.size(0), h, w))
            for i in range(bbox_target.size(0)):
                temp = Image.fromarray(bbox_target[i, :, :].numpy())
                temp = F.rotate(temp, angle, Image.NEAREST, self.expand, self.center)
                res[i, :, :] = torch.from_numpy(np.array(temp))

            bbox_target = res
        else:
            bbox_target = F.rotate(bbox_target, angle, Image.NEAREST, self.expand, self.center)
        # img.show()
        # target.show()
        return img, target, bbox_target

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string


class ToTensor(object):
    def __init__(self, del_ignore=False, show=False):
        self.del_ignore = del_ignore
        self.show = show

    def __call__(self, img: Image, target: Image, bbox_target: Image):
        """

        :param: img, target Image
        :return: if input is image return tensor[bs, 3, h, w],
        else if input is segment targets, return tensor[bs, 1, h, w]
        """
        if self.show:
            img.show(),
            target = target.convert('P')
            target.show()
            if isinstance(bbox_target, torch.Tensor):
                transforms.ToPILImage()(bbox_target.argmax(dim=0).float()*11).show()
                bbox_target.argmax(dim=0)
            else:
                bbox_target = bbox_target.convert('L')
                bbox_target.show()
        img = transforms.ToTensor()(img)
        # for target
        # print(target.type())
        # del 255.
        target = torch.from_numpy(np.array(target)).long()
        if isinstance(bbox_target, torch.Tensor):
            pass
        else:
            bbox_target = torch.from_numpy(np.array(bbox_target)).long().unsqueeze(dim=0)
        if self.del_ignore:
            zeros = torch.zeros(target.shape).long()
            target = torch.where(target <= 20, target, zeros)

        target = target.unsqueeze(dim=0)

        return img, target, bbox_target


class Compose2(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target, bbox_target):
        for t in self.transforms:
            img, target, bbox_target = t(img, target, bbox_target)
        return img, target, bbox_target

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class RandomScale(object):
    def __init__(self, scales=(0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0), interpolation=Image.BILINEAR):
        self.scales = scales
        self.interpolation = interpolation

    def __call__(self, img, target, bbox_target):
        W, H = img.size
        scale = random.choice(self.scales)
        w, h = int(W * scale), int(H * scale)
        img_resized = F.resize(img, (h, w), self.interpolation)
        target_resized = F.resize(target, (h, w), Image.NEAREST)
        bbox_target_resized = F.resize(bbox_target, (h, w), Image.NEAREST)
        return img_resized, target_resized, bbox_target_resized


class RandomCrop(object):
    """Crop the given PIL Image at a random location.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception.
    """

    def __init__(self, size, padding=0, pad_if_needed=False):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img, target, bbox_target):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        if self.padding > 0:
            img = F.pad(img, self.padding)
            target = F.pad(target, self.padding)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, (int((1 + self.size[1] - img.size[0]) / 2), 0))
            target = F.pad(target, (int((1 + self.size[1] - target.size[0]) / 2), 0))
            bbox_target = F.pad(bbox_target, (int((1 + self.size[1] - bbox_target.size[0]) / 2), 0))
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, (0, int((1 + self.size[0] - img.size[1]) / 2)))
            target = F.pad(target, (0, int((1 + self.size[0] - target.size[1]) / 2)))
            bbox_target = F.pad(bbox_target, (0, int((1 + self.size[0] - bbox_target.size[1]) / 2)))

        i, j, h, w = self.get_params(img, self.size)

        return F.crop(img, i, j, h, w), F.crop(target, i, j, h, w), F.crop(bbox_target, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target, bbox_target):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            img, target = F.hflip(img), F.hflip(target)
            if isinstance(bbox_target, torch.Tensor):
                bbox_target = bbox_target.numpy()
                bbox_target = bbox_target[:, :, ::-1].copy()
                bbox_target = torch.from_numpy(bbox_target)
            else:
                bbox_target = F.hflip(bbox_target)
        # img.show(), target.show()
        return img, target, bbox_target

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, target, bbox_target):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        # print(type(img), type(target), type(bbox_target))
        return F.normalize(img, self.mean, self.std), target, bbox_target

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class CenterCrop(object):
    """Crops the given PIL Image at the center.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, stride=32.):
        self.stride = stride

    def __call__(self, img, target, bbox_target):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        w, h = img.size
        w_stride, h_stride = math.floor(w / self.stride), math.floor(h / self.stride)
        h_w_resize = (int(h_stride * self.stride), int(w_stride * self.stride))
        return F.center_crop(img, h_w_resize), F.center_crop(target, h_w_resize), F.center_crop(bbox_target, h_w_resize)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class CutOut:
    def __init__(self, size, mean, p=0.5):
        self.mean = mean
        self.p = p
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    @staticmethod
    def get_params(img, paste_size):
        """Get parameters for left, upper, right, and lower

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """

        w, h = img.size
        l = random.randint(0, w) - paste_size[0] // 2
        u = random.randint(0, h) - paste_size[1] // 2
        l = max(0, l)
        u = max(0, u)
        r = min(l + paste_size[0], w)
        d = min(u + paste_size[1], h)
        return l, u, r, d

    def __call__(self, img: Image, target: Image):
        if random.random() < self.p:
            l, u, r, d = self.get_params(img, self.size)
            img.paste(tuple([int(i * 255) for i in self.mean]), (l, u, r, d))
            target.paste(0, (l, u, r, d))
        return img, target


class SampleParing:
    def __init__(self, dataset: Dataset, p=0.5, alpha=1):
        self.dataset = dataset
        self.p = p
        self.alpha = alpha

    def get_pairing(self):
        index = random.choice(range(len(self.dataset)))
        pairing_img, pairing_target = self.dataset.__getitem__(index)
        return pairing_img

    def __call__(self, img: torch.Tensor, target: torch.Tensor):
        if random.uniform(0, 1) < self.p:
            pairing_img = self.get_pairing()
            assert pairing_img.shape == img.shape
            img = (img + self.get_pairing() * self.alpha) / 2
        return img, target


def _test_cutout():
    t = Image.open('/Users/chenlinwei/Desktop/2007_000033.png')
    print(t.mode)
    i = Image.open('/Users/chenlinwei/Desktop/2007_000033.jpg')
    print(i.mode)
    co = CutOut(128, MEAN)
    i, t = co(i, t)
    i.show()
    t.show()


def _test_sample_pairing():
    from dataset.voc_dataset import get_voc_data_set
    from train_utils import config
    sp = SampleParing(dataset=get_voc_data_set(config()))

    t = Image.open('/Users/chenlinwei/Desktop/2007_000033.png')
    print(t.mode)
    i = Image.open('/Users/chenlinwei/Desktop/2007_000033.jpg')
    print(i.mode)
    i = i.resize((320, 320))

    i = transforms.ToTensor()(i)
    for _ in range(10):
        i_sp, t = sp(i, t)
        i_sp = transforms.ToPILImage()(i_sp)
        i_sp.show()
    pass


if __name__ == '__main__':
    # _test_sample_pairing()
    _test_cutout()
    pass
