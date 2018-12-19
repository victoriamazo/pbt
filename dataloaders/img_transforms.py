import numbers
import random
from PIL import Image, ImageOps
import PIL
import numpy as np
import cv2
try:
    import accimage
except ImportError:
    accimage = None

import torch


class Compose(object):
    """
        Composes several transforms together.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args


class ToTensor(object):
    """Convert a batch of ``PIL.Images`` or ``numpy.ndarrays`` to tensor.

    Converts a batch of PIL.Images or numpy.ndarrays (batch_size x H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (batch_size x C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, pic):
        """
        Args:
            batch of pics (PIL.Images or numpy.ndarrays): Images of shape (batch_size, H, W(, C))
            to be converted to tensor.
        Returns:
            Tensor: Converted batch of images of shape (batch_size, C, H, W).
        """
        if isinstance(pic, np.ndarray):
            # handle numpy array
            if len(pic.shape) == 3:
                pic = pic.reshape(pic.shape[0], pic.shape[1], pic.shape[2], -1)

                img = torch.from_numpy(pic.transpose((0, 3, 1, 2)))
            # backward compatibility
            return img.float().div(255)

        if accimage is not None and isinstance(pic, accimage.Image):
            nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
            pic.copyto(nppic)
            return torch.from_numpy(nppic)

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        else:
            return img


class Resize(object):
    """
        Resizes the given PIL.Image to the given size.
    size can be a tuple (target_height, target_width) or an integer,
    in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img1, img2=None, *args):
        assert img2 is None or img1.size == img2.size

        w, h = img1.size
        resize_h, resize_w = self.size
        if w == resize_w and h == resize_h:
            return (img1, img2, args)

        results = [img1.resize(self.size, PIL.Image.ANTIALIAS)]
        if img2 is not None:
            results.append(img2.resize(self.size, PIL.Image.ANTIALIAS))

        results.extend(args)
        return results


class RandCrop(object):
    """
        Crops the given PIL.Image at a random location to have a region of
    crop_max in the range [crop_min_h, 1] and [crop_min_w, 1]
    (crop_min_h and crop_min_w should be in the range (0,1]).
    crop_max can be a tuple (crop_min_h, cropWmax) or an integer, in which case
    the target will be in the range [crop_min_h, 1] and [crop_min_h, 1]
    """

    def __init__(self, crop_max):
        if isinstance(crop_max, numbers.Number):
            self.crop_max = (int(crop_max), int(crop_max))
        else:
            self.crop_max = crop_max

    def __call__(self, img1, img2=None, *args):
        assert img2 is None or img1.size == img2.size

        crop_min_h, crop_min_w = self.crop_max
        assert crop_min_h > 0 and crop_min_w > 0 and crop_min_h <= 1.0 and crop_min_w <= 1.0
        if crop_min_h == 1.0 and crop_min_w == 1.0:
            return (img1, img2, args)

        w, h = img1.size
        rand_w = random.randint(int(crop_min_w*w), w)
        rand_h = random.randint(int(crop_min_h*h), h)
        x1 = random.randint(0, w - rand_w)
        y1 = random.randint(0, h - rand_h)
        results = [img1.crop((x1, y1, x1 + rand_w, y1 + rand_h))]
        if img2 is not None:
            results.append(img2.crop((x1, y1, x1 + rand_w, y1 + rand_h)))
        results.extend(args)
        return results


class RandHFlip(object):
    """
        Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """

    def __init__(self, hflip):
        self.hflip = hflip


    def __call__(self, img1, img2=None, *args):
        if self.hflip and random.random() < 0.5:
            if img2 is not None:
                results = [img1.transpose(Image.FLIP_LEFT_RIGHT), img2.transpose(Image.FLIP_LEFT_RIGHT)]
            else:
                results = img1.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            if img2 is not None:
                results = [img1, img2]
            else:
                results = img1
        return results


class Normalize(object):
    """
        Normalizes a PIL image or np.aray with a given min and std.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        assert std > 0, 'std for image normalization is 0'

    def __call__(self, img1, img2=None, *args):
        img1 = np.array(img1).astype(np.float32)
        img1 = (img1 - self.mean)/self.std
        if img2 is not None:
            img2 = np.array(img2).astype(np.float32)
            img2 = (img2 - self.mean) / self.std
            results = [img1, img2]
        else:
            results = img1
        return results


class Pad(object):
    """
        Pads the given PIL.Image on all sides with the given "pad" value
    """

    def __init__(self, padding, fill=0):
        assert isinstance(padding, numbers.Number)
        assert isinstance(fill, numbers.Number) or isinstance(fill, str) or \
               isinstance(fill, tuple)
        self.padding = padding
        self.fill = fill

    def __call__(self, img1, img2=None, *args):
        if img2 is not None:
            img2 = ImageOps.expand(img2, border=self.padding, fill=255)
        if self.fill == -1:
            img1 = np.asarray(img1)
            img1 = cv2.copyMakeBorder(img1, self.padding, self.padding,
                                       self.padding, self.padding,
                                       cv2.BORDER_REFLECT_101)
            img1 = Image.fromarray(img1)
            return (img1, img2, args)
        else:
            return ImageOps.expand(img1, border=self.padding, fill=self.fill), img2


