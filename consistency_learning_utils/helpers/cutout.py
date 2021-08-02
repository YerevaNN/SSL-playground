import torch
import numpy as np
import random

class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        probability (float): Probability of cutting.
        area_ratio_range (int): Range to randomly select the area ratio of the cut.
        aspect_ratio_range (int): Range to randomly select the x/y ratio of the cut.
    """
    def __init__(self, probability, area_ratio_range, aspect_ratio_range):
        self.probability = probability
        self.area_ratio_range = area_ratio_range
        self.aspect_ratio_range = aspect_ratio_range

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Augmented image.
        """
        rand_number = random.random()
        if rand_number > self.probability:
            return img
        
        h = img.size(1)
        w = img.size(2)

        while True:
            area_ratio = random.uniform(self.area_ratio_range[0], self.area_ratio_range[1])
            aspect_ratio = random.uniform(self.aspect_ratio_range[0], self.aspect_ratio_range[1])

            cut_area = area_ratio * h * w

            ylength = int((cut_area * aspect_ratio) ** 0.5)
            xlength = int((cut_area / aspect_ratio) ** 0.5)

            y = np.random.randint(h)
            x = np.random.randint(w)

            if x + xlength < w and y + ylength < h:
                
                mask = np.ones((h, w), np.float32)
                y1 = np.clip(y, 0, h)
                y2 = np.clip(y + ylength, 0, h)
                x1 = np.clip(x, 0, w)
                x2 = np.clip(x + xlength, 0, w)

                mask[y1: y2, x1: x2] = 0.

                mask = torch.from_numpy(mask)
                mask = mask.expand_as(img)
                img = img * mask

                return img
