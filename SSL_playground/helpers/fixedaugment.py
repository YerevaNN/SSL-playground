# Code taken from https://github.com/DeepVoltaire/AutoAugment

from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import random

class CIFAR10Policy(object):
    """ Randomly choose one of the best 25 Sub-policies on CIFAR10.

        Example:
        >>> policy = CIFAR10Policy()
        >>> transformed = policy(image)

        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     CIFAR10Policy(),
        >>>     transforms.ToTensor()])
    """
    def __init__(self, fillcolor=(128, 128, 128)):
        self.operations = [
            Operation("shearX", 8, fillcolor),
            Operation("shearY", 8, fillcolor),
            Operation("translateX", 9, fillcolor),
            Operation("translateY", 9, fillcolor),
            Operation("rotate", 2, fillcolor),
            Operation("color", 3, fillcolor),
            Operation("posterize", 7, fillcolor),
            Operation("solarize", 8, fillcolor),
            Operation("contrast", 7, fillcolor),
            Operation("sharpness", 6, fillcolor),
            Operation("brightness", 3, fillcolor),
            Operation("autocontrast", 2, fillcolor),
            Operation("equalize", 8, fillcolor),
            Operation("invert", 3, fillcolor)
        ]

    def __call__(self, img):
        operation_idx = random.randint(0, len(self.operations) - 1)
        return self.operations[operation_idx](img)

    def __repr__(self):
        return "FixedAugment CIFAR10 Policy"

class Operation(object):
    def __init__(self, operation, magnitude_idx, fillcolor=(128, 128, 128)):

        ranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),
            "translateY": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(np.int),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [1] * 10,
            "equalize": [1] * 10,
            "invert": [1] * 10
        }
        # from https://stackoverflow.com/questions/5252170/specify-image-filling-color-when-rotating-in-python-with-pil-and-setting-expand
        def rotate_with_fill(img, magnitude):
            rot = img.convert("RGBA").rotate(magnitude)
            return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(img.mode)

        self.func = {
            "shearX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, magnitude, 0, 0, 1, 0), Image.BICUBIC, fillcolor=fillcolor),
            "shearY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, magnitude, 1, 0), Image.BICUBIC, fillcolor=fillcolor),
            "translateX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, magnitude * img.size[0], 0, 1, 0),
                fillcolor=fillcolor),
            "translateY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * img.size[1]), fillcolor=fillcolor),
            "rotate": lambda img, magnitude: rotate_with_fill(img, magnitude),
            # "rotate": lambda img, magnitude: img.rotate(magnitude * random.choice([-1, 1])),
            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(1 + magnitude),
            "posterize": lambda img, magnitude: ImageOps.posterize(img, magnitude),
            "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),
            "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(1 + magnitude),
            "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(1 + magnitude),
            "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(1 + magnitude),
            "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),
            "equalize": lambda img, magnitude: ImageOps.equalize(img),
            "invert": lambda img, magnitude: ImageOps.invert(img)
        }

        self.operation = self.func[operation]
        if operation in ("shearX", "shearY", "translateX", "translateY", "brightness", "color", "contrast", "sharpness"):
            self.magnitude = ranges[operation][magnitude_idx] * random.choice([-1, 1])
        else:
            self.magnitude = ranges[operation][magnitude_idx]

    def __call__(self, img):
        augmentation_vector = np.zeros(14)
        img = self.operation(img, self.magnitude)
        augmentation_vector[list(self.func.values()).index(self.operation)] = self.magnitude
        return img, augmentation_vector