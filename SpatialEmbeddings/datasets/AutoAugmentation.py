from PIL import Image, ImageOps, ImageEnhance
import random


class AutoAugmentation:
    def __init__(self):
        self.policy = GOOD_POLICY

    def __call__(self, img, target):
        t = random.choice(self.policy)
        for name, p, level in t:
            transform = ALL_TRANSFORM_DICT.get(name)
            if random.random() < p:
                img, target = transform(img, target, level)
        return img, target

    def _test_call(self, img, target):
        for p in self.policy:
            for name, p, level in p:
                transform = ALL_TRANSFORM_DICT.get(name)
                if transform is None:
                    print(name)
                    raise NotImplementedError
                if random.random() < p:
                    img, target = transform(img, target, level)
        return img, target


def magnitude_to_float(level, min_val, max_val):
    assert 0 <= level < 10
    return float((level / 10) * (max_val - min_val) + min_val)


def magnitude_to_int(level, min_val, max_val):
    assert 0 <= level < 10
    return int((level / 10) * (max_val - min_val) + min_val)


class Posterize:
    def __init__(self, range=(0, 4)):
        self.name = 'Posterize'
        self.range = range

    def __call__(self, img, target, level):
        n = magnitude_to_int(level, *self.range)
        img = ImageOps.posterize(img, 4 - n)
        return img, target


class Solarize:
    def __init__(self, range=(0, 256)):
        self.name = 'Solarize'
        self.range = range

    def __call__(self, img, target, level):
        n = magnitude_to_int(level, *self.range)
        img = ImageOps.solarize(img, 256 - n)
        return img, target


class Sharpness:
    def __init__(self, range=(0.0, 1.8)):
        self.name = 'Sharpness'
        self.range = range

    def __call__(self, img, target, level):
        f = magnitude_to_float(level, *self.range) + 0.1
        img = ImageEnhance.Sharpness(img).enhance(f)
        return img, target


class Equalize:
    def __init__(self):
        self.name = 'Equalize'

    def __call__(self, img, target, level):
        img = ImageOps.equalize(img)
        return img, target


class ShearX:
    def __init__(self, range=(0, 0.3)):
        self.name = 'ShearX'
        self.range = range

    def __call__(self, img, target, level):
        f = magnitude_to_float(level, *self.range)
        if random.random() > 0.5:
            f = -f
        img = img.transform(img.size, Image.AFFINE, (1, f, 0, 0, 1, 0))
        target = target.transform(img.size, Image.AFFINE, (1, f, 0, 0, 1, 0))
        return img, target


class ShearY:
    def __init__(self, range=(0, 0.3)):
        self.name = 'ShearY'
        self.range = range

    def __call__(self, img, target, level):
        f = magnitude_to_float(level, *self.range)
        if random.random() > 0.5:
            f = -f
        img = img.transform(img.size, Image.AFFINE, (1, 0, 0, f, 1, 0))
        target = target.transform(img.size, Image.AFFINE, (1, 0, 0, f, 1, 0))
        return img, target


class TranslateX:
    def __init__(self, range=(0, 150)):
        self.name = 'TranslateX'
        self.range = range

    def __call__(self, img, target, level):
        f = magnitude_to_float(level, *self.range)
        if random.random() > 0.5:
            f = -f
        img = img.transform(img.size, Image.AFFINE, (1, 0, f, 0, 1, 0))
        target = target.transform(img.size, Image.AFFINE, (1, 0, f, 0, 1, 0))
        return img, target


class TranslateY:
    def __init__(self, range=(0, 150)):
        self.name = 'TranslateY'
        self.range = range

    def __call__(self, img, target, level):
        f = magnitude_to_float(level, *self.range)
        if random.random() > 0.5:
            f = -f
        img = img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, f))
        target = target.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, f))
        return img, target


class Rotate:
    def __init__(self, range=(0, 30)):
        self.name = 'Rotate'
        self.range = range

    def __call__(self, img, target, level):
        f = magnitude_to_float(level, *self.range)
        if random.random() > 0.5:
            f = -f
        img = img.rotate(f)
        target = target.rotate(f)
        return img, target


class Invert:
    def __init__(self):
        self.name = 'Invert'

    def __call__(self, img, target, level):
        img = ImageOps.invert(img)
        return img, target


class Contrast:
    def __init__(self, range=(0.0, 1.8)):
        self.name = 'Contrast'
        self.range = range

    def __call__(self, img, target, level):
        f = magnitude_to_float(level, *self.range) + 0.1
        img = ImageEnhance.Contrast(img).enhance(f)
        return img, target


class AutoContrast:
    def __init__(self):
        self.name = 'AutoContrast'

    def __call__(self, img, target, level):
        img = ImageOps.autocontrast(img, cutoff=10)
        return img, target


class Color:
    def __init__(self, range=(0.0, 1.8)):
        self.name = 'Color'
        self.range = range

    def __call__(self, img, target, level):
        f = magnitude_to_float(level, *self.range) + 0.1
        img = ImageEnhance.Color(img).enhance(f)
        return img, target


class Brightness:
    def __init__(self, range=(0.0, 1.8)):
        self.name = 'Brightness'
        self.range = range

    def __call__(self, img, target, level):
        f = magnitude_to_float(level, *self.range) + 0.1
        img = ImageEnhance.Brightness(img).enhance(f)
        return img, target


ALL_TRANSFORM = [Posterize(), Solarize(), Sharpness(), Equalize(),
                 ShearX(), ShearY(), TranslateX(), TranslateY(), Rotate(),
                 Invert(), Contrast(), AutoContrast(), Color(), Brightness()]

ALL_TRANSFORM_DICT = {t.name: t for t in ALL_TRANSFORM}

GOOD_POLICY = [[('Posterize', 0.4, 8), ('Rotate', 0.6, 9)],
               [('Solarize', 0.6, 5), ('AutoContrast', 0.6, 5)],
               [('Equalize', 0.8, 8), ('Equalize', 0.6, 3)],
               [('Posterize', 0.6, 7), ('Posterize', 0.6, 6)],
               [('Equalize', 0.4, 7), ('Solarize', 0.2, 4)],
               [('Equalize', 0.4, 4), ('Rotate', 0.8, 8)],
               [('Solarize', 0.6, 3), ('Equalize', 0.6, 7)],
               [('Posterize', 0.8, 5), ('Equalize', 1.0, 2)],
               [('Rotate', 0.2, 3), ('Solarize', 0.6, 8)],
               [('Equalize', 0.6, 8), ('Posterize', 0.4, 6)],
               [('Rotate', 0.8, 8), ('Color', 0.4, 0)],
               [('Rotate', 0.4, 9), ('Equalize', 0.6, 2)],
               [('Equalize', 0.0, 7), ('Equalize', 0.8, 8)],
               [('Invert', 0.6, 4), ('Equalize', 1.0, 8)],
               [('Color', 0.6, 4), ('Contrast', 1.0, 8)],
               [('Rotate', 0.8, 8), ('Color', 1.0, 2)],
               [('Color', 0.8, 8), ('Solarize', 0.8, 7)],
               [('Sharpness', 0.4, 7), ('Invert', 0.6, 8)],
               [('ShearX', 0.6, 5), ('Equalize', 1.0, 9)],
               [('Color', 0.4, 0), ('Solarize', 0.6, 3)],
               [('Equalize', 0.4, 7), ('Solarize', 0.2, 4)],
               [('Solarize', 0.6, 5), ('AutoContrast', 0.6, 5)],
               [('Invert', 0.6, 4), ('Equalize', 1.0, 8)],
               [('Color', 0.6, 4), ('Contrast', 1.0, 8)],
               [('Equalize', 0.8, 8), ('Equalize', 0.6, 3)]]


def _test_transform():
    img = Image.open('/Users/chenlinwei/Desktop/2007_000033.jpg')
    target = Image.open('/Users/chenlinwei/Desktop/2007_000033.png')
    t = ALL_TRANSFORM[12]
    for level in range(10):
        _ = t(img, target, level)
        _[0].show()
        # _[1].show()


def _test_policy():
    AA = AutoAugmentation()
    img = Image.open('/Users/chenlinwei/Desktop/2007_000033.jpg')
    target = Image.open('/Users/chenlinwei/Desktop/2007_000033.png')
    for _ in range(10):
        _ = AA._test_call(img, target)
        _[0].show()
        _[1].show()


def _test_magnitude():
    for i in range(10):
        f = magnitude_to_float(i, *(0, 100))
        n = magnitude_to_int(i, *(0, 100))
        print(f, n)


if __name__ == '__main__':
    _test_policy()
    pass
