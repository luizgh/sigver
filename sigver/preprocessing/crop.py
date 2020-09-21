import numpy as np


def center_crop(img, shape):
    img_shape = img.shape[1:]
    start_y = (img_shape[0] - shape[0]) // 2
    start_x = (img_shape[1] - shape[1]) // 2
    cropped = img[:, start_y: start_y + shape[0], start_x:start_x + shape[1]]
    return cropped


def center_crop_multiple(imgs, shape):
    img_shape = imgs.shape[2:]
    start_y = (img_shape[0] - shape[0]) // 2
    start_x = (img_shape[1] - shape[1]) // 2
    cropped = imgs[:, :, start_y: start_y + shape[0], start_x:start_x + shape[1]]
    return cropped


def random_crop(img, shape, rng=np.random.RandomState()):
    img_shape = img.shape[1:]
    start_y = rng.randint(0, img_shape[0] - shape[0])
    start_x = rng.randint(0, img_shape[1] - shape[1])
    cropped = img[:, start_y:start_y + shape[0], start_x:start_x + shape[1]]

    return cropped


def random_crop_multiple(imgs, shape, rng=np.random.RandomState()):
    result = np.empty((imgs.shape[0], imgs.shape[1], shape[0], shape[1]))
    img_shape = imgs.shape[2:]

    for i, img in enumerate(imgs):
        start_y = rng.randint(0, img_shape[0] - shape[0])
        start_x = rng.randint(0, img_shape[1] - shape[1])
        cropped = img[:, start_y:start_y + shape[0], start_x:start_x + shape[1]]
        result[i] = cropped

    return result
