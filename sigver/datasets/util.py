import numpy as np
import functools
from tqdm import tqdm
from sigver.datasets.base import IterableDataset
from sigver.preprocessing.normalize import preprocess_signature
from typing import Tuple, Callable, Dict, Union


def load_dataset(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict, np.ndarray]:
    """ Loads a dataset that was pre-processed in a numpy format

    Parameters
    ----------
    path : str
        The path to a .npz file, containing attributes "x", "y", "yforg",
        "usermapping" and "filenames"

    Returns
    -------
    x : np.ndarray (N x 1 x H x W) are N grayscale signature images of size H x W
    y : np.ndarray (N) indicating the user that wrote the signature
    yforg : np.ndarray (N) indicating wether the signature is a forgery
    user_mapping: dict, mapping the indexes in "y" with the original user
                 numbers from the dataset
    filenames: np.ndarray(str) (N), the filename associated with the signature image
    -------

    """
    with np.load(path) as data:
        x, y, yforg = data['x'], data['y'], data['yforg']
        user_mapping, filenames = data['user_mapping'], data['filenames']

    return x, y, yforg, user_mapping, filenames


def process_dataset(dataset: IterableDataset,
                    save_path: str,
                    img_size: Tuple[int, int],
                    subset: slice = slice(None)):
    """ Processes a dataset (normalizing the images) and saves the result as a
        numpy npz file (collection of np.ndarrays).

    Parameters
    ----------
    dataset : IterableDataset
        The dataset, that knows where the signature files are located
    save_path : str
        The name of the file to save the numpy arrays
    img_size : tuple (H x W)
        The final size of the images
    subset : slice
        Which users to consider. e.g. slice(None) to consider all users, or slice(first, last)


    Returns
    -------
    None

    """
    preprocess_fn = functools.partial(preprocess_signature,
                                      canvas_size=dataset.maxsize,
                                      img_size=img_size,
                                      input_size=img_size)

    processed = process_dataset_images(dataset, preprocess_fn, img_size, subset)
    x, y, yforg, user_mapping, used_files = processed

    np.savez(save_path,
             x=x,
             y=y,
             yforg=yforg,
             user_mapping=user_mapping,
             filenames=used_files)


def process_dataset_images(dataset: IterableDataset,
                           preprocess_fn: Callable[[np.ndarray], np.ndarray],
                           img_size: Tuple[int, int],
                           subset: slice) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict, np.ndarray]:
    """ Process the signature images from a dataset, returning numpy arrays.

    Parameters
    ----------
    dataset : IterableDataset
        The dataset, that knows where the signature files are located
    preprocess_fn : function (image) -> image
        A function that takes as input a signature image, preprocess-it and return a new image
    img_size : tuple (H x W)
        The final size of the images
    subset : slice
        Which users to consider. Either "None" (to consider all users) or a slice(first, last)

    Returns
    -------
    x : np.ndarray (N x 1 x H x W) are N grayscale signature images of size H x W
    y : np.ndarray (N) indicating the user that wrote the signature
    yforg : np.ndarray (N) indicating wether the signature is a forgery
    usermapping: dict, mapping the indexes in "y" with the original user
                 numbers from the dataset
    filenames: np.ndarray(str) (N), the filename associated with the signature image
    """

    user_mapping = {}

    users = dataset.get_user_list()
    users = users[subset]
    print('Number of users: %d' % len(users))

    # Pre-allocate an array X to hold all signatures. We do so because
    # the alternative of concatenating several arrays in the end takes
    # a lot of memory, which can be problematic when using large image sizes
    H, W = img_size
    max_signatures = len(users) * (dataset.genuine_per_user + dataset.skilled_per_user + dataset.simple_per_user)

    x = np.empty((max_signatures, H, W), dtype=np.uint8)
    y = np.empty(max_signatures, dtype=np.int32)
    yforg = np.empty(max_signatures, dtype=np.int32)
    used_files = []

    print('Allocated x of shape: %s' % (x.shape,))

    # Keep track of the number of processed signatures
    N = 0
    for i, user in enumerate(tqdm(users)):
        # Genuine signatures

        user_gen_data = [(preprocess_fn(img), filename) for (img, filename) in dataset.iter_genuine(user)]
        gen_imgs, gen_filenames = zip(*user_gen_data)
        new_img_count = len(gen_imgs)

        indexes = slice(N, N + new_img_count)
        x[indexes] = gen_imgs
        yforg[indexes] = 0
        y[indexes] = i
        used_files += gen_filenames
        N += new_img_count

        # Skilled forgeries
        user_forg_data = [(preprocess_fn(img), filename) for (img, filename) in dataset.iter_forgery(user)]

        if len(user_forg_data) > 0:
            forg_imgs, forg_filenames = zip(*user_forg_data)
            new_img_count = len(forg_imgs)

            indexes = slice(N, N + new_img_count)
            x[indexes] = forg_imgs
            yforg[indexes] = 1
            y[indexes] = i
            used_files += forg_filenames
            N += new_img_count

        # Simple forgeries
        user_forg_data = [(preprocess_fn(img), filename) for (img, filename) in dataset.iter_simple_forgery(user)]

        if len(user_forg_data) > 0:
            forg_imgs, forg_filenames = zip(*user_forg_data)
            new_img_count = len(forg_imgs)

            indexes = slice(N, N + new_img_count)
            x[indexes] = forg_imgs
            yforg[indexes] = 2  # Simple forgeries
            y[indexes] = i
            used_files += forg_filenames
            N += new_img_count

        user_mapping[i] = user

    if N != max_signatures:
        # Some users had less signatures than expected. shrink the arrays
        x.resize((N, 1, H, W), refcheck=False)
        y.resize(N, refcheck=False)
        yforg.resize(N, refcheck=False)
    else:
        x = np.expand_dims(x, 1)

    used_files = np.array(used_files)

    return x, y, yforg, user_mapping, used_files


def get_subset(data: Tuple[np.ndarray, ...],
               subset: Union[list, range],
               y_idx: int = 1) -> Tuple[np.ndarray, ...]:
    """ Gets a data for a subset of users (the second array in data)

    Parameters
    ----------
    data: Tuple (x, y, ...)
        The dataset
    subset: list
        The list of users to include
    y_idx: int
        The index in data that refers to the users (usually index=1)

    Returns
    -------
    Tuple (x, y , ...)
        The dataset containing only data from users in the subset

    """
    to_include = np.isin(data[y_idx], subset)

    return tuple(d[to_include] for d in data)


def remove_forgeries(data: Tuple[np.ndarray, ...],
                     forg_idx: int = 2) -> Tuple[np.ndarray, ...]:
    """ Remove the forgeries from a dataset

    Parameters
    ----------
    data: Tuple (x, y, yforg)
        The dataset
    forg_idx: int
        The index in data that refers to wheter the signature is a forgery (usually
        we pass (x, y, yforg) so forg_idx=2)

    Returns
    -------
    Tuple (x, y, yforg)
        The dataset with only genuine signatures

    """
    to_include = (data[forg_idx] == False)

    return tuple(d[to_include] for d in data)
