import os
from sigver.datasets.base import IterableDataset
from skimage.io import imread
from skimage import img_as_ubyte


class BrazilianDataset(IterableDataset):
    """ Helper class to load the brazilian PUC-PR dataset
    """
    def __init__(self, path, file_extension='jpg'):
        self.path = path
        self.users = list(range(1, 169))

        self.file_extension = file_extension

    @property
    def genuine_per_user(self):
        return 40

    @property
    def skilled_per_user(self):
        return 10

    @property
    def simple_per_user(self):
        return 10

    @property
    def maxsize(self):
        return 700, 1000

    def get_user_list(self):
        return self.users

    def iter_genuine(self, user):
        """ Iterate over genuine signatures for the given user"""

        genuine_files = ['a{:03d}_{:02d}.{}'.format(user, img, self.file_extension)
                         for img in range(1, 41)]
        for f in genuine_files:
            full_path = os.path.join(self.path, f)
            img = imread(full_path, as_gray=True)
            yield img_as_ubyte(img), f

    def iter_forgery(self, user):
        """ Iterate over skilled forgeries for the given user"""

        if user > 60:
            return  # No forgeries after user 60

        forgery_files = ['a{:03d}_{:02d}.{}'.format(user, img, self.file_extension)
                         for img in range(51, 61)]
        for f in forgery_files:
            full_path = os.path.join(self.path, f)
            img = imread(full_path, as_gray=True)
            yield img_as_ubyte(img), f

    def iter_simple_forgery(self, user):
        """ Iterate over simple forgeries for the given user"""

        if user > 60:
            return  # No forgeries after user 60

        forgery_files = ['a{:03d}_{:02d}.{}'.format(user, img, self.file_extension)
                         for img in range(41, 51)]
        for f in forgery_files:
            full_path = os.path.join(self.path, f)
            img = imread(full_path, as_gray=True)
            yield img_as_ubyte(img), f


class BrazilianDatasetWithoutSimpleForgeries(BrazilianDataset):
    def __init__(self, path, file_extension='jpg'):
        super(BrazilianDatasetWithoutSimpleForgeries, self).__init__(path, file_extension)
        self.users = list(range(1, 61))  # Only use first 60 users

    def iter_simple_forgery(self, user):
        yield from ()  # No simple forgeries

    @property
    def simple_per_user(self):
        return 0
