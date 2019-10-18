import os
from skimage.io import imread
from sigver.datasets.base import IterableDataset
from skimage import img_as_ubyte


class GPDSDataset(IterableDataset):
    """ Helper class to load the GPDS-960 Grayscale dataset
    """

    def __init__(self, path, extension='png'):
        self.path = path
        self.users = [int(user) for user in sorted(os.listdir(self.path))]
        self.extension = extension

    @property
    def genuine_per_user(self):
        return 24

    @property
    def skilled_per_user(self):
        return 30

    @property
    def simple_per_user(self):
        return 0

    @property
    def maxsize(self):
        return 952, 1360

    def get_user_list(self):
        return self.users

    def iter_genuine(self, user):
        """ Iterate over genuine signatures for the given user"""

        user_folder = os.path.join(self.path, '{:03d}'.format(user))
        all_files = sorted(os.listdir(user_folder))
        user_genuine_files = filter(lambda x: x[0:2] == 'c-', all_files)
        for f in user_genuine_files:
            full_path = os.path.join(user_folder, f)
            img = imread(full_path, as_gray=True)
            yield img_as_ubyte(img), f

    def iter_forgery(self, user):
        """ Iterate over skilled forgeries for the given user"""

        user_folder = os.path.join(self.path, '{:03d}'.format(user))
        all_files = sorted(os.listdir(user_folder))
        user_forgery_files = filter(lambda x: x[0:2] == 'cf', all_files)
        for f in user_forgery_files:
            full_path = os.path.join(user_folder, f)
            img = imread(full_path, as_gray=True)
            yield img_as_ubyte(img), f

    def get_signature(self, user, img_idx, forgery):
        """ Returns a particular signature (given by user id, img id and
            whether or not it is a forgery
        """

        if forgery:
            prefix = 'cf'
        else:
            prefix = 'c'
        filename = '{}-{:03d}-{:02d}.{}'.format(prefix, user, img_idx,
                                                self.extension)
        full_path = os.path.join(self.path, '{:03d}'.format(user), filename)
        return img_as_ubyte(imread(full_path, as_gray=True))

    def iter_simple_forgery(self, user):
        yield from ()  # No simple forgeries
