import os
from sigver.datasets.base import IterableDataset
from skimage.io import imread
from skimage import img_as_ubyte


class MCYTDataset(IterableDataset):
    """ Helper class to load the MCYT Grayscale dataset
    """

    def __init__(self, path):
        self.path = path
        self.users = [int(user) for user in sorted(os.listdir(self.path))]

    @property
    def genuine_per_user(self):
        return 15

    @property
    def skilled_per_user(self):
        return 15

    @property
    def simple_per_user(self):
        return 0

    @property
    def maxsize(self):
        return 600, 850

    def get_user_list(self):
        return self.users

    def iter_genuine(self, user):
        """ Iterate over genuine signatures for the given user"""

        user_folder = os.path.join(self.path, '{:04d}'.format(user))
        all_files = sorted(os.listdir(user_folder))
        genuine_files = filter(lambda x: x.lower().startswith('{:04d}v'.format(user)), all_files)
        for f in genuine_files:
            full_path = os.path.join(user_folder, f)
            img = imread(full_path, as_gray=True)
            yield img_as_ubyte(img), f

    def iter_forgery(self, user):
        """ Iterate over skilled forgeries for the given user"""

        user_folder = os.path.join(self.path, '{:04d}'.format(user))
        all_files = sorted(os.listdir(user_folder))
        forgery_files = filter(lambda x: ('{:04d}f'.format(user)) in x.lower(), all_files)
        for f in forgery_files:
            full_path = os.path.join(user_folder, f)
            img = imread(full_path, as_gray=True)
            yield img_as_ubyte(img), f

    def iter_simple_forgery(self, user):
        yield from ()  # No simple forgeries
