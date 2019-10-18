import os
import numpy as np
from functools import partial

from sigver.datasets import gpds
from sigver.datasets.util import process_dataset
from sigver.datasets.toremove import center_crop, center_crop_multiple, random_crop, random_crop_multiple
import math
import warnings

def load_all_data(processed_path, original_path, img_size):
    if not os.path.exists(processed_path):
        ds = gpds.GPDSDataset(original_path)
        x, y, yforg, usermapping, filenames = process_dataset(ds, img_size, subset=None, load_forgeries=True)
        np.savez(processed_path, x=x, y=y, yforg=yforg, usermapping=usermapping, filenames=filenames)
    else:
        data = np.load(processed_path)
        x, y, yforg = data['x'], data['y'], data['yforg']
        usermapping, filenames = data['usermapping'], data['filenames']
    return x, y, yforg, usermapping, filenames


class RFDataIterator:
    """ Data iterator for Meta-learning using a two-class formulation with random forgeries

        data: Dataset (x, y) for all users, where
                 x is a 4D tensor (N x C x H x W),
                 y is a vector (N) - the user that wrote the signature

        subset: list of users that should be included in this iterator
        num_gen: number of genuine signatures from the user available for training
        num_rf: number of random forgeries available for training
        num_test: number of signatures (genuine/rf) for test
        rng: Random number generator state
    """

    def __init__(self, data, subset, num_gen, num_rf, num_test, input_shape, batch_size, test=False, rng=np.random.RandomState()):
        self.batch_size = batch_size
        self.num_gen = num_gen
        self.num_test = num_test
        self.num_rf = num_rf
        self.rng = rng
        self.input_shape = input_shape
        self.test = test
        if test:
            self.crop_fn = partial(center_crop, shape=input_shape)
            self.crop_multiple_fn = partial(center_crop_multiple, shape=input_shape)
        else:
            self.crop_fn = partial(random_crop, shape=input_shape)
            self.crop_multiple_fn = partial(random_crop_multiple, shape=input_shape)

        x, y, yforg = data
        to_include = np.isin(y, subset)
        to_include = np.logical_and(to_include, (yforg == False))

        self.x = x[to_include]
        self.y = y[to_include]
        self.users = np.unique(self.y)

    def __iter__(self):
        shuffled_users = self.users.copy()
        self.rng.shuffle(shuffled_users)

        for i in range(0, len(shuffled_users), self.batch_size):
            batch_train_x = []
            batch_train_y = []
            batch_test_x = []
            batch_test_y = []

            for user in shuffled_users[i:i+self.batch_size]:
                # Genuine signatures: class 1
                user_indexes = np.flatnonzero(self.y == user)
                self.rng.shuffle(user_indexes)
                user_signatures = self.x[user_indexes]
                train_x = [self.crop_multiple_fn(user_signatures[0:self.num_gen])]
                test_x = [self.crop_multiple_fn(user_signatures[self.num_gen:self.num_gen + self.num_test])]
                train_y = [np.ones(self.num_gen, dtype=int)]
                test_y = [np.ones(self.num_test, dtype=int)]

                # Random forgeries: class 0
                other_users = list(set(self.users).difference([user]))
                chosen_users = self.rng.choice(other_users, self.num_rf + self.num_test, replace=False)
                train_users = chosen_users[0:self.num_rf]
                test_users = chosen_users[self.num_rf:]

                for other in train_users:
                    other_indexes = np.flatnonzero(self.y == other)
                    other_idx = self.rng.choice(other_indexes, 1)[0]
                    train_x.append([self.crop_fn(self.x[other_idx])])
                    train_y.append([0])

                for other in test_users:
                    other_indexes = np.flatnonzero(self.y == other)
                    other_idx = self.rng.choice(other_indexes, 1)[0]
                    test_x.append([self.crop_fn(self.x[other_idx])])
                    test_y.append([0])

                batch_train_x.append(np.concatenate(train_x).astype(np.float32))
                batch_train_y.append(np.concatenate(train_y)[:, np.newaxis])
                batch_test_x.append(np.concatenate(test_x).astype(np.float32))
                batch_test_y.append(np.concatenate(test_y)[:, np.newaxis])

            batch_train_x = np.stack(batch_train_x)
            batch_train_y = np.stack(batch_train_y)
            batch_test_x = np.stack(batch_test_x)
            batch_test_y = np.stack(batch_test_y)

            yield (batch_train_x, batch_train_y), (batch_test_x, batch_test_y)


class SKDataIterator:
    """ Data iterator for Meta-learning using a two-class formulation with random forgeries, but testing with Skilled Forgeries

        data: Dataset (x, y) for all users, where
                 x is a 4D tensor (N x C x H x W),
                 y is a vector (N) - the user that wrote the signature

        subset: list of users that should be included in this iterator
        num_gen: number of genuine signatures from the user available for training
        num_rf: number of random forgeries available for training
        num_test: number of signatures (genuine/skilled) for test
        rng: Random number generator state
    """

    def __init__(self, data, subset, num_gen, num_rf, num_test, input_shape, batch_size, test=False, rng=np.random.RandomState()):
        self.batch_size = batch_size
        self.num_gen = num_gen
        self.num_test = num_test
        self.num_rf = num_rf
        self.rng = rng
        self.input_shape = input_shape
        self.test = test
        if test:
            self.crop_fn = partial(center_crop, shape=input_shape)
            self.crop_multiple_fn = partial(center_crop_multiple, shape=input_shape)
        else:
            self.crop_fn = partial(random_crop, shape=input_shape)
            self.crop_multiple_fn = partial(random_crop_multiple, shape=input_shape)

        x, y, yforg = data
        to_include = np.isin(y, subset)

        self.x = x[to_include]
        self.y = y[to_include]
        self.yforg = yforg[to_include]
        self.users = np.unique(self.y)

    def __len__(self):
        return int(math.ceil(len(self.users) / self.batch_size))

    def __iter__(self):
        shuffled_users = self.users.copy()
        self.rng.shuffle(shuffled_users)

        for i in range(0, len(shuffled_users), self.batch_size):
            batch_train_x = []
            batch_train_y = []
            batch_test_x = []
            batch_test_y = []

            for user in shuffled_users[i:i+self.batch_size]:
                # Genuine signatures: class 1
                genuine_indexes = np.flatnonzero((self.y == user) & (self.yforg == False))
                self.rng.shuffle(genuine_indexes)
                user_signatures = self.x[genuine_indexes]
                train_x = [self.crop_multiple_fn(user_signatures[0:self.num_gen])]
                test_x = [self.crop_multiple_fn(user_signatures[self.num_gen:self.num_gen + self.num_test])]
                train_y = [np.ones(self.num_gen, dtype=int)]
                test_y = [np.ones(self.num_test, dtype=int)]

                # Random forgeries: class 0
                other_users = list(set(self.users).difference([user]))
                train_users = self.rng.choice(other_users, self.num_rf, replace=False)

                for other in train_users:
                    other_indexes = np.flatnonzero((self.y == other) & (self.yforg == False))
                    other_idx = self.rng.choice(other_indexes, 1)[0]
                    train_x.append([self.crop_fn(self.x[other_idx])])
                    train_y.append([0])

                skilled_indexes = np.flatnonzero((self.y == user) & (self.yforg == True))
                self.rng.shuffle(skilled_indexes)
                skilled_forgeries = self.x[skilled_indexes]

                test_x.append(self.crop_multiple_fn(skilled_forgeries[0:self.num_test]))
                test_y.append(np.zeros(self.num_test, dtype=int))

                batch_train_x.append(np.concatenate(train_x).astype(np.float32))
                batch_train_y.append(np.concatenate(train_y)[:, np.newaxis])
                batch_test_x.append(np.concatenate(test_x).astype(np.float32))
                batch_test_y.append(np.concatenate(test_y)[:, np.newaxis])

            batch_train_x = np.stack(batch_train_x)
            batch_train_y = np.stack(batch_train_y)
            batch_test_x = np.stack(batch_test_x)
            batch_test_y = np.stack(batch_test_y)

            yield (batch_train_x, batch_train_y), (batch_test_x, batch_test_y)


class MAMLDataIterator:
    """ Data iterator for Meta-learning

        data: Dataset (x, y) for all users, where
                 x is a 4D tensor (N x C x H x W),
                 y is a vector (N) - the user that wrote the signature

        subset: list of users that should be included in this iterator
        num_gen: number of genuine signatures from the user available for training
        num_rf: number of random forgeries available for training
        num_test: number of signatures (genuine/rf) for test
        rng: Random number generator state
    """

    def __init__(self, data, subset, num_gen_train, num_rf_train,
                 num_gen_test, num_rf_test, num_sk_test, input_shape, batch_size, test=False, rng=np.random.RandomState()):
        self.batch_size = batch_size
        self.num_gen = num_gen_train
        self.num_rf_train = num_rf_train

        self.num_sk_test = num_sk_test
        self.num_rf_test = num_rf_test
        self.num_gen_test = num_gen_test

        self.rng = rng
        self.input_shape = input_shape
        self.test = test
        if test:
            self.crop_fn = partial(center_crop, shape=input_shape)
            self.crop_multiple_fn = partial(center_crop_multiple, shape=input_shape)
        else:
            self.crop_fn = partial(random_crop, shape=input_shape)
            self.crop_multiple_fn = partial(random_crop_multiple, shape=input_shape)

        x, y, yforg = data
        to_include = np.isin(y, subset)

        self.x = x[to_include]
        self.y = y[to_include]
        self.yforg = yforg[to_include]
        self.users = np.unique(self.y)

    def __iter__(self):
        shuffled_users = self.users.copy()
        self.rng.shuffle(shuffled_users)

        for i in range(0, len(shuffled_users), self.batch_size):
            batch_train_x = []
            batch_train_y = []
            batch_test_x = []
            batch_test_y = []
            batch_test_yforg = []

            for user in shuffled_users[i:i+self.batch_size]:
                # Genuine signatures: class 1
                genuine_indexes = np.flatnonzero((self.y == user) & (self.yforg == False))
                self.rng.shuffle(genuine_indexes)
                user_signatures = self.x[genuine_indexes]
                train_x = [self.crop_multiple_fn(user_signatures[0:self.num_gen])]
                test_x = [self.crop_multiple_fn(user_signatures[self.num_gen:self.num_gen + self.num_gen_test])]
                train_y = [np.ones(self.num_gen, dtype=int)]
                test_y = [np.ones(self.num_gen_test, dtype=int)]
                test_yforg = [np.zeros(self.num_gen_test, dtype=int)]

                # Random forgeries: class 0
                other_users = list(set(self.users).difference([user]))
                chosen_users = self.rng.choice(other_users, self.num_rf_train + self.num_rf_test, replace=False)
                train_users = chosen_users[0:self.num_rf_train]
                test_users = chosen_users[self.num_rf_train:]

                for other in train_users:
                    other_indexes = np.flatnonzero(self.y == other)
                    other_idx = self.rng.choice(other_indexes, 1)[0]
                    train_x.append([self.crop_fn(self.x[other_idx])])
                    train_y.append([0])

                for other in test_users:
                    other_indexes = np.flatnonzero(self.y == other)
                    other_idx = self.rng.choice(other_indexes, 1)[0]
                    test_x.append([self.crop_fn(self.x[other_idx])])
                    test_y.append([0])
                    test_yforg.append([0])

                if self.num_sk_test > 0:
                    skilled_indexes = np.flatnonzero((self.y == user) & (self.yforg == True))
                    self.rng.shuffle(skilled_indexes)
                    skilled_forgeries = self.x[skilled_indexes]

                    test_x.append(self.crop_multiple_fn(skilled_forgeries[0:self.num_sk_test]))
                    test_y.append(np.zeros(self.num_sk_test, dtype=int))
                    test_yforg.append(np.ones(self.num_sk_test, dtype=int))

                batch_train_x.append(np.concatenate(train_x).astype(np.float32))
                batch_train_y.append(np.concatenate(train_y)[:, np.newaxis])
                batch_test_x.append(np.concatenate(test_x).astype(np.float32))
                batch_test_y.append(np.concatenate(test_y)[:, np.newaxis])
                batch_test_yforg.append(np.concatenate(test_yforg)[:, np.newaxis])

            batch_train_x = np.stack(batch_train_x)
            batch_train_y = np.stack(batch_train_y)
            batch_test_x = np.stack(batch_test_x)
            batch_test_y = np.stack(batch_test_y)
            batch_test_yforg = np.stack(batch_test_yforg)

            yield (batch_train_x, batch_train_y), (batch_test_x, batch_test_y, batch_test_yforg)


def generate_random_indices(n, train_fraction):
    indices = np.arange(n)
    np.random.shuffle(indices)

    first_test_index = int(train_fraction * n)

    train_indices = indices[0:first_test_index]
    test_indices = indices[first_test_index:]
    return train_indices, test_indices


class DataIterator:
    def __init__(self, data, input_shape, batch_size, test=False, rng=np.random.RandomState()):
        self.batch_size = batch_size
        self.rng = rng
        self.input_shape = input_shape
        self.test = test
        if test:
            self.crop_multiple_fn = partial(center_crop_multiple, shape=input_shape)
        else:
            self.crop_multiple_fn = partial(random_crop_multiple, shape=input_shape)

        self.x, self.y, self.yforg = data

    def __iter__(self):
        random_idx = list(range(len(self.x)))
        self.rng.shuffle(random_idx)
        for i in range(0, len(self.x), self.batch_size):
            idx = random_idx[i: i + self.batch_size]
            batch_x = self.x[idx]
            batch_x = self.crop_multiple_fn(batch_x)
            batch_y = self.y[idx]
            batch_yforg = self.yforg[idx]

            yield batch_x.astype(np.float32), batch_y, batch_yforg


from torch.utils.data import Dataset


class MAMLDataSet(Dataset):
    """ Data iterator for Meta-learning

        data: Dataset (x, y) for all users, where
                 x is a 4D tensor (N x C x H x W),
                 y is a vector (N) - the user that wrote the signature

        subset: list of users that should be included in this iterator
        num_gen: number of genuine signatures from the user available for training
        num_rf: number of random forgeries available for training
        num_test: number of signatures (genuine/rf) for test
        rng: Random number generator state
    """

    def __init__(self, data, subset, num_gen_train, num_rf_train,
                 num_gen_test, num_rf_test, num_sk_test, input_shape, sk_subset=None,
                 test=False, rng=np.random.RandomState()):
        self.num_gen = num_gen_train
        self.num_rf_train = num_rf_train

        self.num_sk_test = num_sk_test
        self.num_rf_test = num_rf_test
        self.num_gen_test = num_gen_test

        self.rng = rng
        self.input_shape = input_shape
        self.test = test
        if test:
            self.crop_fn = partial(center_crop, shape=input_shape)
            self.crop_multiple_fn = partial(center_crop_multiple, shape=input_shape)
        else:
            self.crop_fn = partial(random_crop, shape=input_shape)
            self.crop_multiple_fn = partial(random_crop_multiple, shape=input_shape)

        if sk_subset is None:
            self.sk_subset = set(subset)
        else:
            self.sk_subset = sk_subset

        x, y, yforg = data
        to_include = np.isin(y, subset)

        self.mapping = list(subset)

        self.x = x[to_include]
        self.y = y[to_include]
        self.yforg = yforg[to_include]
        self.users = np.unique(self.y)

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, item):
        user = self.mapping[item]
        # Genuine signatures: class 1
        genuine_indexes = np.flatnonzero((self.y == user) & (self.yforg == False))
        self.rng.shuffle(genuine_indexes)
        user_signatures = self.x[genuine_indexes]
        train_x = [self.crop_multiple_fn(user_signatures[0:self.num_gen])]
        test_x = [self.crop_multiple_fn(user_signatures[self.num_gen:self.num_gen + self.num_gen_test])]
        train_y = [np.ones(self.num_gen, dtype=int)]
        test_y = [np.ones(self.num_gen_test, dtype=int)]
        test_yforg = [np.zeros(self.num_gen_test, dtype=int)]

        # Random forgeries: class 0
        other_users = list(set(self.users).difference([user]))
        if self.num_rf_train + self.num_rf_test > len(other_users):
            replace = True
            warnings.warn('Warning: overlap on Random Forgeries - do not user random forgery metrics')
        else:
            replace = False
        chosen_users = self.rng.choice(other_users, self.num_rf_train + self.num_rf_test, replace=replace)
        train_users = chosen_users[0:self.num_rf_train]
        test_users = chosen_users[self.num_rf_train:]

        for other in train_users:
            other_indexes = np.flatnonzero((self.y == other) & (self.yforg == False))
            other_idx = self.rng.choice(other_indexes, 1)[0]
            train_x.append([self.crop_fn(self.x[other_idx])])
            train_y.append([0])

        for other in test_users:
            other_indexes = np.flatnonzero((self.y == other) & (self.yforg == False))
            other_idx = self.rng.choice(other_indexes, 1)[0]
            test_x.append([self.crop_fn(self.x[other_idx])])
            test_y.append([0])
            test_yforg.append([0])

        # If user has skilled forgeries, and skilled forgeries are being used:
        if self.num_sk_test > 0 and user in self.sk_subset:
            skilled_indexes = np.flatnonzero((self.y == user) & (self.yforg == True))
            self.rng.shuffle(skilled_indexes)
            skilled_forgeries = self.x[skilled_indexes]

            test_x.append(self.crop_multiple_fn(skilled_forgeries[0:self.num_sk_test]))
            test_y.append(np.zeros(self.num_sk_test, dtype=int))
            test_yforg.append(np.ones(self.num_sk_test, dtype=int))

        train_x = np.concatenate(train_x).astype(np.float32) / 255
        train_y = np.concatenate(train_y)[:, np.newaxis]
        test_x = np.concatenate(test_x).astype(np.float32) / 255
        test_y = np.concatenate(test_y)[:, np.newaxis]
        test_yforg = np.concatenate(test_yforg)[:, np.newaxis]

        return train_x, train_y, test_x, test_y, test_yforg

    @staticmethod
    def collate_fn(batch):
        """ Consolidates a batch of dataset items.

        Parameters
        ----------
        batch: list of N dataset items (train_x, train_y, test_x, test_y, test_yforg)

        Returns
        -------
        train_x, train_y, test_x, test_y, test_yforg
            Where each is a list of size N

        """
        assert len(batch[0]) == 5  # train_x, train_y, test_x, test_y, test_yforg

        all_data = []
        for i in range(5):
            all_data.append([b[i] for b in batch])

        return all_data
