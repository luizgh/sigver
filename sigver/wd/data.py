import numpy as np
from typing import Tuple


def split_train_test(exp_set: Tuple[np.ndarray, np.ndarray, np.ndarray],
                     num_gen_train: int,
                     num_gen_test: int,
                     rng: np.random.RandomState) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray],
                                                          Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """ Splits a set into training and testing. Both sets contains the same users. The
        training set contains only genuine signatures, while the testing set contains
        genuine signatures and forgeries. Note that the number of genuine signatures for
        training plus the number of genuine signatures for test must be smaller or equal to
        the total number of genuine signatures (to ensure no overlap)

    Parameters
    ----------
    exp_set: tuple of np.ndarray (x, y, yforg)
        The dataset
    num_gen_train: int
        The number of genuine signatures to be used for training
    num_gen_test: int
        The number of genuine signatures to be used for testing
    rng: np.random.RandomState
        The random number generator (for reproducibility)

    Returns
    -------
    tuple of np.ndarray (x, y, yforg)
        The training set

    tuple of np.ndarray (x, y, yforg)
        The testing set
    """
    x, y, yforg = exp_set
    users = np.unique(y)

    train_idx = []
    test_idx = []

    for user in users:
        user_genuines = np.flatnonzero((y == user) & (yforg == False))
        rng.shuffle(user_genuines)
        user_train_idx = user_genuines[0:num_gen_train]
        user_test_idx = user_genuines[-num_gen_test:]

        # Sanity check to ensure training samples are not used in test:
        assert len(set(user_train_idx).intersection(user_test_idx)) == 0

        train_idx += user_train_idx.tolist()
        test_idx += user_test_idx.tolist()

        user_forgeries = np.flatnonzero((y == user) & (yforg == True))
        test_idx += user_forgeries.tolist()

    exp_train = x[train_idx], y[train_idx], yforg[train_idx]
    exp_test = x[test_idx], y[test_idx], yforg[test_idx]

    return exp_train, exp_test


def create_training_set_for_user(user: int,
                                 exp_train: Tuple[np.ndarray, np.ndarray, np.ndarray],
                                 num_forg_from_exp: int,
                                 other_negatives: np.ndarray,
                                 rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray]:
    """ Creates a training set for training a WD classifier for a user

    Parameters
    ----------
    user: int
        The user for which a dataset will be created
    exp_train: tuple of np.ndarray (x, y, yforg)
        The training set split of the exploitation dataset
    num_forg_from_exp: int
        The number of random forgeries from each user in the exploitation set
        (other than "user") that will be used as negatives
    other_negatives: np.ndarray
        A collection of other negative samples (e.g. from a development set)
    rng: np.random.RandomState
        The random number generator (for reproducibility)

    Returns
    -------
    np.ndarray (N), np.ndarray (N)
        The dataset for the user (x, y), where N is the number of signatures
        (genuine + random forgeries)
    """
    exp_x, exp_y, exp_yforg = exp_train

    positive_samples = exp_x[(exp_y == user) & (exp_yforg == 0)]
    if num_forg_from_exp > 0:
        users = np.unique(exp_y)
        other_users = list(set(users).difference({user}))
        negative_samples_from_exp = []
        for other_user in other_users:
            idx = np.flatnonzero((exp_y == other_user) & (exp_yforg == False))
            chosen_idx = rng.choice(idx, num_forg_from_exp, replace=False)
            negative_samples_from_exp.append(exp_x[chosen_idx])
        negative_samples_from_exp = np.concatenate(negative_samples_from_exp)
    else:
        negative_samples_from_exp = []

    if len(other_negatives) > 0 and len(negative_samples_from_exp) > 0:
        negative_samples = np.concatenate((negative_samples_from_exp, other_negatives))
    elif len(other_negatives) > 0:
        negative_samples = other_negatives
    elif len(negative_samples_from_exp) > 0:
        negative_samples = negative_samples_from_exp
    else:
        raise ValueError('Either random forgeries from exploitation or from development sets must be used')

    train_x = np.concatenate((positive_samples, negative_samples))
    train_y = np.concatenate((np.full(len(positive_samples), 1),
                              np.full(len(negative_samples), -1)))

    return train_x, train_y


def get_random_forgeries_from_dev(dev_set: Tuple[np.ndarray, np.ndarray, np.ndarray],
                                  num_forg_from_dev: int,
                                  rng: np.random.RandomState):
    """ Obtain a set of random forgeries form a development set (to be used
        as negative samples)

    Parameters
    ----------
    dev_set: tuple of np.ndarray (x, y, yforg)
        The development dataset
    num_forg_from_dev: int
        The number of random forgeries (signatures) from each user in the development
        set to be considered
    rng: np.random.RandomState
        The random number generator (for reproducibility)

    Returns
    -------
    np.ndarray (N x M)
        The N negative samples (M is the dimensionality of the feature set)

    """
    x, y, yforg = dev_set
    users = np.unique(y)

    random_forgeries = []
    for user in users:
        idx = np.flatnonzero((y == user) & (yforg == False))
        chosen_idx = rng.choice(idx, num_forg_from_dev, replace=False)
        random_forgeries.append(x[chosen_idx])

    return np.concatenate(random_forgeries)