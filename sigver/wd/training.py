import numpy as np
from tqdm import tqdm
from typing import Tuple, Optional, Dict
import sklearn
import sklearn.svm
import sklearn.pipeline as pipeline
import sklearn.preprocessing as preprocessing

import sigver.wd.metrics as metrics
import sigver.wd.data as data


def train_wdclassifier_user(training_set: Tuple[np.ndarray, np.ndarray],
                            svmType: str,
                            C: float,
                            gamma: Optional[float]) -> sklearn.svm.SVC:
    """ Trains an SVM classifier for a user

    Parameters
    ----------
    training_set: Tuple (x, y)
        The training set (features and labels). y should have labels -1 and 1
    svmType: string ('linear' or 'rbf')
        The SVM type
    C: float
        Regularization for the SVM optimization
    gamma: float
        Hyperparameter for the RBF kernel

    Returns
    -------
    sklearn.svm.SVC:
        The learned classifier

    """

    assert svmType in ['linear', 'rbf']

    train_x = training_set[0]
    train_y = training_set[1]

    # Adjust for the skew between positive and negative classes
    n_genuine = len([x for x in train_y if x == 1])
    n_forg = len([x for x in train_y if x == -1])
    skew = n_forg / float(n_genuine)

    # Train the model
    if svmType == 'rbf':
        model = sklearn.svm.SVC(C=C, gamma=gamma, class_weight={1: skew})
    else:
        model = sklearn.svm.SVC(kernel='linear', C=C, class_weight={1: skew})

    model_with_scaler = pipeline.Pipeline([('scaler', preprocessing.StandardScaler(with_mean=False)),
                                           ('classifier', model)])

    model_with_scaler.fit(train_x, train_y)

    return model_with_scaler


def test_user(model: sklearn.svm.SVC,
              genuine_signatures: np.ndarray,
              random_forgeries: np.ndarray,
              skilled_forgeries: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Test the WD classifier of an user

    Parameters
    ----------
    model: sklearn.svm.SVC
        The learned classifier
    genuine_signatures: np.ndarray
        Genuine signatures for test
    random_forgeries: np.ndarray
        Random forgeries for test (signatures from other users)
    skilled_forgeries: np.ndarray
        Skilled forgeries for test

    Returns
    -------
    np.ndarray, np.ndarray, np.ndarray
        The predictions(scores) for genuine signatures,
        random forgeries and skilled forgeries

    """
    # Get predictions
    genuinePred = model.decision_function(genuine_signatures)
    randomPred = model.decision_function(random_forgeries)
    skilledPred = model.decision_function(skilled_forgeries)

    return genuinePred, randomPred, skilledPred


def train_all_users(exp_train: Tuple[np.ndarray, np.ndarray, np.ndarray],
                    dev_set: Tuple[np.ndarray, np.ndarray, np.ndarray],
                    svm_type: str,
                    C: float,
                    gamma: float,
                    num_forg_from_dev: int,
                    num_forg_from_exp: int,
                    rng: np.random.RandomState) -> Dict[int, sklearn.svm.SVC]:
    """ Train classifiers for all users in the exploitation set

    Parameters
    ----------
    exp_train: tuple of np.ndarray (x, y, yforg)
        The training set split of the exploitation set (system users)
    dev_set: tuple of np.ndarray (x, y, yforg)
        The development set
    svm_type: string ('linear' or 'rbf')
        The SVM type
    C: float
        Regularization for the SVM optimization
    gamma: float
        Hyperparameter for the RBF kernel
    num_forg_from_dev: int
        Number of forgeries from each user in the development set to
        consider as negative samples
    num_forg_from_exp: int
        Number of forgeries from each user in the exploitation set (other
        than the current user) to consider as negative sample.
    rng: np.random.RandomState
        The random number generator (for reproducibility)

    Returns
    -------
    Dict int -> sklearn.svm.SVC
        A dictionary of trained classifiers, where the keys are the users.

    """
    classifiers = {}

    exp_y = exp_train[1]
    users = np.unique(exp_y)

    if num_forg_from_dev > 0:
        other_negatives = data.get_random_forgeries_from_dev(dev_set, num_forg_from_dev, rng)
    else:
        other_negatives = []

    for user in tqdm(users):
        training_set = data.create_training_set_for_user(user, exp_train, num_forg_from_exp, other_negatives, rng)
        classifiers[user] = train_wdclassifier_user(training_set, svm_type, C, gamma)

    return classifiers


def test_all_users(classifier_all_user: Dict[int, sklearn.svm.SVC],
                   exp_test: Tuple[np.ndarray, np.ndarray, np.ndarray],
                   global_threshold: float) -> Dict:
    """ Test classifiers for all users and return the metrics

    Parameters
    ----------
    classifier_all_user: dict (int -> sklearn.svm.SVC)
        The trained classifiers for all users
    exp_test: tuple of np.ndarray (x, y, yforg)
        The testing set split from the exploitation set
    global_threshold: float
        The threshold used to compute false acceptance and
        false rejection rates

    Returns
    -------
    dict
        A dictionary containing a variety of metrics, including
        false acceptance and rejection rates, equal error rates

    """
    xfeatures_test, y_test, yforg_test = exp_test

    genuinePreds = []
    randomPreds = []
    skilledPreds = []

    users = np.unique(y_test)
    for user in users:
        model = classifier_all_user[user]

        # Test the performance for the user without replicates
        skilled_forgeries = xfeatures_test[(y_test == user) & (yforg_test == 1)]
        test_genuine = xfeatures_test[(y_test == user) & (yforg_test == 0)]
        random_forgeries = xfeatures_test[(y_test != user) & (yforg_test == 0)]

        genuinePredUser = model.decision_function(test_genuine)
        skilledPredUser = model.decision_function(skilled_forgeries)
        randomPredUser = model.decision_function(random_forgeries)

        genuinePreds.append(genuinePredUser)
        skilledPreds.append(skilledPredUser)
        randomPreds.append(randomPredUser)

    # Calculate al metrics (EER, FAR, FRR and AUC)
    all_metrics = metrics.compute_metrics(genuinePreds, randomPreds, skilledPreds, global_threshold)

    results = {'all_metrics': all_metrics,
               'predictions': {'genuinePreds': genuinePreds,
                               'randomPreds': randomPreds,
                               'skilledPreds': skilledPreds}}

    print(all_metrics['EER'], all_metrics['EER_userthresholds'])
    return results


def train_test_all_users(exp_set: Tuple[np.ndarray, np.ndarray, np.ndarray],
                         dev_set: Tuple[np.ndarray, np.ndarray, np.ndarray],
                         svm_type: str,
                         C: float,
                         gamma: float,
                         num_gen_train: int,
                         num_forg_from_exp: int,
                         num_forg_from_dev: int,
                         num_gen_test: int,
                         global_threshold: float = 0,
                         rng: np.random.RandomState = np.random.RandomState()) \
        -> Tuple[Dict[int, sklearn.svm.SVC], Dict]:
    """ Train and test classifiers for every user in the exploitation set,
        and returns the metrics.

    Parameters
    ----------
    exp_set: tuple of np.ndarray (x, y, yforg)
        The exploitation set
    dev_set: tuple of np.ndarray (x, y, yforg)
        The development set
    svm_type: string ('linear' or 'rbf')
        The SVM type
    C: float
        Regularization for the SVM optimization
    gamma: float
        Hyperparameter for the RBF kernel
    num_gen_train: int
        Number of genuine signatures available for training
    num_forg_from_dev: int
        Number of forgeries from each user in the development set to
        consider as negative samples
    num_forg_from_exp: int
        Number of forgeries from each user in the exploitation set (other
        than the current user) to consider as negative sample.
    num_gen_test: int
        Number of genuine signatures for testing
    global_threshold: float
        The threshold used to compute false acceptance and
        false rejection rates
    rng: np.random.RandomState
        The random number generator (for reproducibility)

    Returns
    -------
    dict (int -> sklearn.svm.SVC)
        The classifiers for all users

    dict
        A dictionary containing a variety of metrics, including
        false acceptance and rejection rates, equal error rates

    """
    exp_train, exp_test = data.split_train_test(exp_set, num_gen_train, num_gen_test, rng)

    classifiers = train_all_users(exp_train, dev_set, svm_type, C, gamma,
                                  num_forg_from_dev, num_forg_from_exp, rng)

    results = test_all_users(classifiers, exp_test, global_threshold)

    return classifiers, results
