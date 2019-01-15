from abc import ABC, abstractmethod
from typing import List, Iterable, Tuple
import numpy as np


class IterableDataset(ABC):
    """ Base class for Datasets. The dataset classes specify information about
     the dataset (such as the number of signatures per user), as well as implement
     methods to iterate over the signatures of an user. """

    @property
    def maxsize(self):
        """ Maximum size of the images in the dataset"""
        raise NotImplementedError

    @property
    def genuine_per_user(self):
        """ Number of genuine signatures per user"""
        raise NotImplementedError

    @property
    def skilled_per_user(self):
        """ Number of skilled forgeries per user"""
        raise NotImplementedError

    @property
    def simple_per_user(self):
        """ Number of simple forgeries per user"""
        raise NotImplementedError

    @abstractmethod
    def get_user_list(self) -> List[int]:
        """ Returns the list of users in the dataset. """
        pass

    @abstractmethod
    def iter_genuine(self, user: int) -> Iterable[Tuple[np.ndarray, str]]:
        """ Iterates through the genuine signatures of an user

        Parameters
        ----------
        user : int
            The user, from whom the signatures will be returned

        Returns
        -------
        Generator (iterable) of <image, filename>

        """
        pass

    @abstractmethod
    def iter_simple_forgery(self, user: int) -> Iterable[Tuple[np.ndarray, str]]:
        """ Iterates through the simple forgeries of an user

        Parameters
        ----------
        user : int
            The user, from whom the signatures will be returned

        Returns
        -------
        Generator (iterable) of <image, filename>

        """
        pass

    @abstractmethod
    def iter_forgery(self, user: int) -> Iterable[Tuple[np.ndarray, str]]:
        """ Iterates through the skilled forgeries of an user

        Parameters
        ----------
        user : int
            The user, from whom the signatures will be returned

        Returns
        -------
        Generator (iterable) of <image, filename>

        """
        pass
