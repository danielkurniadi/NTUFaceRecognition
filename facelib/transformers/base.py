import cv2
import dlib
import numpy as np
from imutils import face_utils

from tqdm import tqdm

import sklearn
from sklearn.base import BaseEstimator, TransformerMixin

from abc import ABC, abstractmethod


class BaseDetector(ABC, BaseEstimator, TransformerMixin):
    """ Base class for detector transforms
    """

    def fit(self, *args, **kwargs):
        return self

    @abstractmethod
    def transform(self, inputs):
        pass

    @abstractmethod
    def _detect_single_face(self, image):
        return dlib.rectangle()

    @abstractmethod
    def _detect_multi_faces(self, image):
        return [dlib.rectangle()]


class LambdaTransform(BaseEstimator, TransformerMixin):
    """ Mapped transformer that you can create by assigning a function that receive
    input and return output
    """

    def __init__(self, hook_function):
        self._transform = hook_function

    def fit(self, *args, **kwargs):
        return self

    def transform(self, *args, **kwargs):    
        return [self._transform(*x, **kwargs) for x in tqdm(zip(args))]
