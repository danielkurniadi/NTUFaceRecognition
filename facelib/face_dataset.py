
import os
import sys
import time
import glob

import cv2
import numpy as np

from .utils import get_image_paths


class FaceDataset:
    """ Face dataset scope is to load image data from disk
    into operable numpy array and label. The data directory must conforms to
	keras-style data directory.

    """

    def __init__(self, file_root, color_mode=cv2.IMREAD_COLOR):
        """ Initialise face dataset
        """
		self.color_mode = color_mode
        self.file_paths = get_image_paths(file_root)
        self._index = -1

	@staticmethod
	def _get_label_from_filepath(file_path):
		file_path = os.path.abspath(file_path)
		file_dir = os.path.dirname(file_path)
		label = os.path.basename(file_dir)

		return label

	def __iter__(self):
		self._index = -1
		return self

    def __next__(self):
        self._index += 1

		if self._index == len(self.file_paths)-1:
			raise StopIteration

		file_path = self.file_paths[self._index]

		label = self._get_label_from_filepath(file_path)
		image = cv2.imread(file_path, self.color_mode)

		if image is None:
			raise FileNotFoundError("Face data not found in {0}".format(file_path))

		return image, label

