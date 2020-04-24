import os
import sys
import glob

import cv2
import numpy as np


IMAGETYPES = ('*.bmp', '*.png', '*.jpg', '*.jpeg', '*.tif') # Supported image types


def get_image_paths(file_root, pattern=None):
    """ Get ordered list of filepaths
    """
    file_paths = []
    for typ in IMAGETYPES:
		# search all files conforming to the IMAGE_TYPES extensions
        file_paths.extend(glob.glob(os.path.join(file_root,'**',typ), recursive=True))
    # filter filenames
    if pattern is not None:
        filtered = []
        filtered = [f for f in file_paths if pattern in os.path.split(f)[-1]]
        file_paths = filtered
        del filtered
	# sort image file paths
    file_paths.sort()
    return file_paths


def load_face_dataset(file_root):
    """ Load all images and also list out the filepaths
    """
    if not os.path.isdir(file_root):
        raise FileNotFoundError("Face dataset file root: {} not found.".format(file_root))

    file_paths = get_image_paths(file_root)
    
    if len(file_paths) == 0:
        warnings.warn("No image found in face dataset file root: {}".format(file_root))
        return []
    
    images = [cv2.imread(file_path) for file_path in file_paths]
    return images, file_paths
