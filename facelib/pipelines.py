import os
import sys
import time
import glob

import cv2
import numpy as np
import scipy
import imutils
from imutils import face_utils

from .transforms import FaceHaarDetectCrop, FaceAlign
from .face_dataset import FaceDataset


# Define directories and paths
lib_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(lib_dir)
models_dir = os.path.join(root_dir, 'models/')


# -------------------------------
# BASE CLASS
# -------------------------------

class BaseFacePipeline(object):
	""" Abstract class for end-to-end face image processing pipeline.

	Scope of pipeline class is to provide convenient to developer facing API. Use cases for training and
	live real-time testing.

	Hence, pipeline acts as a glue code with developer-defined data flow
	and customisable for multi-threaded or multi-process environment, whichever convenient.

	"""
	__abstract__ = True

	_IMSIZE = (300, 300)
	_CONFTHRESH = 0.6
	_LEFT_EYE_POS = (0.23, 0.23)

	__face_detection_model = os.path.join(models_dir, 'haar10_300x300_iter_140000.caffemodel')
	__face_detection_prototxt = os.path.join(models_dir, 'deploy.prototxt')
	__face_landmark_predictor = os.path.join(models_dir, 'shape_predictpr_5_face_landmark.dat')

	def __init__(self, data_loader, save_processed=False, multi_face=False):
		""" Initialize transformation stages and setups
		"""
		self.color_mode = color_mode
		self.multi_face = multi_face
		self.save_processed = save_processed

		self._face_detect_crop = FaceHaarDetectCrop(
			self.__face_detection_prototxt,
			self.__face_detection_prototxt,
			self._CONFTHRESH,
			multi_face=multi_face,
			strict=False
		)

		self._face_align = FaceAlign(
			self.__face_landmark_predictor,
			self._LEFT_EYE_POS
		)

		self.data_loader = data_loader

		# TODO: make it multithread? do something like this:
		# from multithreading import Queue
		# face_rect_queue = Queue()
		# face_align_queue = Queue()

		# detect_crop_thread = Thread(self._face_detect_crop, args, queue=face_rect_queue)
		# ....

	def process_face_image(self, image):
		""" Parse data from the data loader
		"""
		rects = self._face_detect_crop(image)

		if self.multi_face == True:
			# rects is List[dlib.rectangle]
			aligned = [self._face_align(image, rect) for rect in rects]
		
		# rects is a dlib.rectangle
		aligned = self._face_align(image, rects)

		return aligned

	def __call__(self, n_jobs=None):
		""" Run pipeline and yield result, also handles saving processed
		result if needed
		"""
		raise NotImplementedError


# -------------------------------
#  CLASS
# -------------------------------

class FaceTrainingPipeline(BaseFacePipeline):
	""" Training pipeline for end-to-end face image processing.

	Scope of pipeline class is to provide convenient to developer facing API. Use cases for training and
	live real-time testing.

	Hence, pipeline acts as a glue code with developer-defined data flow
	and customisable for multi-threaded or multi-process environment, whichever convenient.

	"""
	def __init__(self, file_root, clf_model, embedding_model=None, color_mode=cv2.IMREAD_COLOR,
				save_processed=False, multi_face=False):
		""" Initialise training pipeline that load data from keras-style data directory
		and perform training or evaluation depending on `train` mode.
		"""
		self.file_root = file_root
		self.clf_model = clf_model
		self.embedding_model = embedding_model
		
		face_dataset = FaceDataset(file_root, color_mode)
		super().__init__(face_dataset, save_processed, multi_face)

	def __call__(self, n_jobs=None):
		""" Run pipeline for training and yield results.
		"""
		# TODO: refactor and add multi-threading for computation optimisation
		for image, label in self.data_loader:
			processed_faces = self.process_face_image(image)
			processed_faces = np.array([
				cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
				for face in processed_faces])

		n, h, w = processed_faces.shape
		face_vectors = processed_faces.reshape(n, -1)

		embedding_model = embedding_model.fit(face_vectors)
		embedding_feats = embedding_model.transform(face_vectors)

		clf_model = clf_model.fit(embedding_feats)

		return
