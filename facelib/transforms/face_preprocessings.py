import os, sys
import glob

from tqdm import tqdm

import cv2
import dlib
import numpy as np
from imutils import face_utils

import sklearn
from sklearn.base import BaseEstimator, TransformerMixin

from .base import BaseDetector, LambdaTransform


__all__ = [
    "FaceHaarDetectCrop",
    "FaceAlign",
]

imageMean = [104.0, 177.0, 123.0]
confThresh = 0.5


class FaceHaarDetectCrop(object):
    """ Face detector that detect faces in an image and yield cropped faces image
    """

    def __init__(self, prototxt, caffeModel, imsize=(300,300), 
                 multi_face=False, strict=False):
        """ Initialise face detector

        Args:
            .. prototxt   (str/path): path to prototxt file defining the architecture
            .. caffeModel (str/path): path to checkpoint file of pretrained caffe model
            .. imsize   (Tuple[int]): height and width of image blob
            .. multi_face  (bool): whether to detect only multiple or single face per image
            .. strict (bool): if no face detected, raise error. only if multi_face = False
        """
        if not os.path.isfile(prototxt):
            raise FileNotFoundError("Prototxt file: {} not found.".format(prototxt))
        
        if not os.path.isfile(caffeModel):
            raise FileNotFoundError("CaffeModel file: {} not found.".format(caffeModel))

        self.imsize = imsize
        self.multi_face = multi_face
        self.detector = cv2.dnn.readNetFromCaffe(prototxt, caffeModel=caffeModel)

    def _detect_single_face(self, image):
        """ Apply detection assuming only one face per image
        
        Args:
            .. image (np.ndarray): images in numpy tensor form of shape [C,H,W]
        
        Returns:
            .. face (np.ndarray or None): image in numpy tensor of shape [C,H,W]. None if no face
        """
        h, w = image.shape[:2]

        # construct a blob from the image
        imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, self.imsize), 1.0, self.imsize, imageMean)

        # apply OpenCV's deep learning-based face detector to localize
        # faces in the input image
        self.detector.setInput(imageBlob)
        detections = self.detector.forward()
        
        # ensure at least one face was found
        if len(detections) > 0:
            # we're making the assumption that each image has only ONE
            # face, so find the bounding box with the largest probability
            i = np.argmax(detections[0, 0, :, 2])
            confidence = detections[0, 0, i, 2]

            # filter out weak detections
            if confidence > confThresh:
                # compute the (x, y)-coordinates of the bounding box for
                # the face
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # grab the ROI dimensions
                (fH, fW) = endY - startY, endX - startX

                # ensure the face width and height are sufficiently large
                if fW > 40 and fH > 40:
                    rect = dlib.rectangle(startX, startY, startX+fW, startY+fH)
                    return rect

        if strict:
            raise ValueError("Unable to detect face during FaceDetection in one of the image passed.\n"
                             "Ensure that image is clean and has face in it. "
                             "You can also \n increase the threshold.")
        return None
    
    def _detect_multi_faces(self, image):
        """ Apply detection assuming only one face per image
        
        Args:
            .. image (np.ndarray): images in numpy tensor form of shape [C,H,W]
        
        Returns:
            .. face_images (np.ndarray or None): array of faces images of shape 
        """
        blob = cv2.dnn.blobFromImage(cv2.resize(image, self.imsize, 1.0, self.imsize, imageMean))

        # apply OpenCV's deep learning-based face detector to localize
        # faces in the input image
        self.detector.setInput(imageBlob)
        detections = self.detector.forward()

        rects = []
        
        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections with less than 50% confidence
            if confidence > 0.5:
                # compute the (x, y)-coordinates of the bounding box for
                # the face
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # grab the ROI dimensions
                (fH, fW) = endY - startY, endX - startX

                # ensure the face width and height are sufficiently large
                if fW < 40 or fH < 40:
                    continue

                rect = dlib.rectangle(startX, startY, startX+fW, startY+fH)
                rects.append(rect)

        return rects

    def __call__(self, image):
        """ Perform face detection and cropping for batch of images
        
        Args:
            .. image (np.ndarray): image in tensor form of shape [C,H,W]

        Returns:
            .. rect (dlib.rectangle): rectangle of bounding boxes
                    If multiFaceMode false, output is bounding box: rectangle(x1,y1,x2,y2)
                    If multiFaceMode true, output is array of bounding box List[rectangle(x1,y1.x2,y2)].
        """
        if self.multiFaceMode:
            rects = self._detect_multi_faces(image)
            return rects

        rect = self._detect_single_face(image)
        return rect


class FaceAlignHorizontal:
    """ Face aligner that align faces by finding horizontal line between eyes marker 
    using dlib facial marking then rotate the image to make the horizon line to 0 degree.

    """

    def __init__(face_landmark_model, desired_left_eye=(0.23, 0.23)):
        """ Initialize face aligner

        Args:
            .. face_landmark_model (str/path): path to facial landmarker model file
            .. desired_left_eye (float, float): (x, y) location of left eye relative to image size.
        """
        self.predictor = dlib.shape_predictor(face_landmark_model)

        # image configurations
        self.desired_left_eye = desired_left_eye
        self.imsize = imsize

    def _align_face(self, image, rect):
        """ Rotate and affine transformation to align face in the image to 0 degree.

        Args:
            .. image (np.ndarray): image face of shape [H,W,C]
            .. rect (dlib.rectangle): rect bounding box representation
        """
        if len(image.shape) == 3 and image.shape[2] == 3:
            # BGR image with 3 channel to gray
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        elif len(image.shape) == 2:
            gray = np.squeeze(image)

        # predict 5 facial landmarks coordinates and
        # convert the landmark (x, y) ~ coordinates to numpy array
        shape = self.predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # compute the center of mass for each eye
        left_eye_center = (shape[0] + shape[1]) / 2.0
        right_eye_center = (shape[2] + shape[3]) / 2.0

        # and compute the angle between the eye centroids
        dy = right_eye_center[1] - left_eye_center[1]
        dx = right_eye_center[0] - left_eye_center[0]
        angle = np.degrees(np.arctan2(dy, dx)) - 180
        
        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        desired_right_eye_x = 1.0 - self.desired_left_eye[0]

        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist = np.sqrt((dx ** 2) + (dy ** 2))
        desired_dist = (desired_right_eye_x - self.desired_left_eye[0])
        desired_dist *= desired_face_width
        scale = desired_dist / dist

        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        eyes_center = ((left_eye_center[0] + right_eye_center[0]) // 2,
                       (right_eye_center[1] + right_eye_center[1]) // 2)

        # grab the rotation matrix for rotating and scaling the face
        rot_mat = cv2.getRotationMatrix2D(eyes_center, angle, scale)

        tx = desired_face_width * 0.5
        ty = desired_face_height * self.desired_left_eye[1]

        rot_mat[0, 2] += (tx - eyes_center[0])
        rot_mat[1, 2] += (ty - eyes_center[1])

        # apply the affine transformation
        (w, h) = (desired_face_width, desired_face_height)
        face = cv2.warpAffine(image, rot_mat, (w, h), flags=cv2.INTER_CUBIC)

        return face

    def __call__(self, image, rect):
        """ Perform face alignment. Assuming every image is one face.
        """
        return self._align_face(image, rect)
