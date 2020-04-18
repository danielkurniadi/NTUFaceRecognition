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
    "FaceHaarDetector",
    "FaceAligner",
    "crop_face_bounding_box"
]

imageMean = [104.0, 177.0, 123.0]
confThresh = 0.5


class FaceHaarDetector(BaseDetector):
    """ Face dataset transformer that detect faces in an image and yield bounding box coordinate.
    Works best with single face per input image but support multiple faces per image.

    Note: transformer will return tuple of (original image, rect) objects. You can further use the rect object to
    crop the image or feed it to another transformer.
    """
    
    def __init__(self, multi_faces_mode=False):
        """ Initialise face detector

        Args:
            .. multi_faces_mode  (bool): 
                whether to detect only multiple or single face for each image
        """
        self.multi_faces_mode = multi_faces_mode
        
        # construct dlib haar cascade model
        self.detector = dlib.get_frontal_face_detector()

    # -----------------------
    # Transformer methods
    # -----------------------

    def transform(self, images):
        """ Perform face detection and cropping for batch of images
        
        Args:
            .. images (np.ndarray):
                    array of images in tensor form of shape [H,W,C]

        Returns:
            .. image_rects List[np.ndarray, dlib.rectangle] or List[np.ndarray, List[dlib.rectangle]]:
                    return List of Tuple of image, rect, where image is of shape [H,W,C] and 
                    rect is a bounding box of face coordinates ~ (x,y,w,h).
                    
                    if multi_faces_mode false, output is array of rect (single mode)
                    if multi_faces_mode true, output is array of array of rect (multiple mode)
        """
        if self.multi_faces_mode == True:
            image_rects = [(image, self._detect_single_face(image)) for image in tqdm(images)]
        else:
            image_rects = [(image, self._detect_multi_faces(image)) for image in tqdm(images)]
        return image_rects

    # -----------------------
    # Face detector helpers
    # -----------------------

    def _detect_single_face(self, image):
        """ Apply detection assuming only one face per image
        
        Args:
            .. image (np.ndarray): images in numpy tensor form of shape [H,W,C]
        
        Returns:
            .. face (np.ndarray or None): image in numpy tensor of shape [H,W,C]. None if no face
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detect face bounding box assuming single face
        rect = self.detector(gray, 1)[0]

        return rect

    def _detect_multi_faces(self, image):
        """ Apply detection assuming only one face per image
        
        Args:
            .. image (np.ndarray): images in numpy tensor form of shape [H,W,C]
        
        Returns:
            .. face (np.ndarray or None): image in numpy tensor of shape [H,W,C]. None if no face
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detect face bounding boxes
        rects = self.detector(gray, 1)

        return rects


# -----------------------------------------------------------------------------

def crop_face_bounding_box(multi_faces_mode=False):
    """ Create transformers that consume data of Tuple of (image, rect) or (image, List[rect])
    then perform face cropping and yield cropped face image or List[face]. Face image is of shape [H,W,C].

    Here rect is a dlib.rectangle object representing the face bounding box.
    
    Args:
        .. multi_faces_mode  (bool): 
                whether to detect only multiple or single face for each image
    """

    def _crop_from_rect(image, rect):
        # single face mode where only one rectangle per image
        # rectangle here denote bounding box of a face in an image
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        face = image[y:y+h, x:x+w, :]
        return face

    def _crop_from_rects(image, rects):
        # multi face mode where one image map to many rectangles
        # rectangle here denote bounding box of a face in an image
        faces = []
        for rect in rects:
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            face = image[y:y+h, x:x+w, :]
            faces.append(face)
        return faces

    if multi_faces_mode == True:
        return LambdaTransform(_crop_from_rects)

    return LambdaTransform(_crop_from_rect)

# -----------------------------------------------------------------------------

class FaceAligner(BaseEstimator, TransformerMixin):
    """ Face dataset transformer that detect faces in an image and yield cropped faces image.

    Optionally and when specified, it will also perform align faces by finding eyes horizon.
    line using dlib facial marking then rotate the image to make the horizon line to 0 degree.

    Note: transformer will receive element tuple of (image, rect) object, which comes from FaceDetector transformer.
    """
    
    def __init__(self, face_landmark_model, imsize=(300,300), desired_left_eye=(0.23, 0.23),
                multi_faces_mode=False):
        """ Initialise face aligner

        Args:
            .. face_landmark_model (str/path): 
                    Path to dlib predictor model for facial landmark.

            .. desired_left_eye (Tuple[float]): 
                    (x, y) tuple with the default shown, specifying the 
                    desired output left eye position
            
            .. imsize (Tuple[int]):
                    (height, width) tuple specifying desired output face image size
            
            .. multi_faces_mode (bool):
                    whether each image has multiple rectangles. if true, rect in each input to transforms
                    will be regarded as List of dlib.rectangle.
        """
        if not os.path.isfile(face_landmark_model):
            raise FileNotFoundError("Dlib facial landmark model (.dat) file: {} not found."
                                    .format(face_landmark_model))

        # define dlib dnn
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(face_landmark_model)

        # image configurations
        self.desired_left_eye = desired_left_eye
        self.imsize = imsize

        self.multi_faces_mode = multi_faces_mode

    # -----------------------
    # Transformer methods
    # -----------------------

    def fit(self, *args, **kwargs):
        return self

    def transform(self, images_rects):
        """ Perform face cropping and alignment for batch of images.

        Args:
            .. images_rects (List[np.ndarray, dlib.rectangles]): array of images in tensor form of shape [H,W,C]

        Returns:
            .. faces (List[np.ndarray] or List[List[np.ndarray]]): 
                    if multi_faces_mode false, output is array of image (single mode)
                    if multi_faces_mode true, output is array of array of image faces (multiple mode)
        """
        images, rects = images_rects
        faces = []
        if self.multi_faces_mode == True:
            # each element in faces is a list of image
            faces = [self._align_multi_faces(image, _rects)
                    for image, _rects in tqdm(zip(images, rects))]
        else:
            # each element in faces is image
            faces = [self._align_single_face(image, rect)
                    for image, rect in tqdm(zip(images, rects))]
        return faces

    # -----------------------
    # Face detector helpers
    # -----------------------

    def _align_multi_faces(self, image, rects):
        """ Rotate and affine transformation to align face in the image to 0 degree.
        Args:
            .. image (np.ndarray): image face of shape [H,W,C]
            .. rect (dlib.rectangles): rect bounding box representation
        
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = [self._align_single_face(image, rect) for rect in rects]
        return faces

    def _align_single_face(self, image, rect):
        """ Rotate and affine transformation to align face in the image to 0 degree.
        Args:
            .. image (np.ndarray): image face of shape [H,W,C]
            .. rect (dlib.rectangles): rect bounding box representation
        """
        if image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        elif image.shape[2] == 1:
            gray = image

        # predict 68 facial landmarks coordinates and
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
