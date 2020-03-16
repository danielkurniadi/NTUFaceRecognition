import os
import glob
import cv2
import numpy as np
import dlib
from live import Aligner
from imutils import face_utils
import imutils
import joblib


def _pad_with_black(image):
    old_shape = image.shape[:2]
    target_size = max(old_shape)
    ratio = float(target_size) / max(old_shape)
    new_size = tuple([int(x * ratio) for x in old_shape])

    image = cv2.resize(image, (new_size[1], new_size[0]))

    delta_w = target_size - new_size[1]
    delta_h = target_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                      value=color)

    return padded_image


class FaceDataset:
    def __init__(self,
                 data_root='face_dataset',
                 names=None,
                 detector_prototxt='models/deploy.prototxt',
                 detector_weights='models/res10_300x300_ssd_iter_140000_fp16.caffemodel',
                 detection_confidence=0.5):

        if names is None:
            names = sorted(os.listdir('face_dataset'))

        self.data_root = data_root
        self.names = names
        self.detector_prototxt = detector_prototxt
        self.detector_weights = detector_weights
        self.detection_confidence = detection_confidence
        self.rect = joblib.load('rectangle.joblib')

        self.filenames = []
        self.labels = []
        for i, name in enumerate(self.names):
            person_image_filenames = sorted(glob.glob(os.path.join(self.data_root, name, '*')))
            self.filenames.extend(person_image_filenames)
            self.labels.extend([i] * len(person_image_filenames))

        self.detector = cv2.dnn.readNetFromCaffe(self.detector_prototxt, self.detector_weights)

    def __getitem__(self, i):
        return self.filenames[i], self._get_and_crop_image(self.filenames[i]), self.labels[i]

    def __len__(self):
        return len(self.filenames)

    def _get_and_crop_image(self, filename):
        image = cv2.imread(filename)
        padded_image = _pad_with_black(image)
        pure_face_image = self._detect_face(padded_image)

        return pure_face_image

    def _detect_face(self, cropped_image):
        blob = cv2.dnn.blobFromImage(cv2.resize(cropped_image, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))

        self.detector.setInput(blob)
        detections = self.detector.forward()

        pure_face_image = self._get_detections(cropped_image, detections)
        return pure_face_image

    def _get_detections(self, cropped_image, detections):
        h, w = cropped_image.shape[:2]
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]
            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > self.detection_confidence:
                # compute the (x, y)-coordinates of the bounding box for the
                # object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (start_x, start_y, end_x, end_y) = box.astype("int")
                pure_face_image = cropped_image[start_y: end_y, start_x: end_x]

                return pure_face_image

        raise ValueError("No face detected, reduce the confidence threshold")


class AlignedFaceDataset:
    def __init__(self,
                 data_root='face_dataset_pruned',
                 names=None,
                 detector=dlib.get_frontal_face_detector(),
                 aligner=Aligner()
                 ):
        if names is None:
            names = sorted(os.listdir('face_dataset'))

        self.data_root = data_root
        self.names = names
        self.detector = detector
        self.aligner = aligner

        self.filenames = []
        self.labels = []
        for i, name in enumerate(self.names):
            person_image_filenames = sorted(glob.glob(os.path.join(self.data_root, name, '*')))
            self.filenames.extend(person_image_filenames)
            self.labels.extend([i] * len(person_image_filenames))

    def __getitem__(self, i):
        return self.filenames[i], self.get_aligned_face(self.filenames[i]), self.labels[i]

    def __len__(self):
        return len(self.filenames)

    def draw_face_with_landmarks(self, image, detections):
        aligned_faces = []
        displayed_image = image.copy()
        for rect in detections:
            # compute the bounding box of the face and draw it on the
            # frame
            (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
            displayed_image = cv2.rectangle(displayed_image, (bX, bY), (bX + bW, bY + bH),
                                            (0, 255, 0), 1)
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array

            aligned_face = self.aligner.align(image, rect)
            aligned_faces.append(aligned_face)

        return aligned_faces, displayed_image

    def get_aligned_face(self, filename):
        image = cv2.imread(filename)
        image = imutils.resize(image, width=300)
        padded_image = _pad_with_black(image)
        detections = self.detector(padded_image, 0)
        aligned_faces, _ = self.draw_face_with_landmarks(padded_image, detections)

        # if len(aligned_faces) != 1:
        #     raise RuntimeError("{}, # of detected face(s): {}".format(filename, len(aligned_faces)))

        return aligned_faces[0]


class NormalizedDataset:
    def __init__(self, face_dataset=AlignedFaceDataset(), size=(300, 250)):
        self.face_dataset = face_dataset
        self.size = size

    def __getitem__(self, i):
        filename, image, label = self.face_dataset[i]

        image = cv2.resize(image, self.size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.dnn.blobFromImage(image, 1 / image.std(), self.size, image.mean())
        image = image.flatten()

        return filename, image, label

    def __len__(self):
        return len(self.face_dataset)


if __name__ == "__main__":
    FaceDataset()
