import cv2
from joblib import load
import numpy as np
import dlib
from imutils import face_utils
import os
import imutils

NAMELIST = sorted(os.listdir('face_dataset_pruned'))


class Aligner:
    def __init__(self,
                 landmark_annotator=dlib.shape_predictor('dlib-models/shape_predictor_5_face_landmarks.dat'),
                 desired_left_eye=(0.35, 0.35),
                 desired_face_width=250,
                 desired_face_height=300):
        self.landmark_annotator = landmark_annotator
        self.desired_left_eye = desired_left_eye
        self.desired_face_width = desired_face_width
        self.desired_face_height = desired_face_height

    def annotate_landmark(self, image, rect):
        shape = self.landmark_annotator(image, rect)
        shape = face_utils.shape_to_np(shape)

        # for (i, (x, y)) in enumerate(shape):
        #     image = cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
        #     image = cv2.putText(image, str(i + 1), (x - 10, y - 10),
        #                         cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        return shape

    @staticmethod
    def left_eye_center(shape):
        return (shape[0] + shape[1]) // 2

    @staticmethod
    def right_eye_center(shape):
        return (shape[2] + shape[3]) // 2

    def align(self, image, rect):
        shape = self.annotate_landmark(image, rect)
        left_eye_center = self.left_eye_center(shape)
        right_eye_center = self.right_eye_center(shape)

        dy = right_eye_center[1] - left_eye_center[1]
        dx = right_eye_center[0] - left_eye_center[0]
        angle = np.degrees(np.arctan2(dy, dx)) - 180

        desired_right_eye_x = 1.0 - self.desired_left_eye[0]
        dist = np.sqrt((dx ** 2) + (dy ** 2))
        desired_dist = (desired_right_eye_x - self.desired_left_eye[0])
        desired_dist *= self.desired_face_width
        scale = desired_dist / dist

        eyes_center = ((left_eye_center[0] + right_eye_center[0]) // 2,
                       (right_eye_center[1] + right_eye_center[1]) // 2)
        mat = cv2.getRotationMatrix2D(eyes_center, angle, scale)
        tx = self.desired_face_width * 0.5
        ty = self.desired_face_height * self.desired_left_eye[1]
        mat[0, 2] += (tx - eyes_center[0])
        mat[1, 2] += (ty - eyes_center[1])

        # apply the affine transformation
        (w, h) = (self.desired_face_width, self.desired_face_height)
        output = cv2.warpAffine(image, mat, (w, h), flags=cv2.INTER_CUBIC)
        return output


class Display:
    def __init__(self,
                 detector=dlib.get_frontal_face_detector(),
                 aligner=Aligner(),
                 stream=cv2.VideoCapture(0),
                 detection_confidence=0.90,
                 pca=load('pca.joblib'),
                 classifier=load('classif.joblib'),
                 pca_input_size=(300, 250)):

        self.detector = detector
        self.aligner = aligner
        self.stream = stream
        self.detection_confidence = detection_confidence
        self.pca = pca
        self.classifier = classifier
        self.pca_input_size = pca_input_size

    def transform_image(self, image):
        # image = cv2.resize(image, self.pca_input_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.dnn.blobFromImage(image, 1 / image.std(), self.pca_input_size, image.mean())
        image = image.flatten()

        return image

    def who_is_it(self, image):
        transformed_image = self.transform_image(image)
        reduced_image = self.pca.transform(transformed_image.reshape(1, -1))
        classif_result = self.classifier.predict(reduced_image).item()
        person_name = NAMELIST[classif_result]

        return person_name

    def draw_face_with_landmarks(self, image, detections):
        aligned_faces = []
        displayed_image = image.copy()
        for rect in detections:

            (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
            displayed_image = cv2.rectangle(displayed_image, (bX, bY), (bX + bW, bY + bH),
                                            (0, 255, 0), 1)

            aligned_face = self.aligner.align(image, rect)
            # import pdb
            # pdb.set_trace()
            displayed_image = cv2.putText(displayed_image, self.who_is_it(aligned_face), (bX, bY),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            aligned_faces.append(aligned_face)

        return aligned_faces, displayed_image

    @staticmethod
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

    def run(self):
        while True:
            ret, frame = self.stream.read()

            detections = self.detector(frame, 0)
            aligned_faces, displayed_image = self.draw_face_with_landmarks(frame, detections)

            cv2.imshow('EE4208', displayed_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # When everything done, release the capture
        self.stream.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    display = Display()
    display.run()
