#!/usr/bin/python

import cv2
import os

class FaceDetector:
    def __init__(self, cascade_file_path, scaleFactor=1.1,minNeighbors=5,minFaceSize=(30,30)):
        assert True == os.path.exists(cascade_file_path)
        self._detector = cv2.CascadeClassifier(cascade_file_path)
        self._scaleFactor = scaleFactor
        self._minNeighbors = minNeighbors
        self._minFaceSize = minFaceSize

    def detectFaces(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_bounding_boxes = self._detector.detectMultiScale(
            gray_image, 
            scaleFactor=self._scaleFactor, 
            minNeighbors=self._minNeighbors, 
            minSize=self._minFaceSize
            )
        return face_bounding_boxes