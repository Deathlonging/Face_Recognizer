#!/usr/bin/python

import cv2
import time

from image_source import ImageSource

class WebcamReader(ImageSource):
    def __init__(self, resolution=(640, 480),framerate=32):
        self._camera = cv2.VideoCapture(0)
        self._is_running = True

    def getImages(self):
        time.sleep(0.1)
        assert True == self._camera.isOpened()
        while self._is_running:
            ret, image = self._camera.read()
            if False == ret:
                break
            yield image
        self._camera.release()

    def stop(self):
        self._is_running = False

