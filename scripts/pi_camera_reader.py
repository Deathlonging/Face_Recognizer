#!/usr/bin/python

from picamera import PiCamera
from picamera.array import PiRGBArray
import time

from image_source import ImageSource

class PiCameraReader(ImageSource):
    def __init__(self, resolution=(640, 480),framerate=32):
        self._camera = PiCamera()
        self._camera.resolution = resolution
        self._camera.framerate = framerate
        self._rawCapture = PiRGBArray(self._camera, size=resolution)
        self._is_running = True

    def getImages(self):
        time.sleep(0.1)
        for frame in self._camera.capture_continuous(self._rawCapture, format="bgr", use_video_port=True)
            image = frame.array
            yield image
            self._rawCapture.truncate(0)
            if False == self._is_running:
                break

    def stop(self):
        self._is_running = False

