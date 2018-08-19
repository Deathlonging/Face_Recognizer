#!/usr/bin/python
import cv2
import os

from image_source import ImageSource

class ImageFileReader(ImageSource):
    def __init__(self, image_directory):
        assert True == os.path.exists(image_directory)
        self._image_directory = image_directory

    def getImageNames(self):
        files = next(os.walk(self._image_directory))[2]
        image_files = [file for file in files if self.isImage(file)]
        return image_files

    def getImages(self):
        files = next(os.walk(self._image_directory))[2]
        for file in files:
            if False == self.isImage(file):
                continue
            image_file = os.path.join(self._image_directory,file)
            image = cv2.imread(image_file)
            yield image

    def isImage(self, file):
        file_ending = file.split('.')[-1]
        file_ending = file_ending.lower()
        for image_ending in ['png', 'jpg', 'jpeg']:
            if file_ending == image_ending:
                return True
        return False
