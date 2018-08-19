#!/usr/bin/python
import os
from image_file_reader import ImageFileReader

class FaceImageDatasetReader(ImageFileReader):

    def getPersons(self):
        person_subdirectories = next(os.walk(self._image_directory))[1]
        return person_subdirectories

    def getPersonFaceImages(self, person):
        person_subdirectory = os.path.join(self._image_directory, person)
        assert True == os.path.exists(person_subdirectory)
        person_face_image_reader = ImageFileReader(person_subdirectory)
        person_face_image_generator = person_face_image_reader.getImages()
        for person_face_image in person_face_image_generator:
            yield person_face_image

    def getImages(self):
        persons = self.getPersons()
        for person in persons:
            person_face_image_generator = getPersonFaceImages(person)
            for person_face_image in person_face_image_generator:
                yield person_face_image