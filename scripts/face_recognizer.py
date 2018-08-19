#!/usr/bin/python

import face_recognition
import cv2
import os
import operator
import pickle

from face_image_dataset_reader import FaceImageDatasetReader

class FaceRecognizer:
    UNKNOWN_PERSON_NAME = 'Unknown'
    DATASET_FILENAME = 'FaceEncodingsDataset.pickle'

    def __init__(self, face_dataset_path, detection_method='hog'):
        assert True == os.path.exists(face_dataset_path)
        self._detection_method = detection_method
        self._face_dataset = self.initDataset(face_dataset_path)

    def calculatePersonFace(self, person_name, person_face_images):
        person_face_encodings_list = []
        for person_face_image_counter, person_face_image in enumerate(person_face_images):
            print('Calculate face encodings for {} [{}]'.format(person_name, person_face_image_counter))
            person_face_encodings = self.calculateFaceEncodings(person_face_image)
            person_face_encodings_list.extend(person_face_encodings)
        return person_face_encodings_list

    def calculateFaceEncodings(self, face_image, face_bounding_boxes=[]):
        rgb_face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        if 0 == len(face_bounding_boxes):
            face_bounding_boxes = face_recognition.face_locations(rgb_face_image,model=self._detection_method)
        face_encodings = face_recognition.face_encodings(rgb_face_image, face_bounding_boxes)
        return face_encodings

    def initDataset(self, face_dataset_path):
        face_dataset_file_path = os.path.join(face_dataset_path,FaceRecognizer.DATASET_FILENAME)
        face_image_dataset_reader = FaceImageDatasetReader(face_dataset_path)
        face_image_names = face_image_dataset_reader.getImageNames()
        face_dataset_exists = os.path.exists(face_dataset_file_path)
        if True == face_dataset_exists:
            print('Load Face dataset {}'.format(face_dataset_file_path))
            face_dataset_data = self.loadDataset(face_dataset_file_path)
            face_dataset = face_dataset_data['face_dataset']
        if False == face_dataset_exists or False == self.isDatasetFileUpToDate(face_dataset_data, face_image_names):
            if False == face_dataset_exists:
                print('Face dataset {} is not existing. Creating new one'.format(face_dataset_file_path))
            else:
                print('Face dataset {} is not up to date. Creating new one'.format(face_dataset_file_path))
            face_dataset = self.createDataset(face_image_dataset_reader)
            self.saveDataset(face_dataset, face_image_names, face_dataset_file_path)
        print('Use Face dataset with {} persons'.format([person_name for person_name in face_dataset]))
        return face_dataset

    def loadDataset(self, face_dataset_file_path):
        face_dataset_data = None
        with open(face_dataset_file_path, "rb") as face_dataset_file:
            face_dataset_data = pickle.loads(face_dataset_file.read())
        assert face_dataset_data is not None
        return face_dataset_data

    def isDatasetFileUpToDate(self, face_dataset_data, face_image_names):
        return set(face_dataset_data['face_image_names']) == set(face_image_names)

    def createDataset(self, face_image_dataset_reader):
        face_dataset = {}
        persons = face_image_dataset_reader.getPersons()
        for person in persons:
            person_face_images = face_image_dataset_reader.getPersonFaceImages(person)
            face_dataset[person] = self.calculatePersonFace(person,person_face_images)
        return face_dataset
        
    def saveDataset(self, face_dataset, face_image_names, face_dataset_file_path):
        face_dataset_data = {}
        face_dataset_data['face_dataset'] = face_dataset
        face_dataset_data['face_image_names'] = face_image_names
        with open(face_dataset_file_path, "wb") as face_dataset_file:
            face_dataset_file.write(pickle.dumps(face_dataset_data))


    def recognizeFaces(self, faces_image, face_bounding_boxes):
        face_bounding_boxes_top_right_bottom_left = [(y, x+w, y+h, x) for (x, y, w, h) in face_bounding_boxes]
        face_encodings = self.calculateFaceEncodings(faces_image, face_bounding_boxes)
        match_names = []
        for face_encoding in face_encodings:
            match_counter = {}
            for person in self._face_dataset:
                matches = face_recognition.compare_faces(self._face_dataset[person], face_encoding)
                match_counter[person] = sum(1 for match in matches if match == True)
            max_match_name = max(match_counter.items(), key=operator.itemgetter(1))[0]
            if match_counter[max_match_name] > 0:
                match_names.append(max_match_name)
            else:
                match_names.append(FaceRecognizer.UNKNOWN_PERSON_NAME)
        return match_names