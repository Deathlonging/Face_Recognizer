#!/usr/bin/python

from face_detector import FaceDetector
from face_recognizer import FaceRecognizer
from object_tracker import ObjectTracker

class FaceRecognizerNode:
    def __init__(self, input_source, face_detector_cascade_file_path, face_dataset_path, iou_threshold=0.8):
        self._input_source = input_source
        self._iou_threshold = iou_threshold
        self._face_detector = FaceDetector(face_detector_cascade_file_path)
        self._face_recognizer = FaceRecognizer(face_dataset_path)

        self._recognized_faces = []

    def getRecognizedFaces(self):
        image_generator = self._input_source.getImages()
        for image in image_generator:
            # Track already recognized faces
            self.updatedTracking(image)
            self.removeOverlappingTrackers()

            # Try recognize already recognized unknown faces
            unknown_faces = [recognized_face for recognized_face in self._recognized_faces if FaceRecognizer.UNKNOWN_PERSON_NAME == recognized_face['person']]
            if len(unknown_faces) > 0:
                unknown_face_bounding_boxes = [unknown_face['bounding_box'] for unknown_face in unknown_faces]
                persons = self._face_recognizer.recognizeFaces(image,unknown_face_bounding_boxes)
                for unkown_face_counter, unknown_face in enumerate(unknown_faces):
                    unknown_face['person'] = persons[unkown_face_counter]
            if len(self._recognized_faces) > 0:
                print('Tracking {} faces: {}'.format(len(self._recognized_faces), self._recognized_faces))

            # Detect faces
            face_bounding_boxes = self._face_detector.detectFaces(image)

            # Drop already tracked faces
            new_detected_face_bounding_boxes = self.getNewDetectedFaceBoundingBoxes(image, face_bounding_boxes)

            # Recognize new faces
            if len(new_detected_face_bounding_boxes) > 0:
                new_recognized_persons = self._face_recognizer.recognizeFaces(image,new_detected_face_bounding_boxes)
                new_recognized_faces = []
                for i in range(len(new_detected_face_bounding_boxes)):
                    new_recognized_faces.append({
                        'person' : new_recognized_persons[i], 
                        'bounding_box' : new_detected_face_bounding_boxes[i],
                        'tracker' : ObjectTracker(image,new_detected_face_bounding_boxes[i])
                        })
                print('Detected {} new faces: {}'.format(len(new_recognized_faces), new_recognized_faces))
                self._recognized_faces.extend(new_recognized_faces)
            yield image, self._recognized_faces

    def getNewDetectedFaceBoundingBoxes(self, image, detected_face_bounding_boxes):
        new_detected_face_bounding_boxes = []
        for detected_face_bounding_box in detected_face_bounding_boxes:
            is_new = True
            for recognized_face_index, recognized_face in enumerate(self._recognized_faces):
                tracked_face_bounding_box = recognized_face['bounding_box']
                intersection_over_union = self.calculateIntersectionOverUnion(detected_face_bounding_box,tracked_face_bounding_box)
                if intersection_over_union >= self._iou_threshold:
                    self._recognized_faces[recognized_face_index]['bounding_box'] = detected_face_bounding_box
                    self._recognized_faces[recognized_face_index]['tracker'] = ObjectTracker(image, detected_face_bounding_box)
                    is_new = False
                    break
            if True == is_new:
                new_detected_face_bounding_boxes.append(list(detected_face_bounding_box))
        return new_detected_face_bounding_boxes

    def updatedTracking(self,current_image):
        indices_to_delete = []
        for index, recognized_face in enumerate(self._recognized_faces):
            tracker = recognized_face['tracker']
            is_tracking, tracked_bounding_box = tracker.track(current_image)
            if True == is_tracking:
                recognized_face['bounding_box'] = tracked_bounding_box
            else:
                indices_to_delete.append(index)
        for index_to_delete in reversed(indices_to_delete):
            del self._recognized_faces[index_to_delete]

    def removeOverlappingTrackers(self):
        indices_to_delete = []
        for index, recognized_face in enumerate(self._recognized_faces):
            bounding_box = recognized_face['bounding_box']
            for other_index, other_recognized_face in enumerate(self._recognized_faces):
                if index == other_index:
                    continue
                other_bounding_box = other_recognized_face['bounding_box']
                intersection_over_union = self.calculateIntersectionOverUnion(bounding_box,other_bounding_box)
                # Overlap happened
                if intersection_over_union >= self._iou_threshold:
                    person = recognized_face['person']
                    other_person = other_recognized_face['person']
                    # If and only if one tracker tracks recognized person prior that one
                    if (FaceRecognizer.UNKNOWN_PERSON_NAME == person or FaceRecognizer.UNKNOWN_PERSON_NAME == other_person) and not (FaceRecognizer.UNKNOWN_PERSON_NAME == person and FaceRecognizer.UNKNOWN_PERSON_NAME == other_person):
                        if FaceRecognizer.UNKNOWN_PERSON_NAME == person:
                            indices_to_delete.append(index)
                        else:
                            indices_to_delete.append(other_index)
                    # If both track person or unknown take box with bigger area
                    else:
                        box_area = bounding_box[2] * bounding_box[3]
                        other_box_area = bounding_box[2] * bounding_box[3]
                        if box_area > other_box_area:
                            indices_to_delete.append(other_index)
                        else:
                            indices_to_delete.append(index)

        if len(indices_to_delete) > 1:
            print(indices_to_delete)
            print(reversed(list(set(indices_to_delete))))
        for index_to_delete in reversed(list(set(indices_to_delete))):
            del self._recognized_faces[index_to_delete]


# From https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/ but slightly changed

    def calculateIntersectionOverUnion(self, boxA, boxB):
        A_x_min, A_y_min, A_w_old, A_h_old = boxA
        B_x_min, B_y_min, B_w_old, B_h_old = boxB
        A_x_max, A_y_max = A_x_min + A_w_old, A_y_min + A_h_old
        B_x_max, B_y_max = B_x_min + B_w_old, B_y_min + B_h_old
        A_w, A_h, B_w, B_h = A_w_old+1, A_h_old+1, B_w_old+1, B_h_old+1
        # determine the (x, y)-coordinates of the intersection rectangle
        I_x_min = max(A_x_min, B_x_min)
        I_y_min = max(A_y_min, B_y_min)
        I_x_max = min(A_x_max, B_x_max)
        I_y_max = min(A_y_max, B_y_max)
        I_w, I_h = I_x_max - I_x_min + 1, I_y_max - I_y_min + 1

        # compute the area of intersection rectangle
        interArea = max(0, I_w) * max(0, I_h)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = A_w * A_h
        boxBArea = B_w * B_h

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou