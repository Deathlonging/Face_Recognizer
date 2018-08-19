#!/usr/bin/python

import cv2

class ObjectTracker:
    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    def __init__(self, object_frame, object_bounding_box, tracker_type='MOSSE',):
        assert tracker_type in ObjectTracker.tracker_types
        self._tracker_type = tracker_type
        self._tracker = self.createTracker(tracker_type)
        self._tracker.init(object_frame, tuple(object_bounding_box))

    def createTracker(self, tracker_type):
        (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
        if int(minor_ver) < 3:
            tracker = cv2.Tracker_create(tracker_type)
        else:
            if tracker_type == 'BOOSTING':
                tracker = cv2.TrackerBoosting_create()
            if tracker_type == 'MIL':
                tracker = cv2.TrackerMIL_create()
            if tracker_type == 'KCF':
                tracker = cv2.TrackerKCF_create()
            if tracker_type == 'TLD':
                tracker = cv2.TrackerTLD_create()
            if tracker_type == 'MEDIANFLOW':
                tracker = cv2.TrackerMedianFlow_create()
            if tracker_type == 'GOTURN':
                tracker = cv2.TrackerGOTURN_create()
            if tracker_type == 'MOSSE':
                tracker = cv2.TrackerMOSSE_create()
            if tracker_type == "CSRT":
                tracker = cv2.TrackerCSRT_create()
        return tracker

    def track(self, object_frame):
        is_tracking, object_bounding_box = self._tracker.update(object_frame)
        object_bounding_box = [int(coord) for coord in object_bounding_box]
        return is_tracking, object_bounding_box

    def __str__(self):
        return '{}[{}]'.format(self.__class__.__name__,self._tracker_type)