#!/usr/bin/python

import argparse

from face_recognizer_node import FaceRecognizerNode

def main_routine(input_source):
    face_recognizer_node = FaceRecognizerNode(input_source, args['cascade'], args['face_dataset_path'])
    recognizedFacesGenerator = face_recognizer_node.getRecognizedFaces()
    for image, detected_faces in recognizedFacesGenerator:
        yield image, detected_faces

PICAM_ARG_STR = 'picam'
FILE_READER_ARG_STR = 'file_reader'
WEBCAM_ARG_STR = 'webcam'

def picam():
    from pi_camera_reader import PiCameraReader
    input_source = PiCameraReader()
    return input_source

def file_reader():
    from image_file_reader import ImageFileReader
    input_source = ImageFileReader(args['dataset_path'])
    return input_source


def webcam():
    from webcam_reader import WebcamReader
    input_source = WebcamReader()
    return input_source


INPUT_SOURCE_FUNCTION_LOOKUP = {
        PICAM_ARG_STR : picam,
        FILE_READER_ARG_STR : file_reader, 
        WEBCAM_ARG_STR : webcam
    }

PRINTING_FUNCTION_ARG_STR ='print'
SHOW_IMAGE_ARG_STR = 'imshow'
BROADCAST_ARG_STR = 'broadcast'

def print_output(result_generator):
    for image_counter, (image, detected_faces) in enumerate(result_generator):
        print('Image[{}]: {} Persons: {}'.format(image_counter,image.shape, detected_faces))

def show_image_output(result_generator):
    import cv2
    for image, detected_faces in result_generator:
        for detected_face in detected_faces:
            left, top, width, height = detected_face['bounding_box']
            person_name = detected_face['person']
            # draw the predicted face name on the image
            right, bottom = left + width, top + height
            cv2.rectangle(image, (left, top), (right, bottom),
                (0, 255, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(image, person_name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 255, 0), 2)
 
        # display the image to our screen
        cv2.imshow("Frame", image)
        key = cv2.waitKey(1) & 0xFF

def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))

def broadcast_output(result_generator):
    import paho.mqtt.client as mqtt

    client = mqtt.Client("FaceRecognizer")
    client.on_connect = on_connect

    client.connect(args['hostname'], args['port'], 60)

    client.loop_start()

    for _, detected_faces in result_generator:
        for detected_face in detected_faces:
            person = detected_face['person']
            client.publish("face_recognition/detected_person", person)

OUTPUT_SOURCES_FUNCTION_LOOKUP = {
        BROADCAST_ARG_STR : broadcast_output, 
        PRINTING_FUNCTION_ARG_STR : print_output, 
        SHOW_IMAGE_ARG_STR : show_image_output
    }

if '__main__' == __name__:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--cascade", required=True, help = "path to where the face cascade resides")
    ap.add_argument("-f", "--face_dataset_path", required=True, help = "path to where the to recognized face images reside.")
    ap.add_argument("-i", "--input", help = "choose input source: {}".format([key for key in INPUT_SOURCE_FUNCTION_LOOKUP]), default=PICAM_ARG_STR)
    ap.add_argument("-o", "--output", help = "choose output source: {}".format([key for key in OUTPUT_SOURCES_FUNCTION_LOOKUP]), default=BROADCAST_ARG_STR)
    ap.add_argument("-host", "--hostname", help = "hostname for mqtt connection", default="localhost")
    ap.add_argument("-p", "--port", help = "port for mqtt connection", default=1883)
    ap.add_argument("-d", "--test_image_dataset_path", help = "path to where the test dataset of face images resides. Only necessary if input {} is chosen".format(FILE_READER_ARG_STR))
    args = vars(ap.parse_args())
    assert args['input'] in INPUT_SOURCE_FUNCTION_LOOKUP
    assert args['output'] in OUTPUT_SOURCES_FUNCTION_LOOKUP
    input_source = INPUT_SOURCE_FUNCTION_LOOKUP[args['input']]()
    result_generator = main_routine(input_source)
    OUTPUT_SOURCES_FUNCTION_LOOKUP[args['output']](result_generator)


