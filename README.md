# Face Recognizer
This simple face detector and recognizer is a client for openHAB designed for the Raspberry Pi. The face recognition functionalilty was derived from https://www.pyimagesearch.com/2018/06/25/raspberry-pi-face-recognition/ Also object tracking was which was derived from https://www.learnopencv.com/object-tracking-using-opencv-cpp-python/ . The haarcascade_frontalface.xml was copied from https://github.com/opencv/opencv/tree/master/data/haarcascades .

Warning: Not yet tested with mqtt and openHAB!

## Test
This Face Recognizer was tested under/with:
Ubuntu 16.04
python 3.6.6
opencv 3.4.1

## Usage
```
python scripts/run_face_recognition.py --cascade=haarcascade_frontalface.xml --face_dataset_path=dataset --input=webcam --output=imshow
```
## Todo
* Face Recognition:
  * Implement function to create dataset from current input source for specific person
  * Implement event handler to start create dataset from current input source for specific person
  * Implement automatic face encodings creation for known persons from current input source
  * Implement face encodings dataset reduction to unique encodings
* Tests:
  * Test implementation for file_reader as inpout source on raspberry pi
  * Test implementation for pi_cam as input source on raspberry pi
  * Test implementation for broadcast as output on raspberry pi with console mqtt server
  * Test implementation for broadcast as output on raspberry pi with openHAB
