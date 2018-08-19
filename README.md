# Face Recognizer
This simple face detector and recognizer is a client for openHAB designed for the Raspberry Pi. The face recognition funtionalilty was derived from https://www.pyimagesearch.com/2018/06/25/raspberry-pi-face-recognition/ Also object tracking was which was derived from https://www.learnopencv.com/object-tracking-using-opencv-cpp-python/ . The haarcascade_frontalface.xml was copied from https://github.com/opencv/opencv/tree/master/data/haarcascades .

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