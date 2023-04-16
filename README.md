# Parking Entry System - Object Detection & OCR

This repository contains a Python script for object detection of parked cars at entry points of the parking lot. The script uses OpenCV to capture a video stream from a camera and then uses TensorFlow to detect parked cars and Tesseract OCR to recognize the license plate number. The script then checks whether the license plate is registered or not, using the Firebase Firestore database.

## Files Included

This submodule includes the following files:

- `detect.py`: the main Python script that implements the object detection and  the OCR system.
- `detect.tflite`: the TensorFlow Lite model that was trained for object detection.
- `label.txt`: a file containing the labels used in the object detection model.
- `requirements.txt`: a file containing the required Python libraries to run the object detection system.

## System Requirements

The parking entry system requires the following:

- Raspberry Pi with a 32-bit system, as OpenCV-Python is not compatible with Tesseract OCR in 64-bit systems.
- OpenCV-Python
- TensorFlow Lite
- Tesseract OCR
- Firebase Firestore

## Usage

To use this script, follow these steps:

1-Install the required dependencies:
```
pip3 install opencv-python
sudo apt-get install libcblas-dev
sudo apt-get install libhdf5-dev
sudo apt-get install libhdf5-serial-dev
sudo apt-get install libatlas-base-dev
sudo apt-get install libjasper-dev
sudo apt-get install libqtgui4
sudo apt-get install libqt4-test
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install python3-tflite-runtime
pip3 install pytesseract
```
2-Clone this repository to your local machine.
3-Install the required Python libraries by running the following command: `pip install -r requirements.txt`
4-Configure the `detect.py` script with your Firebase Firestore database credentials.
5-Connect your camera to the system and run the `detect.py` script.


