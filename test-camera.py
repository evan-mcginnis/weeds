import argparse
import numpy
import matplotlib.pyplot as plt
import logging
import logging.config
import yaml
import os
import sys
import cv2 as cv


from CameraFile import CameraBasler

def testBasler():
    camera = CameraBasler("")

    camera.initialize()
    camera.start()
    image = camera.capture()
    cv.namedWindow('title', cv.WINDOW_NORMAL)
    cv.imshow('title', image)
    k = cv.waitKey(1)
    if k & 0xFF == ord('q'):
        camera.disconnect()
        sys.exit(0)

def list_ports():
    """
    Test the ports and returns a tuple with the available ports and the ones that are working.
    """
    is_working = True
    dev_port = 0
    working_ports = []
    available_ports = []
    while is_working:
        #camera = cv.VideoCapture(dev_port, cv.CAP_DSHOW)
        camera = cv.VideoCapture(dev_port)
        if not camera.isOpened():
            is_working = False
            print("Port %s is not working." %dev_port)
        else:
            is_reading, img = camera.read()
            w = camera.get(3)
            h = camera.get(4)
            if is_reading:
                print("Port %s is working and reads images (%s x %s)" %(dev_port,h,w))
                working_ports.append(dev_port)
                cv.imwrite("test-{}.jpg".format(dev_port), img)
            else:
                print("Port %s for camera ( %s x %s) is present but does not reads." %(dev_port,h,w))
                available_ports.append(dev_port)
        dev_port +=1
    return available_ports,working_ports

list_ports()


# from pypylon import pylon
# import cv2
#
# from Performance import Performance
#
# performace = Performance("camera.csv")
# performace.initialize()
#
# # conecting to the first available camera
# camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
# print("Using device ", camera.GetDeviceInfo().GetModelName())
# ("Details", camera.GetDeviceInfo())
# details = camera.GetDeviceInfo()
# # Grabing Continusely (video) with minimal delay
# camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
# converter = pylon.ImageFormatConverter()
#
# # converting to opencv bgr format
# converter.OutputPixelFormat = pylon.PixelType_BGR8packed
# converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
#
# i = 0
# while camera.IsGrabbing():
#     performace.start()
#     grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
#
#     if grabResult.GrabSucceeded():
#         i  = i + 1
#         performace.stopAndRecord("GRAB")
#         # Access the image data
#         image = converter.Convert(grabResult)
#         img = image.GetArray()
#         cv2.namedWindow('title', cv2.WINDOW_NORMAL)
#         cv2.imshow('title', img)
#         print("Image {}\n".format(i))
#         k = cv2.waitKey(1)
#         if k & 0xFF == ord('q'):
#             break
#     grabResult.Release()
#
# # Releasing the resource
# camera.StopGrabbing()
#
# cv2.destroyAllWindows()
#

# import cv2 as cv
#
# camera = cv.VideoCapture(1)
# ret, frame = camera.read()
# camera.release()
# cv.imwrite("camera.jpg", frame)


