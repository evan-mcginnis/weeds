
'''
A simple Program for grabing video from basler camera and converting it to opencv img.
Tested on Basler acA1300-200uc (USB3, linux 64bit , python 3.5)
'''
from pypylon import pylon
import cv2

from Performance import Performance

performace = Performance("basler.csv")
performace.initialize()

# connecting to the first available camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

# Grabning Continusely (video) with minimal delay
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
converter = pylon.ImageFormatConverter()

# converting to opencv bgr format
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

i = 0
while camera.IsGrabbing():
    performace.start()
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if grabResult.GrabSucceeded():
        i  = i + 1
        performace.stopAndRecord("GRAB")
        # Access the image data
        image = converter.Convert(grabResult)
        img = image.GetArray()
        cv2.namedWindow('title', cv2.WINDOW_NORMAL)
        cv2.imshow('title', img)
        print("Image {}\n".format(i))
        k = cv2.waitKey(1)
        if k & 0xFF == ord('q'):
            break
    grabResult.Release()

# Releasing the resource
camera.StopGrabbing()

cv2.destroyAllWindows()

