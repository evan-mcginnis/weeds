# Grab_Strategies.cpp
# ===============================================================================
# #    This sample shows the use of the Instant Camera grab strategies.
# ===============================================================================

import sys
import time

from pypylon import pylon

# Contains an Image Event Handler that prints a message for each event method call.

# BEGIN COPIED
# Contains a Configuration Event Handler that prints a message for each event method call.

from pypylon import pylon


class ConfigurationEventPrinter(pylon.ConfigurationEventHandler):
    def OnAttach(self, camera):
        print("OnAttach event")

    def OnAttached(self, camera):
        print("OnAttached event for device ", camera.GetDeviceInfo().GetModelName())

    def OnOpen(self, camera):
        print("OnOpen event for device ", camera.GetDeviceInfo().GetModelName())

    def OnOpened(self, camera):
        print("OnOpened event for device ", camera.GetDeviceInfo().GetModelName())

    def OnGrabStart(self, camera):
        print("OnGrabStart event for device ", camera.GetDeviceInfo().GetModelName())

    def OnGrabStarted(self, camera):
        print("OnGrabStarted event for device ", camera.GetDeviceInfo().GetModelName())

    def OnGrabStop(self, camera):
        print("OnGrabStop event for device ", camera.GetDeviceInfo().GetModelName())

    def OnGrabStopped(self, camera):
        print("OnGrabStopped event for device ", camera.GetDeviceInfo().GetModelName())

    def OnClose(self, camera):
        print("OnClose event for device ", camera.GetDeviceInfo().GetModelName())

    def OnClosed(self, camera):
        print("OnClosed event for device ", camera.GetDeviceInfo().GetModelName())

    def OnDestroy(self, camera):
        print("OnDestroy event for device ", camera.GetDeviceInfo().GetModelName())

    def OnDestroyed(self, camera):
        print("OnDestroyed event")

    def OnDetach(self, camera):
        print("OnDetach event for device ", camera.GetDeviceInfo().GetModelName())

    def OnDetached(self, camera):
        print("OnDetached event for device ", camera.GetDeviceInfo().GetModelName())

    def OnGrabError(self, camera, errorMessage):
        print("OnGrabError event for device ", camera.GetDeviceInfo().GetModelName())
        print("Error Message: ", errorMessage)

    def OnCameraDeviceRemoved(self, camera):
        print("OnCameraDeviceRemoved event for device ", camera.GetDeviceInfo().GetModelName())
# END COPIED

# BEGIN COPIED
#from pypylon import pylon


class ImageEventPrinter(pylon.ImageEventHandler):
    def OnImagesSkipped(self, camera, countOfSkippedImages):
        print("OnImagesSkipped event for device ", camera.GetDeviceInfo().GetModelName())
        print(countOfSkippedImages, " images have been skipped.")
        print()

    def OnImageGrabbed(self, camera, grabResult):
        print("OnImageGrabbed event for device ", camera.GetDeviceInfo().GetModelName())

        # Image grabbed successfully?
        if grabResult.GrabSucceeded():
            print("SizeX: ", grabResult.GetWidth())
            print("SizeY: ", grabResult.GetHeight())
            img = grabResult.GetArray()
            print("Gray values of first row: ", img[0])
            print()
        else:
            print("Error: ", grabResult.GetErrorCode(), grabResult.GetErrorDescription())
# END COPIED

#from samples.imageeventprinter import ImageEventPrinter
#from samples.configurationeventprinter import ConfigurationEventPrinter

# Begin basler sample code

# Grab_UsingGrabLoopThread.cpp
# This sample illustrates how to grab and process images using the grab loop thread
# provided by the Instant Camera class.

from pypylon import genicam
from pypylon import pylon


import time


def getkey():
    return input("Enter \"t\" to trigger the camera or \"e\" to exit and press enter? (t/e) ")


# Example of an image event handler.
class SampleImageEventHandler(pylon.ImageEventHandler):
    def OnImageGrabbed(self, camera, grabResult):
        print("CSampleImageEventHandler::OnImageGrabbed called.")
        print()
        print()


if __name__ == '__main__':
    try:
        # Create an instant camera object for the camera device found first.
        camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

        # Register the standard configuration event handler for enabling software triggering.
        # The software trigger configuration handler replaces the default configuration
        # as all currently registered configuration handlers are removed by setting the registration mode to RegistrationMode_ReplaceAll.
        camera.RegisterConfiguration(pylon.SoftwareTriggerConfiguration(), pylon.RegistrationMode_ReplaceAll,
                                     pylon.Cleanup_Delete)

        # For demonstration purposes only, add a sample configuration event handler to print out information
        # about camera use.t
        camera.RegisterConfiguration(ConfigurationEventPrinter(), pylon.RegistrationMode_Append, pylon.Cleanup_Delete)

        # The image event printer serves as sample image processing.
        # When using the grab loop thread provided by the Instant Camera object, an image event handler processing the grab
        # results must be created and registered.
        camera.RegisterImageEventHandler(ImageEventPrinter(), pylon.RegistrationMode_Append, pylon.Cleanup_Delete)

        # For demonstration purposes only, register another image event handler.
        camera.RegisterImageEventHandler(SampleImageEventHandler(), pylon.RegistrationMode_Append, pylon.Cleanup_Delete)

        # Start the grabbing using the grab loop thread, by setting the grabLoopType parameter
        # to GrabLoop_ProvidedByInstantCamera. The grab results are delivered to the image event handlers.
        # The GrabStrategy_OneByOne default grab strategy is used.
        camera.StartGrabbing(pylon.GrabStrategy_OneByOne, pylon.GrabLoop_ProvidedByInstantCamera)

        # Wait for user input to trigger the camera or exit the program.
        # The grabbing is stopped, the device is closed and destroyed automatically when the camera object goes out of scope.
        while True:
            time.sleep(0.05)
            key = getkey()
            print(key)
            if (key == 't' or key == 'T'):
                # Execute the software trigger. Wait up to 100 ms for the camera to be ready for trigger.
                if camera.WaitForFrameTriggerReady(100, pylon.TimeoutHandling_ThrowException):
                    camera.ExecuteSoftwareTrigger();
            if (key == 'e') or (key == 'E'):
                break
    except genicam.GenericException as e:
        # Error handling.
        print("An exception occurred.", e.GetDescription())

# End basler sample code
