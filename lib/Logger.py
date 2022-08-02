
#
# L O G G E R
#
import os
import cv2 as cv

class Logger:
    def __init__(self):
        self.rootDirectory = ""
        self.sequence = 0
        return

    def connect(self, directoryName: str):
        try:
            os.makedirs(directoryName, exist_ok=True)
        except Exception as e:
            print("Cannot create directory: {}".format(directoryName))

        if os.path.isdir(directoryName):
            self.rootDirectory = directoryName
            return True
        else:
            return False

    def logImage(self, name: str, image):
        cv.imwrite(self.rootDirectory + "/" + name + "-" + str(self.sequence) + ".jpg", image)
        self.sequence = self.sequence + 1
        return
