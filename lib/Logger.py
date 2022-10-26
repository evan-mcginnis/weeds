
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

    @property
    def directory(self) -> str:
        return(self.rootDirectory)

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

    def logImage(self, name: str, image) -> str:
        pathname = "{}/{}-{:05d}.jpg".format(self.rootDirectory, name, self.sequence)
        filename = "{}-{:05d}.jpg".format(name, self.sequence)

        cv.imwrite(pathname, image)
        self.sequence = self.sequence + 1
        return filename
