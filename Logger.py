
#
# L O G G E R
#
import os

class Logger:
    def __init__(self):
        self.rootDirectory = ""
        return

    def connect(self, directoryName: str):
        self.rootDirectory = directoryName
        return True

    def logImage(self, sequence: int):
        return
