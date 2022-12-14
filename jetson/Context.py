#
# C O N T E X T
#
# The context in which an image was taken
#

class Context:
    def __init__(self):
        self._speed = 0.0
        self._make = "Basler"
        self._model = "2500"
        self._latitude = 0.0
        self._longitude = 0.0
        self._exposure = 0

    @property
    def exposure(self) -> int:
        return self._exposure

    @exposure.setter
    def exposure(self, theExposure: int):
        self.exposure = theExposure

    @property
    def latitude(self) -> float:
        return self._latitude

    @latitude.setter
    def latitude(self, theLatitude: float):
        self._latitude = theLatitude

    @property
    def longitude(self) -> float:
        return self._longitude

    @longitude.setter
    def longitude(self, theLongitude: float):
        self._longitude = theLongitude

    @property
    def speed(self) -> float:
        return self._speed

    @speed.setter
    def speed(self, theSpeed: float):
        self._speed = theSpeed

    @property
    def make(self) -> str:
        return self._make

    @make.setter
    def make(self, theMake: str):
        self._make = theMake

    @property
    def model(self) -> str:
        return self._make

    @model.setter
    def model(self, theModel: str):
        self._make = theModel
