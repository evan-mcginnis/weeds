#
# E M I T T E R  P L A T E
#
# Various attributes of an emitter plate

import constants
from OptionsFile import OptionsFile

class EmitterPlate:

    def __init__(self, options: OptionsFile):
        """
        The dimensions and offsets for an emitter plate
        :param options: An options file with a section describing the plate
        """
        try:
            self._x_offset = float(options.option(constants.PROPERTY_SECTION_EMITTER, constants.PROPERTY_OFFSET_X))
            self._y_offset = float(options.option(constants.PROPERTY_SECTION_EMITTER, constants.PROPERTY_OFFSET_Y))
            self._distanceBetweenTiers = float(options.option(constants.PROPERTY_SECTION_EMITTER, constants.PROPERTY_DISTANCE_TIER))
            self._distanceBetweenEmitters = float(options.option(constants.PROPERTY_SECTION_EMITTER, constants.PROPERTY_DISTANCE_EMITTERS))
        except KeyError as key:
            raise ValueError(key)

        self._tiersPerPlate = 4
        self._emittersPerTier = 3

    @property
    def xOffset(self) -> float:
        """
        The X offset of the emitter plate the distance from the center of the camera to the center of the emitter
        in the leading edge row
        :return: the offset in mm
        """
        return self._x_offset

    @property
    def yOffset(self) -> float:
        """
        The y offset of the center emitter in the leading edge from the y edge of the image
        :return: the offset in mm
        """
        return self._y_offset

    @property
    def distanceBetweenTiers(self) -> float:
        """
        The distance between adjacent tiers
        :return: the distance in mm
        """
        return self._distanceBetweenTiers

    @property
    def distanceBetweenEmitters(self) -> float:
        """
        The distance between the centers of adjacent emitters
        :return: the distance in mm
        """
        return self._distanceBetweenEmitters

    @property
    def numberOfTiers(self) -> int:
        """
        The number of tiers of emitters in a plate
        :return:
        """
        return self._tiersPerPlate

    @property
    def emittersPerTier(self) -> int:
        """
        The number of emitters in a single tier
        :return:
        """
        return self._emittersPerTier
