#
#
# National Instruments
#
#

from constants import SubsystemType

class NationalInstruments:
    def __init__(self, type: SubsystemType):
        self._type = type
        return

    @property
    def type(self) -> SubsystemType:
        return self._type

class VirtualNationalInstruments(NationalInstruments):
    def __init__(self):
        super().__init__(SubsystemType.VIRTUAL)

class PhysicalNationalInstruments(NationalInstruments):
    def __init__(self):
        self._system = None
        super().__init__(SubsystemType.PHYSICAL)

    @property
    def system(self):
        return self._system

    @system.setter
    def system(self, theSystem):
        self._system = theSystem


