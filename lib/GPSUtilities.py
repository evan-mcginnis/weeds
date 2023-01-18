#
# G P S U T I L I T I E S
#

import math

class GPSUtilities:

    @classmethod
    def decdeg2dms(self, degs: float) -> ():
        neg = degs < 0
        degs = (-1) ** neg * degs
        degs, d_int = math.modf(degs)
        mins, m_int = math.modf(60 * degs)
        secs = 60 * mins
        return neg, d_int, m_int, secs

    @classmethod
    def dms2decdeg(self, dms: (), direction: str) -> float:
        degrees, minutes, seconds = dms
        dd = float(degrees) + float(minutes) / 60 + float(seconds) / (60 * 60);
        if direction == 'E' or direction == 'S':
            dd *= -1
        return dd;