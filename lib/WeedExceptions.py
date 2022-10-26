#
# W E E D E X C E P T I O N S
#
class WeedExceptions(Exception):
    def __init__(self, message):
        self._message = message
        super().__init__(self._message)

    def __str__(self):
        return f'{self._message}'

class DepthUnknownStream(WeedExceptions):
    def __init__(self, message):
        super().__init__(self._message)

class XMPPServerUnreachable(WeedExceptions):
    def __init__(self, message):
        super().__init__(self._message)

class XMPPServerAuthFailure(WeedExceptions):
    def __init__(self, message):
        super().__init__(self._message)