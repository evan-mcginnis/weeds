#
# W E E D E X C E P T I O N S
#
class WeedExceptions(Exception):
    pass

class XMPPServerUnreachable(Exception):
    def __init__(self, message):
        self._message = message
        super().__init__(self._message)

    def __str__(self):
        return f'{self._message}'

class XMPPServerAuthFailure(WeedExceptions):
    pass