#
# W E E D E R
#
from statemachine import StateMachine, State

class WeederState(StateMachine):
    new = State('New', initial=True)
    capturing = State('Capturing')
    claimed = State('Claimed')
    idle = State('Idle')
    failed = State('Failed')
    missing = State('Missing')

    toIdle = new.to(idle)
    toCapture = claimed.to(capturing)
    toStop = capturing.to(idle)
    toClaim = idle.to(claimed)
    toFailed = claimed.to(failed)
    toMissing = new.to(missing)