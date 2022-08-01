#
# C O N S O L E
#
#
from time import sleep
from typing import Callable

from MUCCommunicator import MUCCommunicator
from OptionsFile import OptionsFile

import logging
import logging.config
from argparse import ArgumentParser
# import dns.resolver
import xmpp
import logging
import threading
import constants
import uuid

from Messages import SystemMessage

def messageSystemCB(conn,msg: xmpp.protocol.Message):
    # Make sure this is a message sent to the room, not directly to us
    if msg.getType() == "groupchat":
            body = msg.getBody()
            # Check if this is a real message and not just an empty keep-alive message
            if body is not None:
                log.debug("system message from {}: [{}]".format(msg.getFrom(), msg.getBody()))
    elif msg.getType() == "chat":
            print("private: " + str(msg.getFrom()) +  ":" +str(msg.getBody()))
    else:
        log.error("Unknown message type {}".format(msg.getType()))

def processMessages(communicator: MUCCommunicator):
    """
    Process messages for the chatroom -- note that this routine will never return.
    :param communicator: The chatroom communicator
    """
    log.info("Connecting to chatroom")
    communicator.connect(True)

# Setup the command line arguments.
parser = ArgumentParser()

# Output verbosity options.
# parser.add_argument("-q", "--quiet", help="set logging to ERROR",
#                     action="store_const", dest="loglevel",
#                     const=logging.ERROR, default=logging.INFO)
# parser.add_argument("-de", "--debug", help="set logging to DEBUG",
#                     action="store_const", dest="loglevel",
#                     const=logging.DEBUG, default=logging.INFO)

parser.add_argument("-c", "--cmd",action="store", required=True, help="Command")
parser.add_argument('-d', '--dns', action="store", required=False, help="DNS server address")
parser.add_argument("-i", "--ini", action="store", default=constants.PROPERTY_FILENAME, help="options configuration file")
parser.add_argument("-lg", "--logging", action="store", default=constants.PROPERTY_LOGGING_FILENAME, help="Logging configuration file")
parser.add_argument("-w", "--wait", action="store", required=False, default=False, help="Wait for messages in system room")
arguments = parser.parse_args()

logging.config.fileConfig(arguments.logging)
log = logging.getLogger("control")

#
# D N S
#

# if arguments.dns is not None:
#     # Force resolutions to come from a server that has the entries we want
#     my_resolver = dns.resolver.Resolver()
#     my_resolver.nameservers = [arguments.dns]

# Keep track of the threads we create
threads = list()

options = OptionsFile(arguments.ini)
options.load()

systemMessage = SystemMessage()

if arguments.cmd == constants.ACTION_START:
    systemMessage.action = constants.Action.START
    systemMessage.name = str(uuid.uuid4())
else:
    print("Unknown operation: ".format(arguments.cmd))

systemRoom = MUCCommunicator(options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_JID_CONTROL),
                             options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_NICK_CONTROL),
                             options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_DEFAULT_PASSWORD),
                             options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_ROOM_SYSTEM),
                             messageSystemCB)

log.debug("Starting system receiver")
sys = threading.Thread(name=constants.THREAD_NAME_SYSTEM, target=processMessages, args=(systemRoom,))
threads.append(sys)
sys.start()

while not systemRoom.connected:
    log.debug("Waiting for room connection.")
    sleep(5)

systemRoom.sendMessage(systemMessage.formMessage())

for index, thread in enumerate(threads):
    logging.info("Main    : before joining thread %d.", index)
    thread.join()
    logging.info("Main    : thread %d done", index)




