#
# A P P L I C A T O R
#
# Odometer and Treatment
#
import time
from typing import Callable

from MUCCommunicator import MUCCommunicator

import logging
import logging.config
from getpass import getpass
from argparse import ArgumentParser
import dns.resolver
import yaml
import os
import sys
import threading
import asyncio

import constants

messageNumber = 0

def processMessage(msg):
    global messageNumber
    print("Applicator::Processing message {}".format(messageNumber))
    messageNumber += 1
    return

def applyTreatment():
    print("Apply 1cm treatment")

def startupCommunication(muc: MUCCommunicator):
    log.info("Startup communication handler")

    muc.send_message(mto=constants.ROOM_ODOMETRY,
                        mbody="applicator::startupCommunication",
                        mtype='groupchat')
    muc.process()

def startupProcessing(muc: MUCCommunicator, treatment: Callable):
    """
    Start up the treatment system.  This will call the treatment callback at every interval.

    :param virtualOdometer:
    :param muc:
    :param treatment: Applies treatment as in the current plan
    """

    # Run diagnostics on the odometer before we begin.
    # diagnosticResult, diagnosticText = odometer.diagnostics()

    # if not diagnosticResult:
    #     print(diagnosticText)
    #     sys.exit(1)

    #muc.registerMessageHandler(applyTreatment)
    #odometer.registerMUC(muc)
    #muc.connect()
    #muc.process()
    while True:
        time.sleep(5)


# Setup the command line arguments.
parser = ArgumentParser()

# Output verbosity options.
# parser.add_argument("-q", "--quiet", help="set logging to ERROR",
#                     action="store_const", dest="loglevel",
#                     const=logging.ERROR, default=logging.INFO)
# parser.add_argument("-de", "--debug", help="set logging to DEBUG",
#                     action="store_const", dest="loglevel",
#                     const=logging.DEBUG, default=logging.INFO)

parser.add_argument('-d', '--dns', action="store", required=True, help="DNS server address")

# JID and password options.
parser.add_argument("-j", "--jid", dest="jid", help="JID to use")
parser.add_argument("-p", "--password", dest="password", help="password to use")
parser.add_argument("-r", "--room", dest="room", help="MUC room to join")
parser.add_argument("-n", "--nick", dest="nick", help="MUC nickname")

parser.add_argument("-lg", "--logging", action="store", default="info-logging.yaml", help="Logging configuration file")

arguments = parser.parse_args()

#
# D N S
#

if arguments.dns is not None:
    # Force resolutions to come from a server that has the entries we want
    my_resolver = dns.resolver.Resolver()
    my_resolver.nameservers = [arguments.dns]

#
# L O G G I N G
#
# Confirm the YAML file exists
if not os.path.isfile(arguments.logging):
    print("Unable to access logging configuration file {}".format(arguments.logging))
    sys.exit(1)

# Initialize logging
with open(arguments.logging, "rt") as f:
    config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)
log = logging.getLogger(__name__)

#
# M U C
#
# Set up so we can connect to the channel for odometry
log.info("Creating MUC: {0}".format(constants.ROOM_ODOMETRY))
chatOdometry = MUCCommunicator(arguments.jid,
                               arguments.password,
                               constants.ROOM_ODOMETRY,
                               constants.NICK_JETSON,
                               processMessage)

# I have absolutely no idea why this can't be done in an object method, but I've tried it
# and could not get it to work
log.info("Registering plugins for MUC: {0}".format(constants.ROOM_ODOMETRY))
chatOdometry.register_plugin('xep_0030')  # Service Discovery
chatOdometry.register_plugin('xep_0045')  # Multi-User Chat
chatOdometry.register_plugin('xep_0199')  # XMPP Ping

log.info("Connect and process startup for MUC: {0}".format(constants.ROOM_ODOMETRY))
chatOdometry.connect()
chatOdometry.startup(5)
# At this point, we care connected to the chat room, but we are NOT processing messages

chatTreatment = MUCCommunicator(constants.JID_RIO,
                                arguments.password,
                                constants.ROOM_TREATMENT,
                                constants.NICK_TREATMENT,
                                processMessage)

# I have absolutely no idea why this can't be done in an object method, but I've tried it
# and could not get it to work
chatTreatment.register_plugin('xep_0030')  # Service Discovery
chatTreatment.register_plugin('xep_0045')  # Multi-User Chat
chatTreatment.register_plugin('xep_0199')  # XMPP Ping

chatTreatment.connect()
chatTreatment.startup(5)

# Keep track of the threads we create
threads = list()


# The communication thread
communicator = threading.Thread(target=startupCommunication, args=(chatOdometry,))
threads.append(communicator)
communicator.start()

# The processing thread
processing = threading.Thread(target=startupProcessing, args=(chatOdometry, applyTreatment))
threads.append(processing)
processing.start()

for index, thread in enumerate(threads):
    logging.info("Main    : before joining thread %d.", index)
    thread.join()
    logging.info("Main    : thread %d done", index)

# Start the odometer
#startupOdometer(True, chatOdometry, applyTreatment)



