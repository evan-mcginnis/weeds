#
# A W S  U P L O A D
#
from time import sleep

import boto3
import uuid
import datetime
import os
import pathlib
import argparse
import glob
import tarfile
import sys
import syslog
import shutil
import constants
import xmpp
import logging
import logging.config
import threading

from OptionsFile import OptionsFile
from MUCCommunicator import MUCCommunicator
from Messages import SystemMessage

now = datetime.datetime.now()
timeStamp = now.strftime('%Y-%m-%d-%H-%M-%S-')

PREFIX = 'yuma-' + timeStamp

PATTERN_IMAGES = '*.jpg'
KEY_IMAGES = 'images'
TAR_IMAGES = 'images.tar'

s3Resource = boto3.resource('s3')

def createBucketName(prefix: str) -> str:
    return ''.join([prefix, str(uuid.uuid4())])

def createBucket(bucket_prefix, s3_connection):
    session = boto3.session.Session()
    current_region = session.region_name
    bucketName = createBucketName(bucket_prefix)
    print("Bucket: {}".format(bucketName))
    bucketResponse = s3_connection.create_bucket(Bucket=bucketName,
                                                 CreateBucketConfiguration={ 'LocationConstraint': current_region})
    print(bucketName, current_region)
    return bucketName, bucketResponse

def findFiles(directory: str):
    flist = []
    # Find all the files in the directory.
    images = "*"
    flist = glob.glob('*')
    return flist

def prepare():
    # Find, add, and delete the images
    images = glob.glob(PATTERN_IMAGES)
    tar = tarfile.open(name=TAR_IMAGES,mode="w|")
    for image in images:
        tar.add(image)
        try:
            os.remove(image)
        except OSError as e:
            print("Could not remove image {}: {}".format(image, e.strerror))
    tar.close()
    return

def upload(bucket : str, options: OptionsFile):
    # Create the target bucket that will hold data from this run
    bucketName, bucketResponse = createBucket(bucket, s3Resource.meta.client)

    first_bucket = s3Resource.Bucket(name=bucketName)

    # Prepare things for S3 upload
    prepare()

    # Find all the files in the directory
    files = glob.glob('*')

    for file in files:
        try:
            key = options.option(constants.PROPERTY_SECTION_KEYS, file)
        except KeyError:
            key = file

        object = s3Resource.Object(bucket_name=bucketName, key=file)
        print("Uploading {}".format(file))
        object.upload_file(Filename=file)

# The callback for messages received in the system room.
# When the total distance is the width of the image, grab an image and process it.
#
def callbackSystem(conn,msg: xmpp.protocol.Message):
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

def startupLogging():
    if not os.path.isfile(arguments.logging):
        print("Unable to access logging configuration file {}".format(arguments.logging))
        sys.exit(1)

    # Initialize logging
    logging.config.fileConfig(arguments.logging)
    log = logging.getLogger("S3")
    return log

#
# This method will never return.  Connect and start processing messages
#
def processMessages(communicator: MUCCommunicator):
    """
    Process messages for the chatroom -- note that this routine will never return.
    :param communicator: The chatroom communicator
    """
    log.info("Connecting to chatroom")
    communicator.connect(True)

parser = argparse.ArgumentParser("AWS Upload")
parser.add_argument("-d", "--directory", action="store", required=True, help="Run data directory")
parser.add_argument("-w", "--watch", action="store_true", required=False, help="Watch this directory for run data")
parser.add_argument("-u", "--upload", action="store_true", required=False, help="Upload this directory and exit")
parser.add_argument("-s", "--s3", action="store", required=False, default="s3.ini", help="S3 INI")
parser.add_argument("-i", "--ini", action="store", required=False, default=constants.PROPERTY_FILENAME, help="INI")
parser.add_argument("-lg", "--logging", action="store", default=constants.PROPERTY_LOGGING_FILENAME, help="Logging configuration file")

arguments = parser.parse_args()

log = startupLogging()

options = OptionsFile(arguments.ini)
options.load()

_connected = os.path.isdir(arguments.directory)
if _connected:
    os.chdir(arguments.directory)
else:
    print("Cannot change to specified directory: {}".format(arguments.directory))
    sys.exit(-1)

# If this is a daemon, connect to the system chatroom
if arguments.watch:
    # The room that will get status reports about this process
    systemRoom = MUCCommunicator(options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_JID_CLOUD),
                                 options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_NICK_CLOUD),
                                 options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_DEFAULT_PASSWORD),
                                 options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_ROOM_SYSTEM),
                                 callbackSystem,
                                 None)
    threads = list()
    log.debug("Starting system receiver")
    sys = threading.Thread(name=constants.THREAD_NAME_SYSTEM, target=processMessages, args=(systemRoom,))
    threads.append(sys)
    sys.start()

if arguments.watch or arguments.upload:
    watching = True
    while watching:
        directories = options.option(constants.PROPERTY_SECTION_GENERAL, constants.PROPERTY_PREFIX) + "*"
        #directories = glob.glob("yuma-*")
        for directory in directories:
            if os.path.isdir(directory):
                os.chdir(directory)
            else:
                syslog.syslog(syslog.LOG_WARNING, "File found in output directory: " + directory)
            timeStamp = now.strftime('%Y-%m-%d-%H-%M-%S-')
            PREFIX = 'yuma-' + timeStamp
            upload(PREFIX, options)

            syslog.syslog(syslog.LOG_INFO,"Created S3 bucket: {}".format(PREFIX))

            # Cleanup
            os.chdir("..")
            try:
                shutil.rmtree(directory)
            except OSError as e:
                syslog.syslog(syslog.LOG_ERR, "Could not remove directory: {}".format(e.strerror))

            # If this is just a single run, exit now
            if arguments.upload:
                watching = False
            # Otherwise sleep a bit and try again.
            else:
                sleep(5 * 60)

# Wait for the workers to finish
for index, thread in enumerate(threads):
    thread.join()


sys.exit(0)