#
# A W S  U P L O A D
#
import time
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

import botocore.exceptions
from botocore.config import Config

import constants
import xmpp
import logging
import logging.config
import threading
import dns.resolver

from OptionsFile import OptionsFile
from MUCCommunicator import MUCCommunicator

# TODO: This causes problems on the left jetson.  Not sure why
# This does not show up up the program os run from the command line, but only if it is a service

#from Messages import SystemMessage

now = datetime.datetime.now()
timeStamp = now.strftime('%Y-%m-%d-%H-%M-%S-')

PREFIX = 'yuma-' + timeStamp

PATTERN_IMAGES = '*' + constants.EXTENSION_IMAGE
KEY_IMAGES = 'images'
TAR_IMAGES = 'images.tar'

s3Resource = boto3.resource('s3')

def createBucketName(prefix: str) -> str:
    """
    Create the bucket name
    :param prefix: Typically this will be a UUID to ensure uniqueness
    :return: The created name as an string
    """
    return ''.join([prefix, str(uuid.uuid4())])

def createBucket(bucket_prefix, s3_connection):
    """
    Create an empty S3 bucket
    :param bucket_prefix: The name of the bucket
    :param s3_connection: An opened S3 connection
    :return: (bucketName, AWS response)
    """
    session = boto3.session.Session()
    current_region = session.region_name
    #bucketName = createBucketName(bucket_prefix)
    bucketName = bucket_prefix
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

def prepareImagesTAR(options: OptionsFile):
    """
    Adds the images to the tar file and then removes the original
    :return:
    """
    # Find, add, and delete the images
    images = glob.glob(PATTERN_IMAGES)
    if len(images) == 0:
        log.debug("No images found")
        return

    # This is so we can call the images left and right
    theTAR = options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_NICK_JETSON) + "-" + TAR_IMAGES
    tar = tarfile.open(name=theTAR,mode="w|")
    for image in images:
        try:
            tar.add(image)
        except tarfile.TarError as e:
            log.fatal("Unable to add {} to tarfile.".format(image))
            log.fatal("{}".format(e))
        try:
            os.remove(image)
        except OSError as e:
            log.fatal("Could not remove image {}: {}".format(image, e.strerror))
    tar.close()
    return

def upload(bucket : str, options: OptionsFile) -> bool:
    """
    Upload a session's data to AWS
    :param bucket: Name of the S3 Bucket
    :param options: Options file
    """
    log.debug("Uploading: {}".format(bucket))
    # Create the target bucket that will hold data from this run if it is not already there
    config = Config(
        connect_timeout=10,
        read_timeout=10,
        retries={'max_attempts': 2})

    # Typically, unable to connect is caused by a network failure somewhere along the chain
    try:
        s3 = boto3.resource('s3', config=config)
    except Exception as ex:
        log.fatal("Unable to connect to AWS")
        log.fatal("Raw: {}".format(ex))
        return False

    try:
        buckets = s3.buckets.all()
        for aBucket in buckets:
            log.debug("Bucket: {}".format(aBucket.name))
        log.debug("Fetched bucket names")
    except Exception as ex:
        log.error("Hit an exception in upload")
        log.error(ex)
        return False

    bucketExists = False
    for aBucket in buckets:
        log.debug("Bucket: {}".format(aBucket.name))
        if aBucket.name == bucket:
            log.debug("Bucket {} already exists".format(bucket))
            bucketExists = True

    if bucketExists:
        bucketName = bucket
    else:
        try:
            log.debug("Creating bucket")
            bucketName, bucketResponse = createBucket(bucket, s3Resource.meta.client)
            first_bucket = s3Resource.Bucket(name=bucketName)
        except botocore.exceptions.ClientError as client:
            log.warning("Unable to create S3 bucket: {}".format(client))
            if client.response['Error']['Code'] == 'BucketAlreadyOwnedByYou':
                log.warning("Bucket {} already exists".format(bucket))
                bucketName = bucket
            else:
                log.warning("Ignoring this directory")
                return False

    log.debug("Preparing TAR")
    # Prepare things for S3 upload
    prepareImagesTAR(options)

    # Find all the files in the directory
    files = glob.glob('*')

    for file in files:
        try:
            key = options.option(constants.PROPERTY_SECTION_KEYS, file)
        except KeyError:
            key = file

        object = s3Resource.Object(bucket_name=bucketName, key=key)
        log.info("Uploading {}".format(file))
        object.upload_file(Filename=file)
    return True

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
parser.add_argument('-r', '--dns', action="store", required=False, help="DNS server address")

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
    systemRoom = MUCCommunicator(options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_SERVER),
                                 options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_JID_CLOUD),
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

my_resolver = dns.resolver.Resolver(configure=False)

if arguments.dns is not None:
    print("DNS: {}".format(arguments.dns))
    my_resolver.nameservers = [arguments.dns]


if arguments.watch or arguments.upload:
    watching = True
    while watching:

        try:
            time.sleep(30)
            answer = my_resolver.resolve('aws.amazon.com')
            #log.debug("AWS can be resolved, so uploads will be attempted")
        except dns.resolver.NoNameservers:
            log.debug("Can't resolve aws.amazon.com. This is normal under field conditions")
            time.sleep(30)
            continue
        except dns.exception.Timeout:
            log.debug("Can't resolve aws.amazon.com. This is normal under field conditions")
            # Sleep for a bit and wait for the tractor to be back in the shed
            time.sleep(30)
            continue

        directories = glob.glob(options.option(constants.PROPERTY_SECTION_GENERAL, constants.PROPERTY_PREFIX) + "*")
        for directory in directories:
            if os.path.isdir(directory):
                log.debug("Operating on: {}".format(directory))
                os.chdir(directory)
            else:
                syslog.syslog(syslog.LOG_WARNING, "File found in output directory: " + directory)

            # Look for the appearance of the .meta file, a signal that the run is complete
            # If it is not there, we will pick up this directory next time,
            # This will also allow catch-up.  Let's say 5 runs completed, and could not be uploaded because aws could not
            # be reached.  This loop will upload all of them.

            metadataFile = "*" + options.option(constants.PROPERTY_SECTION_GENERAL, constants.PROPERTY_SUFFIX_META)
            finished = glob.glob(metadataFile)
            if len(finished) > 0:

                uploaded = upload(directory, options)

                syslog.syslog(syslog.LOG_INFO,"Using S3 bucket: {}".format(directory))

                # Cleanup
                if uploaded:
                    try:
                        log.debug("Removing directory after upload.")
                        os.chdir("..")
                        shutil.rmtree(directory)
                    except OSError as e:
                        syslog.syslog(syslog.LOG_ERR, "Could not remove directory: {}".format(e.strerror))
            else:
                log.debug("No metadata file ({}) found, so skipping directory".format(metadataFile))
                os.chdir("..")

        # If this is just a single run, exit now
        if arguments.upload:
            watching = False

if arguments.watch:
    # Wait for the workers to finish
    for index, thread in enumerate(threads):
        thread.join()


sys.exit(0)