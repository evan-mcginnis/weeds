#
# A W S  D O W N L O A D
#

import argparse

# this is so we don't get deprecation warnings about the version of python used
import errno
import glob
import os
import sys
import tarfile

from pathlib import Path

import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)
from botocore.config import Config

warnings.filterwarnings("ignore")
import boto3
import botocore.exceptions

parser = argparse.ArgumentParser("AWS S3 Download")
parser.add_argument("-s", "--s3", action="store", required=False, default="s3.ini", help="S3 INI")
parser.add_argument("-b", "--bucket", action="store", required=True, help="S3 bucket name or ALL")
parser.add_argument("-o", "--output", action="store", required=False, default=".", help="Output directory")
arguments = parser.parse_args()
config = Config(
    connect_timeout=10,
    read_timeout=10,
    retries={'max_attempts': 2})


def expandTARs(outputDirectory: str) -> bool:
    try:
        os.chdir(outputDirectory)
    except Exception as ex:
        print("Unable to change directory to {}".format(outputDirectory))
        return False

    # Find the tar files
    archives = glob.glob('*.tar')
    for archive in archives:
        basename = Path(archive).stem
        try:
            os.mkdir(basename)
        except OSError as err:
            print("Unable to make directory {}".format(basename))
            if err.errno == errno.EACCES:
                print("Permission denied")
            elif err.errno == errno.ENOSPC:
                print("No space on device")
            return False

        try:
            tar = tarfile.open(name=archive, mode="r")
            tar.extractall(path=basename)
        except tarfile.TarError as err:
            print("Unable to extract from {}.".format(archive))
            print("Raw: {}".format(err))
            return False

    try:
        os.chdir("..")
    except Exception as ex:
        print("Unable to change directory to ..")
        return False

    return True


def downloadS3Bucket(bucket: str, outputDirectory: str) -> bool:
    try:
        client = boto3.client('s3', config=config)
    except Exception as ex:
        print("Unable to connect to AWS")
        print("Raw: {}".format(ex))
        return False

    try:
        response = client.list_objects_v2(Bucket=bucket)
    except botocore.exceptions.ClientError as error:
        print("Can't find bucket: {}".format(bucket))
        sys.exit(1)

    path = outputDirectory + "/" + bucket
    try:
        os.makedirs(path, exist_ok=True)
        os.chdir(path)
    except OSError as e:
        print("Unable to prepare and set directory to {}".format(path))
        print("Raw: {}".format(e))
        return False

    if 'Contents' in response:
        for item in response['Contents']:
            print('Downloading file', item['Key'])
            try:
                client.download_file(bucket, item['Key'], item['Key'])
            except Exception as ex:
                print("Unable to download {}".format(item['Key']))
                print("Raw: {}".format(ex))

            while response['KeyCount'] == 1000:
                response = client.list_objects_v2(
                    Bucket=bucket,
                    StartAfter=response['Contents'][0]['Key'],
                )
                for item in response['Contents']:
                    print('Downloading file', item['Key'])
                    # client.delete_object(Bucket=bucket, Key=item['Key'])
    try:
        os.chdir("..")
    except Exception as ex:
        print("Unable to change directory to [..]")
        return False


    return True


try:
    s3 = boto3.resource('s3', config=config)
except Exception as ex:
    print("Unable to connect to AWS")
    print("Raw: {}".format(ex))

if arguments.bucket == "ALL":
    for bucket in s3.buckets.all():
        print("Download: {}".format(bucket.name))
        result = downloadS3Bucket(bucket.name, arguments.output)

        if result:
            print("Expand TAR files")
            result = expandTARs(bucket.name)

else:
    print("Download: {}".format(arguments.bucket))
    result = downloadS3Bucket(arguments.bucket, arguments.output)
    if result:
        print("Expand TAR files")
        result = expandTARs(arguments.bucket)

sys.exit(result == True)
