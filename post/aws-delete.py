#
# A W S  D E L E T E
#

import argparse

# this is so we don't get deprecation warnings about the version of python used
import sys

import warnings

# warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")
import boto3
import botocore.exceptions

parser = argparse.ArgumentParser("AWS S3 Delete")
parser.add_argument("-s", "--s3", action="store", required=False, default="s3.ini", help="S3 INI")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("-b", "--bucket", action="store", required=False, help="S3 bucket name or ALL")
group.add_argument("-p", "--prefix", action="store", required=False,
                   help="S3 bucket prefix. i.e., tucson to delete all buckets beginning with the name tucson")
arguments = parser.parse_args()


# empty existing bucket
def empty_s3_bucket(bucket: str):
    client = boto3.client('s3')
    try:
        response = client.list_objects_v2(Bucket=bucket)
    except botocore.exceptions.ClientError as error:
        print("Can't find bucket: {}".format(bucket))
        sys.exit(1)
    if 'Contents' in response:
        for item in response['Contents']:
            print('deleting file', item['Key'])
            client.delete_object(Bucket=bucket, Key=item['Key'])
            while response['KeyCount'] == 1000:
                response = client.list_objects_v2(
                    Bucket=bucket,
                    StartAfter=response['Contents'][0]['Key'],
                )
                for item in response['Contents']:
                    print('deleting file', item['Key'])
                    client.delete_object(Bucket=bucket, Key=item['Key'])


s3 = boto3.resource('s3')

s3 = boto3.resource('s3')

# Delete all the buckets
if arguments.bucket == "ALL":
    for bucket in s3.buckets.all():
        print("Delete: {}".format(bucket.name))
        empty_s3_bucket(bucket.name)
        bucket = s3.Bucket(bucket.name)
        bucket.delete()

# Delete the ones with a given prefix
elif arguments.prefix is not None:
    for bucket in s3.buckets.all():
        if bucket.name.startswith(arguments.prefix):
            print("Delete: {}".format(bucket.name))
            empty_s3_bucket(bucket.name)
            bucket = s3.Bucket(bucket.name)
            bucket.delete()

# Delete a specific bucket
else:
    empty_s3_bucket(arguments.bucket)
    bucket = s3.Bucket(arguments.bucket)
    bucket.delete()
