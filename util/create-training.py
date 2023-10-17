import argparse
import sys
import os
import logging
import logging.config
import datetime

import pandas as pd

import Persistence
import constants
from OptionsFile import OptionsFile
from Persistence import Mongo
from Persistence import Blob

parser = argparse.ArgumentParser("Create training file")

parser.add_argument("-ba", "--beginAGL", action="store", type=float, required=False, default=0.0, help="Begin Distance AGL")
parser.add_argument("-ea", "--endAGL", action="store", type=float, required=False, default=0.0, help="End Distance AGL")
parser.add_argument("-c", "--crop", action="store", required=False, default="unknown", help="Crop")
parser.add_argument("-bd", "--beginAcquired", action="store", required=False, type=datetime.date.fromisoformat, help="Beginning Date -- YYYY-MM-DD")
parser.add_argument("-ed", "--endAcquired", action="store", required=False, type=datetime.date.fromisoformat, help="Ending Date -- YYYY-MM-DD")
parser.add_argument('-ini', '--ini', action="store", required=False, default=constants.PROPERTY_FILENAME, help="Options INI")
parser.add_argument("-lg", "--logging", action="store", default="info-logging.ini", help="Logging configuration file")
parser.add_argument("-m", "--ml", action="store", required=False, default="lr", help="ML technique (lr, svm, etc.")
parser.add_argument("-host", "--host", action="store", required=False, help="DB Host")
parser.add_argument("-port", "--port", type=int, action="store", required=False, help="DB Port")
parser.add_argument("-dbname", "--dbname", action="store", required=False, help="DB Name")
parser.add_argument("-o", "--output", action="store", required=True, help="Output training data file to create")

arguments = parser.parse_args()

#
# L O G G I N G
#
if not os.path.isfile(arguments.logging):
    print("Unable to access logging configuration file {}".format(arguments.logging))
    sys.exit(1)

logging.config.fileConfig(arguments.logging)

# A blank query string
query = {}

# if arguments.begin is not None:
#     query[constants.KEYWORD_DATE_BEGIN] = str(arguments.begin)
# if arguments.end is not None:
#     query[constants.KEYWORD_DATE_END] = str(arguments.end)
# if arguments.agl is not None:
#     query[constants.KEYWORD_AGL] = arguments.agl
#
# D A T A B A S E
#
options = OptionsFile(arguments.ini)
options.load()
dbHost = options.option(constants.PROPERTY_SECTION_DATABASE, constants.PROPERTY_HOST) if arguments.host is None else arguments.host
dbPort = int(options.option(constants.PROPERTY_SECTION_DATABASE, constants.PROPERTY_PORT)) if arguments.port is None else arguments.port
dbName = options.option(constants.PROPERTY_SECTION_DATABASE, constants.PROPERTY_DB) if arguments.dbname is None else arguments.dbname
if dbHost is None or dbPort is None or dbName is None:
    logging.fatal("Specify host, port, and database to use a database on the command line or INI file")
    sys.exit(-1)

persistenceConnection = Mongo()
persistenceConnection.connect(dbHost, dbPort, "", "", dbName)
if not persistenceConnection.connected:
    logging.fatal("Unable to connect to database")
    sys.exit(-1)

# Find a list of base images to use
images = Persistence.RawImage.findByParameters(persistenceConnection,
                                               begin_agl=arguments.beginAGL,
                                               end_agl=arguments.endAGL,
                                               #begin=arguments.beginAcquired,
                                               #end=arguments.endAcquired,
                                               crop=arguments.crop)

blobs = []
print(f"Found {len(images)} images")
for image in images:
    mongoID = image.id
    # Find the blobs corresponding to each image
    blobsMeetingCriteria = Blob.findByParameters(persistenceConnection, parent=mongoID, ml=arguments.ml)
    # The list of all blobs
    for blob in blobsMeetingCriteria:
        factors = blob.factors
        factors[constants.NAME_TYPE] = blob.classified
        factors[constants.NAME_NAME] = image.name + constants.DASH + blob.name
        blobs.append(factors)

print(f"Found {len(blobs)} blobs")
training = pd.DataFrame(blobs)
training.to_csv(arguments.output, index=False)

# if arguments.altitude is not None:
#     #blobsMeetingCriteria = Blob.find(persistenceConnection, ALTITUDE=arguments.altitude)
#     blobsMeetingCriteria = Blob.find(persistenceConnection, **query)
#     print(f"Found {len(blobsMeetingCriteria)} records.")
#     for blob in blobsMeetingCriteria:
#         factors = blob.factors
#         factors[constants.NAME_TYPE] = blob.classified
#         observations.append(factors)
#         #print(f"{blob}\n")
#     training = pd.DataFrame(observations)
#     training.to_csv(arguments.output, index=False)



