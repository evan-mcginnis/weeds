import argparse
import sys
import os
import logging
import logging.config

import pandas as pd

import constants
from OptionsFile import OptionsFile
from Persistence import Mongo
from Persistence import Blob

parser = argparse.ArgumentParser("Create training file")

parser.add_argument("-a", "--altitude", action="store", type=float, required=False, help="Altitude")
parser.add_argument("-b", "--begin", action="store", required=False, help="Beginning Date")
parser.add_argument("-e", "--end", action="store", required=False, help="Ending Date")
parser.add_argument("-host", "--host", action="store", required=False, help="DB Host")
parser.add_argument('-ini', '--ini', action="store", required=False, default=constants.PROPERTY_FILENAME, help="Options INI")
parser.add_argument("-lg", "--logging", action="store", default="info-logging.ini", help="Logging configuration file")
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

observations = []
if arguments.altitude is not None:
    blobsMeetingCriteria = Blob.find(persistenceConnection, ALTITUDE=arguments.altitude)
    for blob in blobsMeetingCriteria:
        factors = blob.factors
        factors[constants.NAME_TYPE] =  blob.classified
        observations.append(factors)
        #print(f"{blob}\n")
    training = pd.DataFrame(observations)
    training.to_csv(arguments.output, index=False)



