from __future__ import annotations
from typing import List, Union
from datetime import datetime

import pandas as pd
import pymongo.database
from bson.objectid import ObjectId
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

import constants
from WeedExceptions import PersistenceError
import time
from datetime import datetime
import logging

NAME_DB = "weeds"
NAME_COLLECTION_IMAGES = "images"
NAME_COLLECTION_BLOBS = "blobs"

NAME_NAME = "name"
NAME_URL = "url"
NAME_LAT = "lat"
NAME_LONG = "long"
NAME_DATE = "acquired"
NAME_CROP = "crop"
NAME_SEGMENTED = "segmented"
NAME_PROCESSED = "processed"
NAME_ALTITUDE = "altitude"
NAME_HASH = "hash"
NAME_ID = "_id"
NAME_BLOBS = "blobs"
NAME_ACQUIRED = "acquired"

class Persistence:
    def __init__(self):
        self._log = logging.getLogger(__name__)

    @property
    def connected(self) -> bool:
        return True

    @property
    def db(self):
        return ""

class Disk(Persistence):
    def __init__(self):
        super().__init__()


class Mongo(Persistence):
    def __init__(self):
        """
        Provides actions for a MongoDB store
        """
        super().__init__()
        self._connected = False
        self._client = None
        self._database = None


    def connect(self, host: str, port: int, username: str, password: str, database: str):
        """
        Connect to a mongodb server. Will retry 3 times to connect
        :param host: Host name or IP
        :param port: Port exposed
        :param username: Ignored
        :param password: Ignored
        :param database: Name of database
        """
        self._client = MongoClient(host, port)
        self._database = self._client[database]

        retries = 3
        while not self._connected and retries:
            try:
                # The ping command is cheap and does not require auth.
                self._client.admin.command('ping')
                self._connected = True
            except ConnectionFailure:
                print("Server not available")
                retries -= 1
                time.sleep(2)

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def db(self) -> pymongo.database.Database:
        return self._database

DIRTY_GENERAL = "general"
DIRTY_SEGMENTED = "segmented"
DIRTY_PROCESSED = "processed"

class Blob:
    NAME_FACTORS = "factors"
    NAME_FACTORS_USED = "used"
    NAME_CLASSIFICATION = "classification"
    NAME_HASH = "hash"
    NAME_PARENT = "parent"
    NAME_TECHNIQUE = "technique"

    KEYWORD_ID = "ID"
    KEYWORD_HASH = "HASH"

    KEYWORD_BEGIN = "BEGIN"
    KEYWORD_END = "END"
    KEYWORD_ALTITUDE = "ALTITUDE"

    def __init__(self,
                 name: str,
                 url: str,
                 lat: float,
                 long: float,
                 altitude: float,
                 factors: [],
                 factorsUsed: [],
                 technique: str,
                 classified: int,
                 parent: ObjectId,
                 imageHash: str,
                 blobID: ObjectId):
        """
        A blob in an image
        :param url: URL of the image on disk
        :param lat: Latitude
        :param long: Longitude
        :param altitude: Altitude in meters
        :param factors: Array of factors values
        :param factorsUsed: Array of factors used in classificaton
        :param technique: Name of the technique
        :param classified: 0 or 1
        :param parent: ObjectID of parent image
        :param imageHash: Hash of this blob
        :param blobID: ObjectID in DB
        """
        self._name = name
        self._url = url
        self._lat = lat
        self._long = long
        self._altitude = altitude
        self._factors = factors
        self._factorUsed = factorsUsed
        self._technique = technique
        self._classified = classified
        self._hash = imageHash
        self._id = blobID
        self._parent = parent
        self._dirty = []

        self.DIRTY_CLASSIFIED = "classified"
        self.DIRTY_FACTORS = "factors"
        self.DIRTY_GENERAL = "general"


    def isInDB(self) -> bool:
        return self._id is not None

    @property
    def hash(self) -> str:
        return self._hash

    @hash.setter
    def hash(self, theHash: str):
        self._hash = theHash
        self._dirty.append(self.DIRTY_GENERAL)

    @property
    def technique(self) -> str:
        return self._technique

    @technique.setter
    def technique(self, theTechnique: str):
        self._technique = theTechnique
        self._dirty.append(self.DIRTY_GENERAL)

    @property
    def parent(self) -> ObjectId:
        return self._parent

    @parent.setter
    def parent(self, theParent: ObjectId):
        self._parent = theParent
        self._dirty.append(self.DIRTY_GENERAL)

    @property
    def name(self) -> str:
        return self._name

    @property
    def id(self) -> ObjectId:
        return self._id

    @property
    def gps(self) -> ():
        return self._lat, self._long

    @gps.setter
    def gps(self, coordinates: ()):
        self._lat = coordinates[0]
        self._long = coordinates[1]

    @property
    def altitude(self) -> float:
        return self._altitude

    @altitude.setter
    def altitude(self, theAltitude: float):
        self._altitude = theAltitude
        self._dirty.append(self.DIRTY_GENERAL)

    @property
    def factors(self) -> []:
        return self._factors

    @factors.setter
    def factors(self, theFactors):
        self._factors = theFactors
        self._dirty.append(self.DIRTY_CLASSIFIED)

    @property
    def used(self) -> []:
        return self._factorUsed

    @used.setter
    def used(self, theFactors):
        self._factors = theFactors
        self._dirty.append(self.DIRTY_CLASSIFIED)

    @property
    def classified(self) -> int:
        return self._classified

    @classified.setter
    def classified(self, theClassification: int):
        self._classified = theClassification
        self._dirty.append(self.DIRTY_GENERAL)

    @classmethod
    def find(cls, connection: Mongo, **kwargs) -> List['__class__']:
        """
        Find a set of blobs meeting a specific condition
        :param connection:
        :param kwargs: ID or HASH
        :return: array of blobs
        """
        # TODO: There's probably a better way to provide a type hint for the return list
        imageToFind = {}
        for key, value in kwargs.items():
            if key == Blob.KEYWORD_ID:
                imageToFind = {NAME_ID: ObjectId(value)}
            elif key == Blob.KEYWORD_HASH:
                imageToFind = {NAME_HASH: value}
            elif key == Blob.KEYWORD_BEGIN:
                pass
            elif key == Blob.KEYWORD_END:
                pass
            elif key == Blob.KEYWORD_ALTITUDE:
                imageToFind = {NAME_ALTITUDE: float(value)}
            else:
                raise TypeError

        targetCollection = connection.db[NAME_COLLECTION_BLOBS]
        imagesFound = targetCollection.find(imageToFind)
        results = []
        resultNumber = 0

        for record in imagesFound:
            aBlob = Blob(record[NAME_NAME],
                         record[NAME_URL],
                         record[NAME_LAT],
                         record[NAME_LONG],
                         record[NAME_ALTITUDE],
                         record[Blob.NAME_FACTORS],
                         record[Blob.NAME_FACTORS_USED],
                         record[Blob.NAME_TECHNIQUE],
                         record[Blob.NAME_CLASSIFICATION],
                         record[Blob.NAME_PARENT],
                         record[Blob.NAME_HASH],
                         record[NAME_ID])
            results.append(aBlob)

        return results
    @classmethod
    def findByParameters(cls, connection: Mongo, **kwargs) -> List[Blob]:
        """
        Find a set of blobs meeting a specific condition
        :param connection:
        :param kwargs: ID or HASH
        :return: array of blobs
        """
        # TODO: There's probably a better way to provide a type hint for the return list
        imageToFind = {}

        if constants.KEYWORD_PARENT in kwargs:
            parentID = kwargs[constants.KEYWORD_PARENT]
        else:
            raise (TypeError(f"{constants.KEYWORD_PARENT} must be specified"))

        if constants.KEYWORD_ML_TECHNIQUE in kwargs:
            mlTechnique = kwargs[constants.KEYWORD_ML_TECHNIQUE]
        else:
            raise (TypeError(f"{constants.KEYWORD_ML_TECHNIQUE} must be specified"))


        query = {"$and": [{"parent": {"$eq": parentID}},
                          {"technique": mlTechnique}]}


        targetCollection = connection.db[NAME_COLLECTION_BLOBS]
        imagesFound = targetCollection.find(query)
        results = []


        for record in imagesFound:
            aBlob = Blob(record[NAME_NAME],
                         record[NAME_URL],
                         record[NAME_LAT],
                         record[NAME_LONG],
                         record[NAME_ALTITUDE],
                         record[Blob.NAME_FACTORS],
                         record[Blob.NAME_FACTORS_USED],
                         record[Blob.NAME_TECHNIQUE],
                         record[Blob.NAME_CLASSIFICATION],
                         record[Blob.NAME_PARENT],
                         record[Blob.NAME_HASH],
                         record[NAME_ID])
            results.append(aBlob)

        return results

    def save(self, connection: Mongo):
        """
        Persist the current record using the persistence approach
        :param connection: A connection to the DB
        """

        if not connection.connected:
            raise PersistenceError("Not connected to database")

        if type(connection) is Mongo:
            # Check to see if this is already in DB
            if self._id is None:
                record = {
                    NAME_NAME: self._name,
                    NAME_URL: self._url,
                    NAME_LAT: self._lat,
                    NAME_LONG: self._long,
                    NAME_ALTITUDE: self._altitude,
                    Blob.NAME_FACTORS: self._factors,
                    Blob.NAME_FACTORS_USED: self._factorUsed,
                    Blob.NAME_TECHNIQUE: self._technique,
                    Blob.NAME_CLASSIFICATION: self._classified,
                    Blob.NAME_PARENT: self._parent,
                    NAME_HASH: self._hash
                }

                targetCollection = connection.db[NAME_COLLECTION_BLOBS]
                targetCollection.insert_one(record)
            else:
                targetCollection = connection.db[NAME_COLLECTION_BLOBS]
                # See what needs to be updated
                if DIRTY_PROCESSED in self._dirty:
                    myquery = {NAME_ID: self._id}
                    newvalues = {"$set": {NAME_PROCESSED: self._processed}}

                    targetCollection.update_one(myquery, newvalues)

                if DIRTY_SEGMENTED in self._dirty:
                    myquery = {NAME_ID: self._id}
                    newvalues = {"$set": {NAME_SEGMENTED: self._segmented}}

                    targetCollection.update_one(myquery, newvalues)

        elif type(connection) is Disk:
            metadata = self._url + ".meta"
            with open(metadata, "w") as meta:
                meta.write(f"{self._record}")

    def __str__(self) -> str:
        # self._url = url
        # self._lat = lat
        # self._long = long
        # self._factors = factors
        # self._factorUsed = factorsUsed
        # self._technique = technique
        # self._classified = classified
        # self._hash = imageHash
        # self._id = blobID
        # self._parent = parent
        stringRepresentation = f"NAME: {self._name}\nURL: {self._url}\nLatitude: {self._lat}\nLongitude: {self._long}\nAltitude: {self._altitude}\nTechnique: {self._technique}\nClassified: {self._classified}\n"
        stringRepresentation += f"Used: {self._factorUsed}\nHash: {self._hash}\nParent: {self.parent}"
        return stringRepresentation

class RawImage:
    log = logging.getLogger(__name__)
    def __init__(self,
                 name: str,
                 url: str,
                 lat: float,
                 long: float,
                 agl: float,
                 date: datetime,
                 crop: str,
                 segmented: {},
                 processed: {},
                 hash: str,
                 id: ObjectId):
        """
        An unprocessed, but possibly segmented image
        :param url: URL of the image (typically a file on disk)
        :param lat: Latitude
        :param long: Longitude
        :param agl: Altitude AGL
        :param date: Date of acquisition
        :param crop: crop in the image
        :param segmented: dictionary of segmented versions of the image
        :param processed: dictionary of processed versions of the image
        :param hash: Hash of the image
        """
        self._name = name
        self._url = url
        self._lat = lat
        self._long = long
        self._crop = crop
        self._date = date
        self._hash = id
        self._agl = agl
        self._segmented = segmented
        self._processed = processed
        self._blobs = None
        self._record = {
            NAME_NAME: name,
            NAME_URL: url,
            NAME_LAT: lat,
            NAME_LONG: long,
            NAME_DATE: date,
            NAME_CROP: crop,
            NAME_ALTITUDE: agl,
            NAME_HASH: hash,
            NAME_SEGMENTED: segmented,
            NAME_PROCESSED: processed,
        }
        self._connection = None
        self._id = id

        self._dirty = []

    @property
    def id(self) -> ObjectId:
        return self._id

    @property
    def connection(self):
        return self._connection

    @connection.setter
    def connection(self, theConnection: Mongo):
        self._connection = theConnection

    @property
    def name(self) -> str:
        return self._name

    @property
    def URL(self) -> str:
        return self._url

    @URL.setter
    def URL(self, theURL: str):
        self._dirty.append(DIRTY_GENERAL)
        self._url = theURL

    @property
    def gps(self) -> ():
        return self._lat, self._long

    @gps.setter
    def gps(self, coordinates: ()):
        self._lat = coordinates[0]
        self._long = coordinates[1]
        self._dirty.append(DIRTY_GENERAL)

    @property
    def hash(self) -> str:
        return self._hash

    @hash.setter
    def hash(self, theHash: str):
        self._hash = theHash
        self._dirty.append(DIRTY_GENERAL)

    @property
    def AGL(self) -> float:
        return self._agl

    @AGL.setter
    def AGL(self, theAGL: float):
        self._agl = theAGL
        self._dirty.append(DIRTY_GENERAL)

    @property
    def crop(self) -> str:
        return self._crop

    @crop.setter
    def crop(self, theCrop: str):
        self._crop = theCrop
        self._dirty.append(DIRTY_GENERAL)

    @property
    def segmented(self) -> {}:
        return self._segmented

    @segmented.setter
    def segmented(self, theSegmented: {}):
        self._segmented = theSegmented
        self._dirty.append(DIRTY_SEGMENTED)

    @property
    def processed(self) -> {}:
        return self._processed

    @property
    def blobs(self) -> {}:
        return self._blobs


    @processed.setter
    def processed(self, theProcessed: {}):
        self._processed = theProcessed
        self._dirty.append(DIRTY_PROCESSED)


    def addProcessed(self, technique: str, url: str):
        """
        Add a processed image
        :param technique: <segmentation>-<ml>, i.e., ndi-lda
        :param url: URL of the processed image
        """
        self._processed[technique] = url
        self._dirty.append(DIRTY_PROCESSED)

    def addSegmented(self, technique: str, url: str):
        """
        Add a segmented image
        :param technique: Name of segmentation technique
        :param url: URL of the segmented image
        """
        self._segmented[technique] = url
        self._dirty.append(DIRTY_SEGMENTED)


    def save(self, connection: Persistence):
        """
        Persist the current record using the persistence approach
        :param record: Record in db
        :param connection: A connection to the DB
        """

        if not connection.connected:
            raise PersistenceError("Not connected to database")

        if type(connection) is Mongo:
            # Check to see if this is already in DB
            if self._id is None:
                targetCollection = connection.db[NAME_COLLECTION_IMAGES]
                inserted = targetCollection.insert_one(self._record)
                self._id = inserted.inserted_id
            else:
                targetCollection = connection.db[NAME_COLLECTION_IMAGES]
                # See what needs to be updated
                if DIRTY_PROCESSED in self._dirty:
                    myquery = {NAME_ID: self._id}
                    newvalues = {"$set": {NAME_PROCESSED: self._processed}}

                    targetCollection.update_one(myquery, newvalues)

                if DIRTY_SEGMENTED in self._dirty:
                    myquery = {NAME_ID: self._id}
                    newvalues = {"$set": {NAME_SEGMENTED: self._segmented}}

                    targetCollection.update_one(myquery, newvalues)

        elif type(connection) is Disk:
            metadata = self._url + ".meta"
            with open(metadata, "w") as meta:
                meta.write(f"{self._record}")

    # @classmethod
    # def find(cls, connection: Mongo, **kwargs) -> List['__class__']:
    #     imagesFound = []
    #     imageToFind = {}
    #     specificRequested = False
    #     rangeRequested = False
    #
    #     for key, value in kwargs.items():
    #         if key == Blob.KEYWORD_ID:
    #             if rangeRequested:
    #                 raise ValueError(f"{Blob.KEYWORD_ID} conflicts with date range")
    #             imageToFind = {NAME_ID: ObjectId(value)}
    #             specificRequested = True
    #         elif key == Blob.KEYWORD_HASH:
    #             if rangeRequested:
    #                 raise ValueError(f"{Blob.KEYWORD_HASH} conflicts with date range")

    #             imageToFind = {NAME_HASH: value}
    #         elif key == Blob.KEYWORD_BEGIN:
    #             pass
    #         elif key == Blob.KEYWORD_END:
    #             pass
    #         elif key == Blob.KEYWORD_ALTITUDE:
    #             imageToFind = {NAME_ALTITUDE: float(value)}
    #         else:
    #             raise TypeError
    #
    #     targetCollection = connection.db[NAME_COLLECTION_IMAGES]
    #     imagesFound = targetCollection.find(imageToFind)
    #
    #     return imagesFound

    # I suppose this is more elegant that using the actual class name, but
    # I can't get pycharm to understand the type hint in this form
    # def findByParameters(cls, connection: Mongo, **kwargs) -> List[__class__]:

    @classmethod
    def findByParameters(cls, connection: Mongo, **kwargs) -> List['RawImage']:
        """
        Locates an image with the given parameters
        :param connection:
        :return: list of raw images or None
        """

        query = ""
        # by date range
        beginAGL = 0
        endAGL = 0
        beginSpecified = False
        endSpecified = False
        images = []

        if constants.KEYWORD_BEGIN_AGL in kwargs:
            beginAGL = kwargs[constants.KEYWORD_BEGIN_AGL]
            beginSpecified = True
        if constants.KEYWORD_END_AGL in kwargs:
            endAGL = kwargs[constants.KEYWORD_END_AGL]
            endSpecified = True
        if (beginSpecified and not endSpecified) or (not beginSpecified and endSpecified):
            raise (TypeError(
                f"Both {constants.KEYWORD_BEGIN_AGL} and {constants.KEYWORD_END_AGL} must be specified if either is"))

        beginDateSpecified = False
        endDateSpecified = False
        if constants.KEYWORD_BEGIN_DATE in kwargs:
            beginDate = kwargs[constants.KEYWORD_BEGIN_DATE]
            beginDateSpecified = True
        if constants.KEYWORD_END_DATE in kwargs:
            endDate = kwargs[constants.KEYWORD_END_DATE]
            endDateSpecified = True
        if (beginDateSpecified and not endDateSpecified) or (not beginDateSpecified and endDateSpecified):
            raise (TypeError(
                f"Both {constants.KEYWORD_BEGIN_DATE} and {constants.KEYWORD_END_DATE} must be specified if either is"))

        crop = "unknown"
        if constants.KEYWORD_CROP in kwargs:
            crop = kwargs[constants.KEYWORD_CROP]
        else:
            raise (TypeError(f"{constants.KEYWORD_CROP} is required"))

        #dateStart = datetime.strptime(beginDate, '%Y-%m-%d')
        #dateEnd = datetime.strptime(endDate, '%Y-%m-%d')

        query = {"$and": [{"altitude": {"$gte": beginAGL, "$lte": endAGL}},
                          {"crop": crop},
                          {"acquired": {"$gte": beginDate, "$lte": endDate}}]}

        if type(connection) is Mongo:
            imageToFind = query
            targetCollection = connection.db[NAME_COLLECTION_IMAGES]
            imagesFound = targetCollection.find(imageToFind)

            records = list(imagesFound.clone())
            imagesInDB = len(records)
            # Construct the response
            for record in records:
                image = RawImage(record[NAME_NAME],
                                 record[NAME_URL],
                                 record[NAME_LAT],
                                 record[NAME_LONG],
                                 record[NAME_ALTITUDE],
                                 record[NAME_CROP],
                                 record[NAME_DATE],
                                 record[NAME_SEGMENTED],
                                 record[NAME_PROCESSED],
                                 record[NAME_HASH],
                                 record[NAME_ID])
                images.append(image)
        else:
            # TODO: This is a placeholder
            pass

        return images

    @classmethod
    def find(cls, targetHash: str, connection: Mongo) -> Union[RawImage, None]:
        """
        Locates an image with the specified hash
        :param targetHash: Hash of the image
        :param connection:
        :return: True if found
        """
        if type(connection) is Mongo:
            imageToFind = {"hash": targetHash}
            targetCollection = connection.db[NAME_COLLECTION_IMAGES]
            imagesFound = targetCollection.find(imageToFind)

            record = list(imagesFound.clone())
            imagesInDB = len(record)
            if imagesInDB > 1:
                raise PersistenceError(f"Not normal: Image with hash {targetHash} is in the DB multiple times")

            if imagesInDB > 0:
                return RawImage(record[0][NAME_NAME],
                                record[0][NAME_URL],
                                record[0][NAME_LAT],
                                record[0][NAME_LONG],
                                record[0][NAME_ALTITUDE],
                                record[0][NAME_CROP],
                                record[0][NAME_DATE],
                                record[0][NAME_SEGMENTED],
                                record[0][NAME_PROCESSED],
                                record[0][NAME_HASH],
                                record[0][NAME_ID])


        else:
            # TODO: This is a placeholder
            return None

class Factors:
    def __init__(self, connection: Mongo, parent: str, technique: str, altitude: float):
        """
        Factors associated with one or more images
        :param connection: MongoDB Connection
        :param parent: Parent hash or "all"
        :param technique: technique used for classification, such as "lr" or "lda"
        :param altitude: altitude AGL
        """
        self._connection = connection
        self._parent = parent
        self._technique = technique
        self._altitude = altitude
        self._observations = pd.DataFrame()

    @property
    def observations(self) -> pd.DataFrame:
        return self._observations

    def read(self):
        blobsToFind = {"$and": [{"technique": self._technique}, {"altitude": self._altitude}]}
        targetCollection = self._connection.db[NAME_COLLECTION_BLOBS]
        imagesFound = targetCollection.find(blobsToFind)


    def write(self, filename: str):
        pass





if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Persist images test")

    parser.add_argument("-host", "--host", action="store", required=True, help="DB Host")
    parser.add_argument("-p", "--port", type=int, action="store", required=True, help="DB Port")
    parser.add_argument("-d", "--database", action="store", required=True, help="DB Name")
    parser.add_argument("-hash", "--hash", action="store", required=False, help="Hash of image to find")
    parser.add_argument("-g", "--generate", action="store_true", required=False, default=False, help="Generate entries")
    parser.add_argument("-b", "--generate", action="store_true", required=False, default=False, help="Generate entries")
    arguments = parser.parse_args()

    db = Mongo()
    db.connect(arguments.host, arguments.port, "", "", arguments.database)
    datetime_str = '09/19/22 13:55:26'
    datetime_object = datetime.strptime(datetime_str, '%m/%d/%y %H:%M:%S')
    url = "http://foo"
    lat = 0.0
    long = 0.0
    agl = 0.0
    technique = "lda"
    classification = 1
    factors = {}
    factors['factor1'] = 0.11111
    factors['factor2'] = 0.22222
    factorsUsed = ['factor1', 'factor2']
    image = RawImage("http://foo", 0.0, 0.0, 0.0, datetime_object, {}, {}, arguments.hash, None)

    if arguments.generate:
        image.save(db)
        parent = ObjectId(image.id)
        blob = Blob(url, lat, long, agl, factors, factorsUsed, technique, 0, parent, "3EF", None)
        blob.save(db)
    if arguments.hash is not None:
        found = RawImage.find(arguments.hash, db)
        if found is not None:
            print(f"{found}")
        else:
            print("Image is not in DB")
    else:
        image.save(db)


