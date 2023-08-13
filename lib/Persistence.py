from __future__ import annotations

import pymongo.database
from bson.objectid import ObjectId
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from WeedExceptions import PersistenceError
import time
from datetime import datetime
import logging

NAME_DB = "weeds"
NAME_COLLECTION_IMAGES = "images"
NAME_COLLECTION_BLOBS = "blobs"

NAME_URL = "url"
NAME_LAT = "lat"
NAME_LONG = "long"
NAME_DATE = "date"
NAME_SEGMENTED = "segmented"
NAME_PROCESSED = "processed"
NAME_ALTITUDE = "altitude"
NAME_HASH = "hash"
NAME_ID = "_id"

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

class RawImage:
    def __init__(self,
                 url: str,
                 lat: float,
                 long: float,
                 agl: float,
                 date: datetime,
                 segmented: {},
                 processed: {},
                 hash: str,
                 id: ObjectId):
        """
        An unprocessed, but possiby segmented image
        :param url: URL of the image (typically a file on disk)
        :param lat: Latitude
        :param long: Longitude
        :param agl: Altitude AGL
        :param date: Date of acquisition
        :param segmented: dictionary of segmented versions of the image
        :param processed: dictionary of processed versions of the image
        :param hash: Hash of the image
        """
        self._url = url
        self._lat = lat
        self._long = long
        self._date = date
        self._hash = id
        self._agl = agl
        self._segmented = segmented
        self._processed = processed
        self._record = {
            NAME_URL: url,
            NAME_LAT: lat,
            NAME_LONG: long,
            NAME_DATE: date,
            NAME_ALTITUDE: agl,
            NAME_HASH: hash,
            NAME_SEGMENTED: segmented,
            NAME_PROCESSED: processed
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
    def segmented(self) -> {}:
        return self._segmented

    @segmented.setter
    def segmented(self, theSegmented: {}):
        self._segmented = theSegmented
        self._dirty.append(DIRTY_SEGMENTED)

    @property
    def processed(self) -> {}:
        return self._processed

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
                targetCollection.insert_one(self._record)
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

    @classmethod
    def find(cls, targetHash: str, connection: Mongo) -> RawImage:
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
                print(f"Not normal: Image with hash {targetHash} is in the DB multiple times")
                return None

            if imagesInDB > 0:
                return RawImage(record[0][NAME_URL],
                                record[0][NAME_LAT],
                                record[0][NAME_LONG],
                                record[0][NAME_ALTITUDE],
                                record[0][NAME_DATE],
                                record[0][NAME_SEGMENTED],
                                record[0][NAME_PROCESSED],
                                record[0][NAME_HASH],
                                record[0][NAME_ID])


        else:
            # TODO: This is a placeholder
            return False

    def __str__(self):
        return f"Hash: {self._hash}"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Persist images test")

    parser.add_argument("-host", "--host", action="store", required=True, help="DB Host")
    parser.add_argument("-p", "--port", type=int, action="store", required=True, help="DB Port")
    parser.add_argument("-d", "--database", action="store", required=True, help="DB Name")
    parser.add_argument("-hash", "--hash", action="store", required=False, help="Hash of image to find")
    arguments = parser.parse_args()

    db = Mongo()
    db.connect(arguments.host, arguments.port, "", "", arguments.database)
    datetime_str = '09/19/22 13:55:26'
    datetime_object = datetime.strptime(datetime_str, '%m/%d/%y %H:%M:%S')
    image = RawImage("http://foo", 0.0, 0.0, 0.0, datetime_object, {}, {}, arguments.hash)

    if arguments.hash is not None:
        found = RawImage.find(arguments.hash, db)
        if found is not None:
            print(f"{found}")
        else:
            print("Image is not in DB")
    else:
        image.save(db)


