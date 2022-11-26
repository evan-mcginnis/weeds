#
# O P T I O N S F I L E
#

import configparser
import os.path
from typing import Tuple

# A helper function to allow tuples within the configuration file
# https://stackoverflow.com/questions/56967754/how-to-store-and-retrieve-a-dictionary-of-tuples-in-config-parser-python
# This is so we can have things like this in the options file:
# location = (100,200)

def parse_int_tuple(input):
    return tuple(int(k.strip()) for k in input[1:-1].split(','))

# The configuration file is an INI style file
class OptionsFile:
    def __init__(self, filename: str):
        self._config = configparser.ConfigParser(converters={'tuple': parse_int_tuple})
        self._filename = filename
        return

    @property
    def filename(self) -> str:
        return self._filename

    def load(self) -> bool:
        """
        Load an INI format file.
        :param filename: The INI file
        :return: True on success, False otherwise
        """
        if not os.path.isfile(self._filename):
            rc = False
        else:
            self._config.read(self._filename)
            rc = True

        return rc

    def tuple(self, section: str, name:str) -> Tuple:
        return(self._config[section].gettuple(name))

    def option(self, section: str, name: str) -> str:
        """
        The option from the section in the file
        :param name: The option name
        :return: The option value.  If the value cannot be found, the empty string.
        """
        return(self._config[section][name])


