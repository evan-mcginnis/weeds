#
# O P T I O N S F I L E
#

import configparser
import os.path

# The configuration file is an INI style file
class OptionsFile:
    def __init__(self, filename: str):
        self._config = configparser.ConfigParser()
        self._filename = filename
        return

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

    def option(self, section: str, name: str) -> str:
        """
        The option from the section in the file
        :param name: The option name
        :return: The option value.  If the value cannot be found, the empty string.
        """
        return(self._config[section][name])


