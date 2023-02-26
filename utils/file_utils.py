from typing import Generator, List
from pathlib import Path
from natsort import natsorted


class PosePath(Path):
    """Class with additional functions for browsing through files."""

    def glob(self, pattern: str = '*', natsort: bool = False, to_list: bool = False) -> Generator[Path] | List[Path]:
        """
        Overriden function for better management of stored data.

        :param pattern: Pattern by which glob finds desired files.
        :type pattern: str
        :param natsort: If set to True, returned Paths will be natsorted.
        :type natsort: bool
        :param to_list: If set to True function will return list instead of a generator.
        :type to_list: bool
        :return: Generator or a list of Paths.
        :rtype: Generator[Path] | List[Path]
        """
        paths = super(self).glob(pattern)
        if natsort:
            paths = natsorted(paths)
        if to_list:
            paths = list(paths)
        return paths
