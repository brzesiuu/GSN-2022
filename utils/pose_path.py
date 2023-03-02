from __future__ import annotations
from typing import List
import os
from pathlib import Path
from pathlib import PosixPath as _PosixPath_
from pathlib import WindowsPath as _WindowsPath_

from natsort import natsorted, ns


class PosePath(Path):
    def __new__(cls, *args, **kwargs) -> PosePath:
        """Join self with args (like in Path) and return string."""
        return super().__new__(WindowsPath if os.name == 'nt' else PosixPath, *args, **kwargs)

    def str_join(self, *args) -> str:
        """Join path of one of self.parents with args, return string."""
        return str(self.joinpath(*args))

    def str_join_parent(self, parents_idx, *args):
        return self.parents[parents_idx].str_join(*args)

    def pose_glob(self, pattern: str = '*', to_string: bool = False, natsort: bool = False, to_list: bool = False)\
            -> List[PosePath]:
        """
        Overloaded glob function for listing all files with given pattern in a given directory pointed by self
        :param pattern: Pattern of filenames to look for
        :type pattern: str
        :param to_string: Decides whether filenames in returned list should be converted to strings
        :type to_string: bool
        :param natsort: Decides whether returned list of paths should be natsorted
        :type natsort: bool
        :param to_list: If set to True list instead of a generator will be returned
        :type to_list: bool
        :return: List of paths with given pattern and inside a given directory.
        :rtype: List[PosePath] or a generator.
        """
        glob_list = []
        if isinstance(pattern, str):
            glob_list = self.glob(pattern)
        else:
            glob_list = (p for p in self.glob("*") if p.suffix in set(pattern))
        if natsort:
            glob_list = natsorted(glob_list, alg=ns.PATH)
        if to_list:
            glob_list = [x for x in glob_list]
        return glob_list


class WindowsPath(_WindowsPath_, PosePath):
    pass


class PosixPath(_PosixPath_, PosePath):
    pass
