"""
Print directories containing RLlib checkpoints in the configured
results directory.

Paths are sorted from oldest to newest.
"""

import os
from typing import List, Union

from configuration import RESULTS_DIR

IS_CP_FILE = '.is_checkpoint'


def most_recent_content_modification(dir_path: str) -> Union[int, float]:
    """Return the most recent modification time of a direct child file
    of the given directory.

    Args:
        dir_path (str): Directory to get the time for.

    Returns:
        Time of the most recent modification of a direct child file
            in seconds.
    """
    with os.scandir(dir_path) as dir_contents:
        return max(
            child_entry.stat().st_mtime
            for child_entry in dir_contents
            if child_entry.is_file()
        )


def sort_by_content_modification(dirs: List[str]) -> None:
    """Sort the given directories in-place by the modification time of
    their most recently modified direct child file.

    Args:
        dirs (List[str]): List of directories to sort.
    """
    dirs.sort(key=most_recent_content_modification)


def main() -> None:
    """Print directories containing RLlib checkpoints in the
    `results` directory.
    """
    this_dir = os.path.dirname(__file__)
    results_dir = os.path.join(this_dir, RESULTS_DIR)

    cp_dirs = []
    for (root, dirs, files) in os.walk(results_dir):
        if IS_CP_FILE in files and len(files) > 2:
            cp_dirs.append(root)

    sort_by_content_modification(cp_dirs)
    for cp_dir in cp_dirs:
        print(cp_dir)


if __name__ == '__main__':
    main()
