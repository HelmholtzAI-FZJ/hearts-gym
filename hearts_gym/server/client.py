import re
from typing import Optional

from hearts_gym.server.utils import Address, Request


class Client:
    __slots__ = [
        'player_index',
        'request',
        'address',
        'is_registered',
        '_name',
    ]

    MAX_NAME_BYTES = 64
    """Maximum byte length of a name."""
    CONTROL_CHAR_RE = re.compile('[\x00-\x1f\x7f-\x9f]')
    """Regular expression to match Unicode control characters."""

    def __init__(
            self,
            player_index: int,
            request: Request,
            address: Address,
    ) -> None:
        self.player_index = player_index
        self.request = request
        self.address = address
        self.is_registered = True

        self._name = 'Player ' + str(player_index + 1)

    @property
    def name(self) -> str:
        """Name of the client."""
        return self._name

    @staticmethod
    def check_name_length(name: bytes) -> None:
        """Raise an error if the given name is too long or empty.

        Args:
            name (bytes): The name to filter.
        """
        if len(name) == 0:
            raise ValueError('name cannot be empty')
        if len(name) > Client.MAX_NAME_BYTES:
            raise ValueError(
                f'name cannot be longer than {Client.MAX_NAME_BYTES} bytes')

    @staticmethod
    def _filter_name(name: bytes) -> Optional[str]:
        """Return the given name filtered for printing.

        If the name is completely rejected, return `None`.

        Args:
            name (bytes): The name to filter.

        Returns:
            Optional[str]: The filtered name or `None` if it
                was rejected.
        """
        # TODO we should also filter bad words here
        name = name.decode('utf-8', errors='replace')
        name = Client.CONTROL_CHAR_RE.sub('', name)
        return name

    def set_name(self, name: bytes) -> None:
        """Set the name to the given new one.

        If the new name is rejected, do not change it.

        Args:
            name (bytes): New name to set.
        """
        self.check_name_length(name)
        name = self._filter_name(name)
        if name is None:
            return
        self._name = name
