"""
Utilities for client-server interaction.
"""

import json
import socket
from typing import Any, List, Optional, Tuple
import zlib

from ray.rllib.utils.typing import TensorType

from hearts_gym.envs.hearts_env import Action

Request = socket.socket
Address = Tuple[str, int]

OK_MSG = b'__OK'

MSG_LENGTH_SEPARATOR = b';'
ACTION_SEPARATOR = b','

# Small power of two as recommended by Python documentation.
MAX_RECEIVE_BYTES = 8192

MAX_MSG_BYTES = 999_999_999  # Allow messages of up to 1 GB.
MAX_MSG_PREFIX_LENGTH = len(str(MAX_MSG_BYTES)) + len(MSG_LENGTH_SEPARATOR)
"""Maximum string length of the message length prefix, including
the separator.
"""


def prefix_data(data: bytes) -> bytes:
    """Return the given data prefixed with its length.

    Args:
        data (bytes): Data to prefix.

    Returns:
        bytes: Prefixed data ready for unknown-length receipt.
    """
    data_len = len(data)
    assert data_len <= MAX_MSG_BYTES, 'message is too large'
    return str(data_len).encode() + MSG_LENGTH_SEPARATOR + data


def encode_int(integer: int) -> bytes:
    """Return the given integer encoded as a bytes string.

    For example, `encode_int(10) == b'10'`.

    Args:
        integer (int): Integer value to encode.

    Returns:
        bytes: The literally encoded integer.
    """
    return str(integer).encode()


def encode_actions(actions: TensorType) -> bytes:
    """Return the given actions encoded as a bytes string.

    Args:
        actions (TensorType): Batch of actions to encode.

    Returns:
        bytes: Actions encoded as bytes.
    """
    if len(actions) == 0:
        data = ACTION_SEPARATOR
    else:
        data = ACTION_SEPARATOR.join(map(encode_int, actions))
    data = prefix_data(data)
    return data


def decode_actions(data: bytes) -> List[Action]:
    """Parse actions from the given message data.

    It is assumed that the data has been stripped of its prefix.

    Args:
        data (bytes): The message received.

    Returns:
        List[Action]: Actions contained in the message.
    """
    if data == ACTION_SEPARATOR:
        return []
    return list(map(int, data.split(ACTION_SEPARATOR)))


def encode_data(data: Any) -> bytes:
    """Return the given data encoded as a message from server to client.

    Args:
        data (Any): Data to encode for sending.

    Returns:
        bytes: Encoded data, prefixed with the length of the data and
            a `MSG_LENGTH_SEPARATOR`.
    """
    data: str = json.dumps(data, separators=(',', ':'))
    data: bytes = data.encode()
    data: bytes = zlib.compress(data)
    data: bytes = prefix_data(data)
    return data


def decode_data(data: bytes) -> Any:
    """Return the given data decoded from a message from server
    to client.

    It is assumed that the data has been stripped of its prefix.

    Args:
        data (bytes): Received data to decode.

    Returns:
        Any: Decoded data.
    """
    data: bytes = zlib.decompress(data)
    data: str = data.decode()
    data: Any = json.loads(data)
    return data


def create_client() -> socket.socket:
    """Return a socket for connecting to a server.

    Returns:
        socket.socket: Client socket.
    """
    return socket.socket(socket.AF_INET, socket.SOCK_STREAM)


def send_name(client: socket.socket, name: Optional[str]) -> None:
    """Send message containing a name from the client to the server.

    Args:
        client (socket.socket): Socket of the client.
        name (str): Name the client wants to be identified by.
    """
    if name is None:
        name = OK_MSG.decode()

    data = prefix_data(name.encode())
    try:
        client.sendall(data)
    except Exception:
        print('Unable to send data to server.')
        raise


def send_actions(client: socket.socket, actions: TensorType) -> None:
    """Send the given actions from the client to the server.

    Args:
        client (socket.socket): Socket of the client.
        actions (TensorType): Actions to execute on the server.
    """
    try:
        client.sendall(encode_actions(actions))
    except Exception:
        print('Unable to send data to server.')
        raise


def send_ok(client: socket.socket) -> None:
    """Send an 'OK' message from the client to the server.

    Args:
        client (socket.socket): Socket of the client.
    """
    try:
        client.sendall(OK_MSG)
    except Exception:
        print('Unable to send data to server.')
        raise
