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

MAX_PACKAGE_SIZE = 65535
OK_MSG = b'__OK'

ACTION_SEPARATOR = b','


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
    return ACTION_SEPARATOR.join(map(encode_int, actions))


def decode_actions(data: bytes) -> List[Action]:
    """Parse actions from the given message data.

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
        bytes: Encoded data.
    """
    data: str = json.dumps(data, separators=(',', ':'))
    data: bytes = data.encode()
    data: bytes = zlib.compress(data)
    return data


def decode_data(data: bytes) -> Any:
    """Return the given data decoded from a message from server
    to client.

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
        send_ok(client)
        return

    try:
        client.sendall(name.encode())
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
