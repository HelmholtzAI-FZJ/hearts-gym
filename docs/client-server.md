<link rel="stylesheet" href="style.css">

# Client-Server Interaction

This document describes how the client and server interact: how
messages are encoded, in which order messages are exchanged and what
rules need to be followed for successful communication.

## Need-to-Knows

The client and server communicate using TCP sockets. TCP packages may
at maximum be 65535 bytes in length.

Each message should be clearly separated from the next one. For this,
we prefix messages of unknown length with their length and the
`hearts_gym.server_utils.MSG_LENGTH_SEPARATOR`. As an additional
method, the client responds with an 'OK' message to the server after
receiving a message. During the game loop, this is mostly not required
as messages are already clearly separated due to the nature of the
game. The exception is upon game end where each client receives
information before a new, random starting player is picked. 'OK'
messages are also used to assert that clients are still able to
communicate.

If we were to process only one environment in parallel, sending one
action and receiving one observation on the client each time,
evaluation would take a long time due to connection speed. That is why
the server processes games in parallel, sending batches of
observations to the clients whose turn it is. Unless the rare case
pops up where it is not a client's turn in any game, all clients
interact with different environments at the same time.

The server sends length-prefixed, gzipped, JSON-encoded messages and
receives non-encoded 'OK' messages as well as length-prefixed actions
that are either:
- A single integer.
- Multiple integers separated by
  `hearts_gym.envs.server_utils.ACTION_SEPARATOR`.
- Only the `hearts_gym.envs.server_utils.ACTION_SEPARATOR`, indicating
  'no' action (used if a client received no observations).

## Order of Communication

This is the order in which communication happens. If communication
results in the server and/or client acting up, you should make sure
you follow this order. The server receiving an 'OK' message is not
explicitly listed.

1. **Server**: Starts up.
2. **Client**: Connects to server.
3. **Client**: Sends name to server (or 'OK' message to use a default
   name).
4. **Server**: Sends hello message.
5. **Client**: Responds with 'OK' message.
6. **Server**: Sends metadata message.
7. **Client**: Responds with 'OK' message.

We now enter a waiting loop for the client if not enough players have
connected. This may repeat indefinitely if no wait timeout is set.

8. **Server**: Sends a 'waiting for players' message.
9. **Client**: Responds with 'OK' message.

When the number of clients changes, the following happens:

10. **Server**: Sends a message updating the number of clients.
11. **Client**: Responds with 'OK' message.

If not enough clients are connected, this jumps back to the waiting
loop afterwards. Once enough clients are connected, the game loop
starts:

12. **Server**: Sends observations. Empty list for clients that have
    none.
13. **Client**: Sends actions in same order. 'No' action (see above)
    if client received no observations.

When the game is over, the following happens:

14. **Server**: Sends final observations.
15. **Client**: Responds with 'OK' message.

The clients are now either disconnected or the game loop starts from
the beginning, depending on server settings.
