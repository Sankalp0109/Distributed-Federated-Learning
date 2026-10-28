"""Wire protocol: length-prefixed pickled dicts over TCP, with byte accounting.

Every message is a dict with a "type" field. Frames are a 4-byte big-endian
length followed by the pickle payload, so the byte counters below measure
exactly what goes over the socket (header included).
"""
import pickle
import socket
import struct
import threading

# Message types
REGISTER = "REGISTER"          # worker -> coord: {worker_id, num_samples}
REGISTER_ACK = "REGISTER_ACK"  # coord -> worker
TRAIN = "TRAIN"                # coord -> worker: {round_id, weights, epochs, lr, batch_size, deadline_s}
UPDATE = "UPDATE"              # worker -> coord: {round_id, worker_id, weights, num_samples, train_loss, train_time}
REJECT = "REJECT"              # coord -> worker: {round_id, reason}  (late/stale update)
SHUTDOWN = "SHUTDOWN"          # coord -> worker
DISCONNECT = "DISCONNECT"      # internal only: reader thread saw the connection drop

_HEADER = struct.Struct("!I")


class Channel:
    """A framed, thread-safe-for-send message channel over one TCP socket."""

    def __init__(self, sock: socket.socket):
        self.sock = sock
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.bytes_sent = 0
        self.bytes_recv = 0
        self._send_lock = threading.Lock()

    @classmethod
    def connect(cls, host: str, port: int, timeout: float = 10.0) -> "Channel":
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.settimeout(None)
        return cls(sock)

    def send(self, msg: dict) -> int:
        payload = pickle.dumps(msg, protocol=pickle.HIGHEST_PROTOCOL)
        frame = _HEADER.pack(len(payload)) + payload
        with self._send_lock:
            self.sock.sendall(frame)
            self.bytes_sent += len(frame)
        return len(frame)

    def recv(self) -> dict:
        (n,) = _HEADER.unpack(self._recv_exact(_HEADER.size))
        payload = self._recv_exact(n)
        self.bytes_recv += _HEADER.size + n
        return pickle.loads(payload)

    def _recv_exact(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = self.sock.recv(min(n - len(buf), 1 << 20))
            if not chunk:
                raise ConnectionError("peer closed connection")
            buf.extend(chunk)
        return bytes(buf)

    def close(self):
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.sock.close()
