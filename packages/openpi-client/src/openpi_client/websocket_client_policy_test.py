import threading
import time

from openpi_client import msgpack_numpy
from openpi_client import websocket_client_policy


class _FakeConnection:
    def __init__(self):
        self.sent = []
        self.recv_count = 0
        self.metadata = {"server": "metadata"}
        self.lock = threading.Lock()
        self.in_exchange = False
        self.max_concurrent_exchanges = 0

    def send(self, payload):
        with self.lock:
            assert not self.in_exchange
            self.in_exchange = True
            self.max_concurrent_exchanges += 1
            self.sent.append(msgpack_numpy.unpackb(payload))
        time.sleep(0.02)

    def recv(self):
        if self.recv_count == 0:
            self.recv_count += 1
            return msgpack_numpy.packb(self.metadata)
        time.sleep(0.02)
        with self.lock:
            request = self.sent[-1]
            self.in_exchange = False
        self.recv_count += 1
        if request == {"__openpi_control__": "reset"}:
            return msgpack_numpy.packb({"reset": True})
        return msgpack_numpy.packb({"actions": request["obs"]})


def test_reset_uses_same_connection_and_waits_for_ack(monkeypatch):
    conn = _FakeConnection()
    monkeypatch.setattr(websocket_client_policy.websockets.sync.client, "connect", lambda *args, **kwargs: conn)
    policy = websocket_client_policy.WebsocketClientPolicy("127.0.0.1", 8000)

    policy.reset()

    assert policy.get_server_metadata() == {"server": "metadata"}
    assert conn.sent == [{"__openpi_control__": "reset"}]


def test_infer_and_reset_share_mutex_so_responses_cannot_interleave(monkeypatch):
    conn = _FakeConnection()
    monkeypatch.setattr(websocket_client_policy.websockets.sync.client, "connect", lambda *args, **kwargs: conn)
    policy = websocket_client_policy.WebsocketClientPolicy("127.0.0.1", 8000)
    results = []

    threads = [
        threading.Thread(target=lambda: results.append(policy.infer({"obs": 3}))),
        threading.Thread(target=policy.reset),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert results == [{"actions": 3}]
    assert {"obs": 3} in conn.sent
    assert {"__openpi_control__": "reset"} in conn.sent
    assert conn.max_concurrent_exchanges == len(conn.sent)
