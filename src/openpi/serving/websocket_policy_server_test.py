import asyncio
import threading

from openpi_client import msgpack_numpy
import pytest
import websockets.sync.client

from openpi.serving import websocket_policy_server


class _SessionPolicy:
    def __init__(self, name):
        self.name = name
        self.value = 0
        self.reset_count = 0

        self.buffer_boundaries = []

    def add_buffer(self, payload):
        self.buffer_boundaries.append(payload)
    def infer(self, obs):
        self.value += int(obs.get("delta", 1))
        return {"session": self.name, "value": self.value}

    def reset(self):
        self.reset_count += 1
        self.value = 0
    def snapshot_state(self):
        return {"memory": self.value, "executed_action_mask": [False]}


class _ForkingPolicy:
    def __init__(self):
        self.sessions = []

    def infer(self, obs):
        raise AssertionError("server must use per-connection fork")

    def fork(self):
        session = _SessionPolicy(f"session-{len(self.sessions)}")
        self.sessions.append(session)
        return session


@pytest.fixture
def policy_server():
    policy = _ForkingPolicy()
    server = websocket_policy_server.WebsocketPolicyServer(policy, host="127.0.0.1", port=0, metadata={"m": 1})
    ready = threading.Event()
    stop = threading.Event()
    port_holder = {}

    async def run():
        async with websocket_policy_server._server.serve(
            server._handler,
            "127.0.0.1",
            0,
            compression=None,
            max_size=None,
            process_request=websocket_policy_server._health_check,
        ) as ws_server:
            port_holder["port"] = ws_server.sockets[0].getsockname()[1]
            ready.set()
            while not stop.is_set():
                await asyncio.sleep(0.01)

    thread = threading.Thread(target=lambda: asyncio.run(run()), daemon=True)
    thread.start()
    assert ready.wait(timeout=5)
    yield policy, port_holder["port"]
    stop.set()
    thread.join(timeout=5)


def _connect(port):
    ws = websockets.sync.client.connect(f"ws://127.0.0.1:{port}", compression=None, max_size=None)
    metadata = msgpack_numpy.unpackb(ws.recv())
    assert metadata == {"m": 1}
    return ws


def _request(ws, payload):
    ws.send(msgpack_numpy.packb(payload))
    response = ws.recv()
    assert not isinstance(response, str)
    return msgpack_numpy.unpackb(response)


def test_each_connection_uses_independent_forked_policy_and_reset_is_local(policy_server):
    policy, port = policy_server
    a = _connect(port)
    b = _connect(port)

    assert _request(a, {"delta": 2})["value"] == 2
    assert _request(b, {"delta": 5})["value"] == 5
    assert _request(a, {"__openpi_control__": "reset"}) == {"reset": True}
    assert policy.sessions[0].snapshot_state() == {"memory": 0, "executed_action_mask": [False]}
    assert _request(a, {"delta": 1})["value"] == 1
    assert _request(b, {"delta": 1})["value"] == 6
    assert len(policy.sessions) == 2
    assert policy.sessions[0].reset_count == 1
    assert policy.sessions[1].reset_count == 0

    a.close()
    b.close()



def test_legacy_robomme_reset_payload_returns_expected_ack(policy_server):
    policy, port = policy_server
    ws = _connect(port)

    response = _request(ws, {"reset": True})

    assert response["reset_finished"] is True
    assert response["reset_time_ms"] >= 0.0
    assert policy.sessions[0].reset_count == 1
    ws.close()


def test_legacy_robomme_add_buffer_acknowledges_without_advancing_policy(policy_server):
    policy, port = policy_server
    ws = _connect(port)
    before = policy.sessions[0].snapshot_state()

    response = _request(ws, {"add_buffer": True, "buffer": {"query_index": 3}})

    assert response["add_buffer_finished"] is True
    assert response["add_buffer_time_ms"] >= 0.0
    assert policy.sessions[0].snapshot_state() == before
    assert policy.sessions[0].buffer_boundaries == [{"add_buffer": True, "buffer": {"query_index": 3}}]
    ws.close()

def test_unknown_control_message_returns_structured_error(policy_server):
    _, port = policy_server
    ws = _connect(port)

    response = _request(ws, {"__openpi_control__": "bogus"})

    assert response["error"] == "unknown_control"
    assert "bogus" in response["message"]
    ws.close()
