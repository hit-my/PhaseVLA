import asyncio
import http
import logging
import time
import traceback

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()
        policy = self._policy.fork()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())

                if isinstance(obs, dict) and obs.get("reset") is True:
                    reset_time = time.monotonic()
                    policy.reset()
                    reset_time = time.monotonic() - reset_time
                    await websocket.send(
                        packer.pack({"reset_finished": True, "reset_time_ms": reset_time * 1000})
                    )
                    prev_total_time = time.monotonic() - start_time
                    continue
                if isinstance(obs, dict) and obs.get("__futuremamba_memory_only__") is True:
                    memory_payload = dict(obs)
                    memory_payload.pop("__futuremamba_memory_only__", None)
                    update_memory = getattr(policy, "update_memory_only", None)
                    if not callable(update_memory):
                        raise RuntimeError("policy does not support memory-only updates")
                    update_time = time.monotonic()
                    response = update_memory(memory_payload)
                    response["memory_update_time_ms"] = (
                        time.monotonic() - update_time
                    ) * 1000
                    await websocket.send(packer.pack(response))
                    prev_total_time = time.monotonic() - start_time
                    continue

                if isinstance(obs, dict) and obs.get("add_buffer") is True:
                    buffer_time = time.monotonic()
                    add_buffer = getattr(policy, "add_buffer", None)
                    if callable(add_buffer):
                        add_buffer(obs)
                    buffer_time = time.monotonic() - buffer_time
                    await websocket.send(
                        packer.pack({"add_buffer_finished": True, "add_buffer_time_ms": buffer_time * 1000})
                    )
                    prev_total_time = time.monotonic() - start_time
                    continue


                if isinstance(obs, dict) and "__openpi_control__" in obs:
                    response = _handle_control(policy, obs["__openpi_control__"])
                    await websocket.send(packer.pack(response))
                    prev_total_time = time.monotonic() - start_time
                    continue

                infer_time = time.monotonic()
                action = policy.infer(obs)
                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _handle_control(policy: _base_policy.BasePolicy, control: object) -> dict:
    if control == "reset":
        policy.reset()
        return {"reset": True}
    return {"error": "unknown_control", "message": f"Unknown control message: {control}"}


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
