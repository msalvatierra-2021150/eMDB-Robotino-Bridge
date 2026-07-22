"""Robotino-specific hardening for the official GII e-MDB main loop.

The upstream synchronous ``ServiceClient`` waits indefinitely for service
responses. During dynamic P/C-node learning, the LTM can apply a neighbor
update and publish the new state even if the corresponding DDS service
response is not delivered. In that case the stock main-loop thread remains
blocked forever.

This adapter keeps the GII learning algorithm unchanged. It only bounds the
wait for the LTM neighbor response and treats the published LTM state as the
authoritative fallback confirmation.
"""

from __future__ import annotations

import time

from cognitive_processes.main_loop import MainLoop
from core.service_client import ServiceClient
from core_interfaces.srv import UpdateNeighbor


class _TimedServiceClient(ServiceClient):
    """GII service client with a bounded response wait."""

    def send_request_with_timeout(
        self,
        timeout_sec: float,
        **kwargs,
    ):
        """Send a request and return ``None`` when its response times out."""
        for key, value in kwargs.items():
            setattr(self.req, key, value)

        self.future = self.cli.call_async(self.req)
        self.se.spin_until_future_complete(
            self.future,
            timeout_sec=max(0.01, float(timeout_sec)),
        )

        if not self.future.done():
            self.future.cancel()
            return None

        return self.future.result()


class RobotinoMainLoop(MainLoop):
    """MainLoop with recovery from a lost LTM neighbor-service response."""

    def __init__(
        self,
        name: str,
        ltm_neighbor_response_timeout_s: float = 2.0,
        ltm_neighbor_confirmation_timeout_s: float = 2.0,
        **params,
    ) -> None:
        # MainLoop starts its worker thread inside super().__init__(), so every
        # attribute used by an overridden method must exist before that call.
        self.ltm_neighbor_response_timeout_s = max(
            0.1,
            float(ltm_neighbor_response_timeout_s),
        )
        self.ltm_neighbor_confirmation_timeout_s = max(
            0.1,
            float(ltm_neighbor_confirmation_timeout_s),
        )

        super().__init__(name, **params)

    def _neighbor_present_in_cache(
        self,
        node_name: str,
        neighbor_name: str,
    ) -> bool:
        """Check the latest state-topic-backed LTM cache safely."""
        semaphore = getattr(self, "semaphore", None)
        acquired = False

        if semaphore is not None:
            acquired = semaphore.acquire(timeout=0.10)
            if not acquired:
                return False

        try:
            for nodes_of_type in self.LTM_cache.values():
                node_data = nodes_of_type.get(node_name)
                if node_data is None:
                    continue

                return any(
                    neighbor.get("name") == neighbor_name
                    for neighbor in node_data.get("neighbors", [])
                )

            return False
        finally:
            if acquired:
                semaphore.release()

    def _wait_for_neighbor_confirmation(
        self,
        node_name: str,
        neighbor_name: str,
    ) -> bool:
        """Wait briefly for the LTM state topic to confirm the update."""
        deadline = (
            time.monotonic()
            + self.ltm_neighbor_confirmation_timeout_s
        )

        while time.monotonic() < deadline:
            if self._neighbor_present_in_cache(node_name, neighbor_name):
                return True
            time.sleep(0.02)

        return self._neighbor_present_in_cache(node_name, neighbor_name)

    def _get_neighbor_client(self, service_name: str) -> _TimedServiceClient:
        """Return the timed client using GII's standard client registry."""
        client = self.node_clients.get(service_name)

        if not isinstance(client, _TimedServiceClient):
            client = _TimedServiceClient(UpdateNeighbor, service_name)
            self.node_clients[service_name] = client

        return client

    def add_neighbor(self, node_name: str, neighbor_name: str) -> bool:
        """Add an LTM neighbor without allowing a lost response to freeze eMDB.

        This follows GII's normal ``add_neighbor`` implementation and stores the
        client in ``self.node_clients``. The only additions are a bounded wait
        and confirmation through the state-topic-backed LTM cache.
        """
        service_name = f"{self.LTM_id}/update_neighbor"
        client = self._get_neighbor_client(service_name)

        response = None
        try:
            response = client.send_request_with_timeout(
                self.ltm_neighbor_response_timeout_s,
                node_name=node_name,
                neighbor_name=neighbor_name,
                operation=True,
            )
        except Exception as exc:  # noqa: BLE001 - ROS exceptions vary.
            self.get_logger().error(
                "LTM neighbor request raised an exception for "
                f"{node_name} -> {neighbor_name}: {exc}"
            )

        if response is not None and bool(response.success):
            return True

        # Recovery for the observed failure: the LTM applied the update and
        # published its state, but the service response was not delivered.
        if self._wait_for_neighbor_confirmation(node_name, neighbor_name):
            self.get_logger().warning(
                "LTM neighbor response was missing or unsuccessful, but the "
                f"state topic confirms {neighbor_name} is a neighbor of "
                f"{node_name}; continuing the eMDB loop."
            )
            return True

        if response is None:
            self.get_logger().error(
                f"Timed out adding LTM neighbor {node_name} -> "
                f"{neighbor_name}, and the update was not confirmed in the "
                "LTM state cache."
            )
        else:
            self.get_logger().error(
                f"LTM rejected neighbor update {node_name} -> "
                f"{neighbor_name}."
            )

        return False
