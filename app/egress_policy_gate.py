"""Fail closed until the application pod's cluster egress policy is effective.

The gate compares two ports on the same proxy Service address. The approved
CONNECT port must work, while the proxy's secretless sentinel port must be
dropped by the app's Admin ClusterNetworkPolicy. No HTTP path, TLS payload,
credential, target address, or request log is produced.
"""

from __future__ import annotations

import signal
import socket
import sys
import time
from collections.abc import Sequence


_TARGET_HOST = "www.bild.de"
_MAX_WAIT_SECONDS = 120.0
_REQUIRED_BLOCKED_ROUNDS = 3
_ROUND_INTERVAL_SECONDS = 1.0
_ROUND_TIMEOUT_SECONDS = 8.0
_PROXY_TIMEOUT_SECONDS = 2.0
_SENTINEL_TIMEOUT_SECONDS = 0.75
_MAX_PROXY_RESPONSE_BYTES = 1024
_CONNECTED_STATUS = b"HTTP/1.1 200 Connection Established"
_CONNECT_REQUEST = (
    b"CONNECT www.bild.de:443 HTTP/1.1\r\n"
    b"Host: www.bild.de:443\r\n"
    b"Connection: close\r\n\r\n"
)


class _ProbeTimeout(TimeoutError):
    """A policy probe exceeded its strict per-round deadline."""


def _resolve_service_ipv4(service_host: str, service_port: int) -> tuple[str, ...]:
    try:
        records = socket.getaddrinfo(
            service_host,
            service_port,
            family=socket.AF_INET,
            type=socket.SOCK_STREAM,
            proto=socket.IPPROTO_TCP,
        )
    except OSError:
        return ()
    addresses: list[str] = []
    seen: set[str] = set()
    for family, socktype, proto, _canonname, sockaddr in records:
        if (
            family == socket.AF_INET
            and socktype == socket.SOCK_STREAM
            and proto == socket.IPPROTO_TCP
        ):
            address = sockaddr[0]
            if address not in seen:
                seen.add(address)
                addresses.append(address)
    return tuple(addresses)


def _proxy_connect_succeeds(proxy_address: str, proxy_port: int) -> bool:
    """Return whether the exact, credential-free CONNECT probe receives 200."""
    try:
        with socket.create_connection(
            (proxy_address, proxy_port), timeout=_PROXY_TIMEOUT_SECONDS
        ) as connection:
            connection.settimeout(_PROXY_TIMEOUT_SECONDS)
            connection.sendall(_CONNECT_REQUEST)
            response = bytearray()
            while b"\r\n\r\n" not in response:
                chunk = connection.recv(_MAX_PROXY_RESPONSE_BYTES - len(response))
                if not chunk:
                    return False
                response.extend(chunk)
                if len(response) >= _MAX_PROXY_RESPONSE_BYTES:
                    return False
    except _ProbeTimeout:
        raise
    except OSError:
        return False
    return bytes(response).split(b"\r\n", 1)[0] == _CONNECTED_STATUS


def _sentinel_is_blocked(proxy_address: str, sentinel_port: int) -> bool | None:
    """Return True only for a TCP timeout; other failures are indeterminate."""
    try:
        with socket.create_connection(
            (proxy_address, sentinel_port), timeout=_SENTINEL_TIMEOUT_SECONDS
        ):
            return False
    except _ProbeTimeout:
        raise
    except TimeoutError:
        return True
    except OSError:
        return None


def _policy_is_enforced(proxy_host: str, proxy_port: int, sentinel_port: int) -> bool:
    """Require proxy success and a CNP drop on the same resolved Service IP."""
    for address in _resolve_service_ipv4(proxy_host, proxy_port):
        if not _proxy_connect_succeeds(address, proxy_port):
            continue
        sentinel_state = _sentinel_is_blocked(address, sentinel_port)
        if sentinel_state is True:
            return True
        # Open, refused, unreachable, and every other non-timeout result are
        # not proof of policy enforcement.
        return False
    return False


def _raise_probe_timeout(_signum: int, _frame: object) -> None:
    raise _ProbeTimeout


def _bounded_policy_probe(
    proxy_host: str,
    proxy_port: int,
    sentinel_port: int,
    timeout: float,
) -> bool:
    previous_handler = signal.signal(signal.SIGALRM, _raise_probe_timeout)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        return _policy_is_enforced(proxy_host, proxy_port, sentinel_port)
    except _ProbeTimeout:
        return False
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])


def wait_until_enforced(
    proxy_host: str,
    proxy_port: int,
    sentinel_port: int,
    *,
    max_wait_seconds: float = _MAX_WAIT_SECONDS,
) -> bool:
    """Wait for three consecutive same-path CNP-drop rounds."""
    deadline = time.monotonic() + max_wait_seconds
    consecutive_blocked_rounds = 0
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        enforced = _bounded_policy_probe(
            proxy_host,
            proxy_port,
            sentinel_port,
            min(_ROUND_TIMEOUT_SECONDS, remaining),
        )
        if time.monotonic() >= deadline:
            return False
        if enforced:
            consecutive_blocked_rounds += 1
            if consecutive_blocked_rounds >= _REQUIRED_BLOCKED_ROUNDS:
                return True
        else:
            consecutive_blocked_rounds = 0
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(_ROUND_INTERVAL_SECONDS, remaining))


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 3 or not arguments[0] or "\x00" in arguments[0]:
        return 1
    try:
        proxy_port, sentinel_port = (int(arguments[1]), int(arguments[2]))
    except ValueError:
        return 1
    if not (1 <= proxy_port <= 65535 and 1 <= sentinel_port <= 65535):
        return 1
    if proxy_port == sentinel_port:
        return 1
    try:
        return 0 if wait_until_enforced(arguments[0], proxy_port, sentinel_port) else 1
    except Exception:
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
