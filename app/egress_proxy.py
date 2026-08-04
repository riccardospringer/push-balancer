"""Small fail-closed CONNECT proxy for the score runtime's approved HTTPS targets.

The proxy deliberately has no per-request logging.  It only establishes raw TLS
tunnels after validating an exact hostname allowlist and port 443; TLS remains
end-to-end between the application client and the upstream service.
"""

from __future__ import annotations

import asyncio
import ipaddress
import os
import re
import socket
from collections.abc import Iterable


_AUTHORITY_RE = re.compile(r"^(?P<host>[A-Za-z0-9.-]{1,253}):443$")
_HEADER_NAME_RE = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")
_FORBIDDEN_HEADERS = {"proxy-authorization", "transfer-encoding", "upgrade"}

_MAX_HEADER_BYTES = 8 * 1024
_MAX_HEADER_FIELDS = 40
_HEADER_TIMEOUT_SECONDS = 3.0
_CONNECT_TIMEOUT_SECONDS = 5.0
_IDLE_TIMEOUT_SECONDS = 30.0
_MAX_TUNNEL_SECONDS = 120.0
_MAX_BYTES_PER_DIRECTION = 32 * 1024 * 1024
_CHUNK_BYTES = 64 * 1024
_MAX_CONNECTIONS = 32
_PERMITTED_HOSTS = frozenset(
    {
        "api.stg.editorial.one",
        "api.editorial.one",
        "push-balancer.onrender.com",
        "www.bild.de",
    }
)

# This conservative IANA special-purpose snapshot is explicit so the decision
# does not change across supported Python patch versions. Some entries are
# intentionally globally reachable infrastructure mechanisms; none is a valid
# destination for the approved application upstreams.
_FORBIDDEN_IPV4_NETWORKS = tuple(
    ipaddress.ip_network(network)
    for network in (
        "0.0.0.0/8",
        "10.0.0.0/8",
        "100.64.0.0/10",
        "127.0.0.0/8",
        "169.254.0.0/16",
        "172.16.0.0/12",
        "192.0.0.0/24",  # IETF protocol assignments and anycast services
        "192.0.2.0/24",
        "192.31.196.0/24",  # AS112-v4
        "192.52.193.0/24",  # AMT
        "192.88.99.0/24",  # deprecated 6to4 relay anycast
        "192.168.0.0/16",
        "192.175.48.0/24",  # AS112 direct delegation
        "198.18.0.0/15",
        "198.51.100.0/24",
        "203.0.113.0/24",
        "224.0.0.0/4",
        "240.0.0.0/4",
    )
)

_RESPONSES = {
    400: b"HTTP/1.1 400 Bad Request\r\nConnection: close\r\nContent-Length: 0\r\n\r\n",
    403: b"HTTP/1.1 403 Forbidden\r\nConnection: close\r\nContent-Length: 0\r\n\r\n",
    502: b"HTTP/1.1 502 Bad Gateway\r\nConnection: close\r\nContent-Length: 0\r\n\r\n",
    503: b"HTTP/1.1 503 Service Unavailable\r\nConnection: close\r\nContent-Length: 0\r\n\r\n",
}
_CONNECTED = b"HTTP/1.1 200 Connection Established\r\n\r\n"


class InvalidConnectRequest(ValueError):
    """The request is malformed and must be rejected without forwarding."""


class ForbiddenConnectTarget(PermissionError):
    """The request is valid CONNECT syntax but its target is not approved."""


def _validate_hostname(host: str) -> str:
    normalized = host.lower()
    if (
        not normalized
        or normalized.endswith(".")
        or ".." in normalized
        or normalized.startswith("-")
        or any(label.startswith("-") or label.endswith("-") for label in normalized.split("."))
    ):
        raise InvalidConnectRequest("invalid hostname")
    try:
        ipaddress.ip_address(normalized)
    except ValueError:
        return normalized
    raise InvalidConnectRequest("IP literals are not accepted")


def load_allowed_hosts(raw: str | None = None) -> frozenset[str]:
    """Load and strictly validate the exact hostname allowlist."""
    source = os.environ.get("EGRESS_PROXY_ALLOWED_HOSTS", "") if raw is None else raw
    hosts: set[str] = set()
    for item in source.split(","):
        item = item.strip()
        if not item:
            continue
        if not re.fullmatch(r"[A-Za-z0-9.-]{1,253}", item):
            raise ValueError("EGRESS_PROXY_ALLOWED_HOSTS contains an invalid hostname")
        hosts.add(_validate_hostname(item))
    if not hosts:
        raise ValueError("EGRESS_PROXY_ALLOWED_HOSTS must contain at least one hostname")
    if not hosts.issubset(_PERMITTED_HOSTS):
        raise ValueError("EGRESS_PROXY_ALLOWED_HOSTS contains an unapproved hostname")
    return frozenset(hosts)


def parse_connect_request(data: bytes, allowed_hosts: frozenset[str]) -> str:
    """Return the approved target hostname or raise a generic rejection type."""
    if len(data) > _MAX_HEADER_BYTES or not data.endswith(b"\r\n\r\n"):
        raise InvalidConnectRequest("invalid header block")
    try:
        text = data.decode("ascii")
    except UnicodeDecodeError as exc:
        raise InvalidConnectRequest("headers must be ASCII") from exc

    lines = text[:-4].split("\r\n")
    if not lines or len(lines) - 1 > _MAX_HEADER_FIELDS:
        raise InvalidConnectRequest("invalid number of header fields")
    request_parts = lines[0].split(" ")
    if len(request_parts) != 3 or "" in request_parts:
        raise InvalidConnectRequest("invalid request line")
    method, authority, version = request_parts
    if method != "CONNECT" or version not in {"HTTP/1.0", "HTTP/1.1"}:
        raise InvalidConnectRequest("only CONNECT is accepted")

    match = _AUTHORITY_RE.fullmatch(authority)
    if not match:
        raise InvalidConnectRequest("only DNS hostnames on port 443 are accepted")
    host = _validate_hostname(match.group("host"))

    seen_host = False
    for line in lines[1:]:
        if not line or line[0] in " \t" or ":" not in line:
            raise InvalidConnectRequest("invalid header field")
        name, value = line.split(":", 1)
        if not _HEADER_NAME_RE.fullmatch(name):
            raise InvalidConnectRequest("invalid header name")
        if "\x00" in value or "\r" in value or "\n" in value:
            raise InvalidConnectRequest("invalid header value")
        lower_name = name.lower()
        stripped_value = value.strip(" \t")
        if lower_name in _FORBIDDEN_HEADERS:
            raise InvalidConnectRequest("forbidden proxy header")
        if lower_name == "content-length" and stripped_value != "0":
            raise InvalidConnectRequest("CONNECT request body is not accepted")
        if lower_name == "host":
            if seen_host or stripped_value.lower() != authority.lower():
                raise InvalidConnectRequest("invalid Host header")
            seen_host = True

    if host not in allowed_hosts:
        raise ForbiddenConnectTarget("target is not allowlisted")
    return host


def _address_is_forbidden(address: str) -> bool:
    """Reject every address that is not approved public IPv4 unicast.

    The permitted upstreams are public HTTPS endpoints.  Refusing private and
    special-use results prevents a compromised or rebound DNS answer from
    turning the CONNECT proxy into a path to VPC, cluster, CGNAT, metadata, or
    documentation networks. IPv6 is deliberately fail-closed: the approved
    upstreams publish public IPv4 records and older supported Python patch
    versions classify several IPv6 transition and special-use ranges
    inconsistently.
    """
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return True
    return (
        not isinstance(ip, ipaddress.IPv4Address)
        or not ip.is_global
        or ip.is_multicast
        or ip.is_reserved
        or any(ip in network for network in _FORBIDDEN_IPV4_NETWORKS)
    )


async def _resolve_addresses(host: str) -> list[tuple[int, str]]:
    loop = asyncio.get_running_loop()
    records = await loop.getaddrinfo(
        host,
        443,
        family=socket.AF_INET,
        type=socket.SOCK_STREAM,
        proto=socket.IPPROTO_TCP,
    )
    addresses: list[tuple[int, str]] = []
    seen: set[tuple[int, str]] = set()
    for family, _socktype, _proto, _canonname, sockaddr in records[:16]:
        if family != socket.AF_INET:
            continue
        candidate = (family, sockaddr[0])
        if candidate in seen or _address_is_forbidden(candidate[1]):
            continue
        seen.add(candidate)
        addresses.append(candidate)
    return addresses


async def _connect_upstream(host: str) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    async def connect_after_delay(
        family: int,
        address: str,
        delay: float,
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        if delay:
            await asyncio.sleep(delay)
        return await asyncio.open_connection(
            address,
            443,
            family=family,
            flags=socket.AI_NUMERICHOST,
        )

    async with asyncio.timeout(_CONNECT_TIMEOUT_SECONDS):
        addresses = await _resolve_addresses(host)
        tasks = [
            asyncio.create_task(connect_after_delay(family, address, index * 0.2))
            for index, (family, address) in enumerate(addresses)
        ]
        winner: tuple[asyncio.StreamReader, asyncio.StreamWriter] | None = None
        try:
            for completed in asyncio.as_completed(tasks):
                try:
                    winner = await completed
                    return winner
                except OSError:
                    continue
        finally:
            for task in tasks:
                task.cancel()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, tuple) and result is not winner:
                    await _close_writer(result[1])
    raise OSError("approved upstream is unavailable")


async def _pipe(
    source: asyncio.StreamReader,
    destination: asyncio.StreamWriter,
) -> None:
    transferred = 0
    while True:
        data = await asyncio.wait_for(
            source.read(_CHUNK_BYTES),
            timeout=_IDLE_TIMEOUT_SECONDS,
        )
        if not data:
            if destination.can_write_eof():
                destination.write_eof()
                await destination.drain()
            return
        transferred += len(data)
        if transferred > _MAX_BYTES_PER_DIRECTION:
            raise OSError("tunnel byte limit reached")
        destination.write(data)
        await destination.drain()


async def _close_writer(writer: asyncio.StreamWriter | None) -> None:
    if writer is None:
        return
    writer.close()
    try:
        await writer.wait_closed()
    except OSError:
        pass


class StrictConnectProxy:
    """Async CONNECT server with a small, fixed resource envelope."""

    def __init__(self, allowed_hosts: Iterable[str], max_connections: int = _MAX_CONNECTIONS):
        self.allowed_hosts = frozenset(allowed_hosts)
        if not self.allowed_hosts:
            raise ValueError("at least one allowed host is required")
        self.max_connections = max_connections
        self._active_connections = 0

    async def handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        if self._active_connections >= self.max_connections:
            writer.write(_RESPONSES[503])
            await writer.drain()
            await _close_writer(writer)
            return

        self._active_connections += 1
        upstream_writer: asyncio.StreamWriter | None = None
        connected = False
        try:
            request = await asyncio.wait_for(
                reader.readuntil(b"\r\n\r\n"),
                timeout=_HEADER_TIMEOUT_SECONDS,
            )
            host = parse_connect_request(request, self.allowed_hosts)
            upstream_reader, upstream_writer = await _connect_upstream(host)
            writer.write(_CONNECTED)
            await writer.drain()
            connected = True
            client_to_upstream = asyncio.create_task(_pipe(reader, upstream_writer))
            upstream_to_client = asyncio.create_task(_pipe(upstream_reader, writer))
            tasks = (client_to_upstream, upstream_to_client)
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks),
                    timeout=_MAX_TUNNEL_SECONDS,
                )
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
        except ForbiddenConnectTarget:
            writer.write(_RESPONSES[403])
            await writer.drain()
        except (InvalidConnectRequest, asyncio.IncompleteReadError, asyncio.LimitOverrunError):
            writer.write(_RESPONSES[400])
            await writer.drain()
        except (TimeoutError, OSError):
            if not connected:
                writer.write(_RESPONSES[502])
                await writer.drain()
        finally:
            await _close_writer(upstream_writer)
            await _close_writer(writer)
            self._active_connections -= 1


async def _handle_policy_sentinel(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> None:
    """Hold a silent secretless TCP policy probe until the client closes it."""
    try:
        await asyncio.wait_for(reader.read(1), timeout=30.0)
    except (TimeoutError, asyncio.IncompleteReadError, OSError):
        pass
    finally:
        await _close_writer(writer)


def _load_listener_ports() -> tuple[int, int]:
    proxy_port = int(os.environ.get("EGRESS_PROXY_PORT", "3128"))
    sentinel_port = int(os.environ.get("EGRESS_PROXY_SENTINEL_PORT", "3129"))
    if not 1 <= proxy_port <= 65535:
        raise ValueError("EGRESS_PROXY_PORT must be a valid TCP port")
    if not 1 <= sentinel_port <= 65535:
        raise ValueError("EGRESS_PROXY_SENTINEL_PORT must be a valid TCP port")
    if proxy_port == sentinel_port:
        raise ValueError("proxy and policy sentinel ports must be different")
    return proxy_port, sentinel_port


async def serve() -> None:
    allowed_hosts = load_allowed_hosts()
    bind_host = os.environ.get("EGRESS_PROXY_BIND_HOST", "0.0.0.0")
    proxy_port, sentinel_port = _load_listener_ports()
    proxy = StrictConnectProxy(allowed_hosts)
    proxy_server = await asyncio.start_server(
        proxy.handle_client,
        bind_host,
        proxy_port,
        limit=_MAX_HEADER_BYTES,
        start_serving=False,
    )
    try:
        sentinel_server = await asyncio.start_server(
            _handle_policy_sentinel,
            bind_host,
            sentinel_port,
            start_serving=False,
        )
    except BaseException:
        proxy_server.close()
        await proxy_server.wait_closed()
        raise

    async with proxy_server, sentinel_server:
        await asyncio.gather(
            proxy_server.serve_forever(),
            sentinel_server.serve_forever(),
        )


if __name__ == "__main__":
    asyncio.run(serve())
