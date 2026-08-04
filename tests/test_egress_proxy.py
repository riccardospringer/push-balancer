import asyncio
import logging
import socket
import threading
import urllib.request

import pytest

from app import egress_proxy
from app import config
from app.cms import url_api
from app.routers import feed


ALLOWED = frozenset({"url-server.example.invalid", "sitemap.example.invalid"})


def _request(authority: str, headers: bytes = b"") -> bytes:
    return b"CONNECT " + authority.encode("ascii") + b" HTTP/1.1\r\n" + headers + b"\r\n"


def test_load_allowed_hosts_is_exact_and_fail_closed():
    assert egress_proxy.load_allowed_hosts(
        " API.STG.EDITORIAL.ONE,push-balancer.onrender.com,www.bild.de "
    ) == frozenset(
        {"api.stg.editorial.one", "push-balancer.onrender.com", "www.bild.de"}
    )

    for invalid in (
        "",
        "*.example.invalid",
        "example.invalid.",
        "127.0.0.1",
        "unapproved.example.invalid",
    ):
        with pytest.raises(ValueError):
            egress_proxy.load_allowed_hosts(invalid)


def test_connect_parser_accepts_only_allowlisted_dns_hosts_on_443():
    assert (
        egress_proxy.parse_connect_request(_request("URL-SERVER.example.invalid:443"), ALLOWED)
        == "url-server.example.invalid"
    )

    with pytest.raises(egress_proxy.ForbiddenConnectTarget):
        egress_proxy.parse_connect_request(_request("other.example.invalid:443"), ALLOWED)

    for authority in (
        "url-server.example.invalid:80",
        "url-server.example.invalid.:443",
        "127.0.0.1:443",
        "user@url-server.example.invalid:443",
    ):
        with pytest.raises(egress_proxy.InvalidConnectRequest):
            egress_proxy.parse_connect_request(_request(authority), ALLOWED)


@pytest.mark.parametrize(
    "raw_request",
    [
        b"GET https://url-server.example.invalid/ HTTP/1.1\r\n\r\n",
        _request("url-server.example.invalid:443", b" Transfer-Encoding: chunked\r\n"),
        _request("url-server.example.invalid:443", b"Transfer-Encoding: chunked\r\n"),
        _request("url-server.example.invalid:443", b"Proxy-Authorization: secret\r\n"),
        _request("url-server.example.invalid:443", b"Content-Length: 1\r\n"),
        _request(
            "url-server.example.invalid:443",
            b"Host: other.example.invalid:443\r\n",
        ),
        _request(
            "url-server.example.invalid:443",
            b"Host: url-server.example.invalid:443\r\nHost: url-server.example.invalid:443\r\n",
        ),
    ],
)
def test_connect_parser_rejects_smuggling_and_credentials(raw_request):
    with pytest.raises(egress_proxy.InvalidConnectRequest):
        egress_proxy.parse_connect_request(raw_request, ALLOWED)


@pytest.mark.parametrize(
    "address",
    (
        "0.0.0.0",
        "10.1.2.3",
        "100.64.0.1",
        "127.0.0.1",
        "169.254.169.254",
        "172.16.0.1",
        "192.0.2.1",
        "192.168.0.1",
        "198.18.0.1",
        "198.51.100.1",
        "203.0.113.1",
        "224.0.0.1",
        "192.0.0.9",
        "192.31.196.1",
        "192.52.193.1",
        "192.88.99.1",
        "192.175.48.1",
        "::",
        "::1",
        "::10.0.0.1",
        "::ffff:0:10.0.0.1",
        "::ffff:127.0.0.1",
        "::ffff:8.8.8.8",
        "64:ff9b::10.0.0.1",
        "64:ff9b::a00:1",
        "64:ff9b:1::a00:1",
        "100:0:0:1::1",
        "2001:0000:4136:e378:8000:63bf:3fff:fdd2",
        "2002:a00:1::",
        "2002:0a00:1::1",
        "2001:db8::1",
        "2606:4700:4700::1111",
        "5f00::1",
        "::ffff:10.0.0.1",
        "fc00::1",
        "fec0::1",
        "fe80::1",
        "ff02::1",
        "not-an-address",
    ),
)
def test_private_and_special_network_addresses_are_not_connectable(address):
    assert egress_proxy._address_is_forbidden(address)


@pytest.mark.parametrize("address", ("8.8.8.8", "1.1.1.1"))
def test_globally_routable_ipv4_addresses_are_connectable(address):
    assert not egress_proxy._address_is_forbidden(address)


@pytest.mark.parametrize("network", egress_proxy._FORBIDDEN_IPV4_NETWORKS)
def test_special_ipv4_network_boundaries_are_not_connectable(network):
    assert egress_proxy._address_is_forbidden(str(network.network_address))
    assert egress_proxy._address_is_forbidden(str(network.broadcast_address))


def test_resolution_keeps_only_globally_routable_addresses(monkeypatch):
    class Loop:
        async def getaddrinfo(self, host, port, *, family, type, proto):
            assert host == "api.stg.editorial.one"
            assert port == 443
            assert family == socket.AF_INET
            assert type == socket.SOCK_STREAM
            assert proto == socket.IPPROTO_TCP
            return [
                (
                    socket.AF_INET,
                    socket.SOCK_STREAM,
                    socket.IPPROTO_TCP,
                    "",
                    ("10.0.0.1", 443),
                ),
                (
                    socket.AF_INET,
                    socket.SOCK_STREAM,
                    socket.IPPROTO_TCP,
                    "",
                    ("8.8.8.8", 443),
                ),
                (
                    socket.AF_INET6,
                    socket.SOCK_STREAM,
                    socket.IPPROTO_TCP,
                    "",
                    ("fc00::1", 443, 0, 0),
                ),
                (
                    socket.AF_INET6,
                    socket.SOCK_STREAM,
                    socket.IPPROTO_TCP,
                    "",
                    ("2606:4700:4700::1111", 443, 0, 0),
                ),
            ]

    monkeypatch.setattr(asyncio, "get_running_loop", lambda: Loop())

    assert asyncio.run(egress_proxy._resolve_addresses("api.stg.editorial.one")) == [
        (socket.AF_INET, "8.8.8.8"),
    ]


def test_rejected_request_does_not_log_target_or_headers(monkeypatch, caplog):
    synthetic_secret = "synthetic-proxy-secret"

    async def exercise():
        proxy = egress_proxy.StrictConnectProxy(ALLOWED)
        server = await asyncio.start_server(proxy.handle_client, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        async with server:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(
                _request(
                    "blocked.example.invalid:443",
                    f"X-Synthetic: {synthetic_secret}\r\n".encode("ascii"),
                )
            )
            await writer.drain()
            response = await reader.read()
            writer.close()
            await writer.wait_closed()
        return response

    with caplog.at_level(logging.DEBUG):
        response = asyncio.run(exercise())
    assert response.startswith(b"HTTP/1.1 403")
    assert synthetic_secret not in caplog.text
    assert "blocked.example.invalid" not in caplog.text


def test_proxy_tunnels_only_after_approved_resolution(monkeypatch):
    observed: list[str] = []

    async def fake_connect(host: str):
        observed.append(host)
        upstream_reads = asyncio.StreamReader()
        upstream_reads.feed_data(b"upstream-response")
        upstream_reads.feed_eof()

        class Writer:
            def __init__(self):
                self.data = bytearray()

            def write(self, data):
                self.data.extend(data)

            async def drain(self):
                return None

            def can_write_eof(self):
                return False

            def close(self):
                return None

            async def wait_closed(self):
                return None

        return upstream_reads, Writer()

    monkeypatch.setattr(egress_proxy, "_connect_upstream", fake_connect)

    async def exercise():
        proxy = egress_proxy.StrictConnectProxy(ALLOWED)
        server = await asyncio.start_server(proxy.handle_client, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        async with server:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(_request("url-server.example.invalid:443"))
            await writer.drain()
            response = await reader.read()
            writer.close()
            await writer.wait_closed()
        return response

    response = asyncio.run(exercise())
    assert response.startswith(b"HTTP/1.1 200 Connection Established")
    assert response.endswith(b"upstream-response")
    assert observed == ["url-server.example.invalid"]


def test_connect_uses_next_address_without_waiting_for_blackhole(monkeypatch):
    async def fake_resolve(_host):
        return [
            (socket.AF_INET, "203.0.113.1"),
            (socket.AF_INET, "203.0.113.2"),
        ]

    successful_result = (object(), object())

    async def fake_open_connection(address, _port, family, flags):
        assert family == socket.AF_INET
        assert flags == socket.AI_NUMERICHOST
        if address == "203.0.113.1":
            await asyncio.Event().wait()
        return successful_result

    monkeypatch.setattr(egress_proxy, "_resolve_addresses", fake_resolve)
    monkeypatch.setattr(asyncio, "open_connection", fake_open_connection)

    assert asyncio.run(egress_proxy._connect_upstream("api.stg.editorial.one")) is successful_result


def _record_one_connect(run_client) -> bytes:
    recorded = bytearray()
    ready = threading.Event()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]

        def accept_once():
            ready.set()
            connection, _address = server.accept()
            with connection:
                while b"\r\n\r\n" not in recorded and len(recorded) <= 8192:
                    chunk = connection.recv(4096)
                    if not chunk:
                        break
                    recorded.extend(chunk)
                connection.sendall(
                    b"HTTP/1.1 403 Forbidden\r\nConnection: close\r\nContent-Length: 0\r\n\r\n"
                )

        worker = threading.Thread(target=accept_once, daemon=True)
        worker.start()
        assert ready.wait(timeout=1)
        run_client(port)
        worker.join(timeout=2)
        assert not worker.is_alive()
    return bytes(recorded)


def test_real_url_api_and_sitemap_clients_issue_approved_connect(monkeypatch):
    def configure_proxy(port):
        proxy_url = f"http://127.0.0.1:{port}"
        monkeypatch.setenv("HTTPS_PROXY", proxy_url)
        monkeypatch.setenv("https_proxy", proxy_url)
        monkeypatch.delenv("NO_PROXY", raising=False)
        monkeypatch.delenv("no_proxy", raising=False)

    def call_url_api(port):
        configure_proxy(port)
        monkeypatch.setattr(config, "URL_API_BASE", "https://api.stg.editorial.one/urlapi/v2")
        monkeypatch.setattr(config, "URL_API_KEY", "synthetic-api-key")
        with pytest.raises(url_api.UrlApiUnavailable):
            url_api.get_canonical_article_url("synthetic.cms.id")

    url_api_connect = _record_one_connect(call_url_api)
    assert (
        egress_proxy.parse_connect_request(url_api_connect, egress_proxy._PERMITTED_HOSTS)
        == "api.stg.editorial.one"
    )

    def call_sitemap(port):
        configure_proxy(port)
        monkeypatch.setattr(urllib.request, "_opener", None)
        feed._url_cache.clear()
        assert feed._fetch_url("https://www.bild.de/synthetic-sitemap.xml") is None

    sitemap_connect = _record_one_connect(call_sitemap)
    assert (
        egress_proxy.parse_connect_request(sitemap_connect, egress_proxy._PERMITTED_HOSTS)
        == "www.bild.de"
    )
