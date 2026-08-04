from __future__ import annotations

import logging
import socket

import pytest

from app import egress_policy_gate


class _Clock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class _Connection:
    def __init__(self, response=b"", *, timeout=False):
        self.response = response
        self.timeout = timeout
        self.sent = bytearray()

    def __enter__(self):
        if self.timeout:
            raise socket.timeout
        return self

    def __exit__(self, *_args):
        return None

    def settimeout(self, value):
        self.timeout_value = value

    def sendall(self, data):
        self.sent.extend(data)

    def recv(self, _size):
        response, self.response = self.response, b""
        return response


def test_proxy_probe_is_fixed_credential_free_connect(monkeypatch):
    connection = _Connection(b"HTTP/1.1 200 Connection Established\r\n\r\n")
    monkeypatch.setattr(
        socket,
        "create_connection",
        lambda address, timeout: (assert_equal(address, ("10.0.0.5", 3128)), connection)[1],
    )
    assert egress_policy_gate._proxy_connect_succeeds("10.0.0.5", 3128)
    assert bytes(connection.sent) == (
        b"CONNECT www.bild.de:443 HTTP/1.1\r\n"
        b"Host: www.bild.de:443\r\n"
        b"Connection: close\r\n\r\n"
    )
    assert b"Authorization" not in connection.sent


def assert_equal(actual, expected):
    assert actual == expected


def test_sentinel_timeout_is_the_only_blocked_signal(monkeypatch):
    proxy = _Connection(b"HTTP/1.1 200 Connection Established\r\n\r\n")

    def create_connection(address, timeout):
        assert address == ("10.0.0.5", 3128) or address == ("10.0.0.5", 3129)
        if address[1] == 3128:
            return proxy
        raise socket.timeout

    monkeypatch.setattr(socket, "create_connection", create_connection)
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("10.0.0.5", 3128))
        ],
    )
    assert egress_policy_gate._policy_is_enforced("proxy.synthetic", 3128, 3129)


@pytest.mark.parametrize("error", [ConnectionRefusedError(), OSError(101, "unreachable")])
def test_sentinel_refused_or_unreachable_is_not_policy_evidence(monkeypatch, error):
    proxy = _Connection(b"HTTP/1.1 200 Connection Established\r\n\r\n")

    def create_connection(address, timeout):
        if address[1] == 3128:
            return proxy
        raise error

    monkeypatch.setattr(socket, "create_connection", create_connection)
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("10.0.0.5", 3128))
        ],
    )
    assert not egress_policy_gate._policy_is_enforced("proxy.synthetic", 3128, 3129)


def test_gate_waits_for_three_same_path_drop_rounds(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(egress_policy_gate.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(egress_policy_gate.time, "sleep", clock.sleep)
    rounds = iter([False, True, True, True])
    monkeypatch.setattr(
        egress_policy_gate,
        "_bounded_policy_probe",
        lambda *_args: next(rounds),
    )
    assert egress_policy_gate.wait_until_enforced(
        "proxy.synthetic", 3128, 3129, max_wait_seconds=10
    )


def test_gate_fails_closed_at_deadline(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(egress_policy_gate.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(egress_policy_gate.time, "sleep", clock.sleep)
    def slow_probe(*_args):
        clock.now += 1.0
        return True

    monkeypatch.setattr(egress_policy_gate, "_bounded_policy_probe", slow_probe)
    assert not egress_policy_gate.wait_until_enforced(
        "proxy.synthetic", 3128, 3129, max_wait_seconds=2.5
    )


def test_main_rejects_invalid_or_incomplete_args(monkeypatch):
    monkeypatch.setattr(egress_policy_gate, "wait_until_enforced", lambda *_args: True)
    assert egress_policy_gate.main(["proxy.synthetic", "3128"]) == 1
    assert egress_policy_gate.main(["proxy.synthetic", "3128", "3128"]) == 1
    assert egress_policy_gate.main(["proxy.synthetic", "bad", "3129"]) == 1


def test_main_is_silent_on_failure(monkeypatch, capsys, caplog):
    value = "synthetic-sensitive-proxy.invalid"
    monkeypatch.setattr(
        egress_policy_gate,
        "wait_until_enforced",
        lambda *_args: (_ for _ in ()).throw(RuntimeError(value)),
    )
    with caplog.at_level(logging.DEBUG):
        assert egress_policy_gate.main([value, "3128", "3129"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert value not in caplog.text
