"""Synthetic checks for the provenance-preserving Mac-to-Render relay."""

from __future__ import annotations

import json


def test_relay_forwards_original_authoritative_snapshot(monkeypatch):
    import scripts.push_sync_daemon as relay

    snapshot_ts = 1_800_000_000.0
    responses = {
        "/api/pushes/refresh": {
            "history_authoritative": True,
            "source": "live",
            "snapshot_age_seconds": 0.0,
        },
        "/api/push/statistics/message?relayCacheOnly=1": {
            "messages": [{"id": "synthetic-live"}],
            "_source": "live",
            "_snapshotTs": snapshot_ts,
        },
        "/api/push/statistics/message/channels?relayCacheOnly=1": [
            {"name": "main"}
        ],
    }
    monkeypatch.setattr(
        relay,
        "_fetch",
        lambda path, **_kwargs: responses[path],
    )
    forwarded: dict = {}

    def capture(messages, channels, **kwargs):
        forwarded.update(
            {"messages": messages, "channels": channels, **kwargs}
        )
        return True

    monkeypatch.setattr(relay, "_push", capture)

    assert relay.run_once() is True
    assert forwarded == {
        "messages": [{"id": "synthetic-live"}],
        "channels": [{"name": "main"}],
        "source": "live",
        "snapshot_ts": snapshot_ts,
    }


def test_relay_refuses_non_authoritative_local_history(monkeypatch):
    import scripts.push_sync_daemon as relay

    monkeypatch.setattr(
        relay,
        "_fetch",
        lambda *_args, **_kwargs: {"history_authoritative": False},
    )
    monkeypatch.setattr(
        relay,
        "_push",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("non-authoritative history must not be relayed")
        ),
    )

    assert relay.run_once() is False


def test_relay_requires_receiver_persistence_confirmation(monkeypatch):
    import scripts.push_sync_daemon as relay

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(
                {"ok": True, "received": 1, "history_authoritative": False}
            ).encode()

    monkeypatch.setattr(
        relay.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: Response(),
    )

    assert relay._push(
        [{"id": "synthetic-live"}],
        [],
        source="live",
        snapshot_ts=1_800_000_000.0,
    ) is False


def test_relay_interval_includes_cycle_duration(monkeypatch):
    import scripts.push_sync_daemon as relay

    class StopLoop(Exception):
        pass

    monotonic_values = iter([100.0, 108.0])
    sleeps = []
    monkeypatch.setattr(relay, "SECRET", "synthetic-secret")
    monkeypatch.setattr(relay, "RENDER", "https://deployment.example.invalid")
    monkeypatch.setattr(relay, "INTERVAL", 30)
    monkeypatch.setattr(relay, "run_once", lambda: True)
    monkeypatch.setattr(relay.time, "monotonic", lambda: next(monotonic_values))

    def stop_after_sleep(seconds):
        sleeps.append(seconds)
        raise StopLoop

    monkeypatch.setattr(relay.time, "sleep", stop_after_sleep)

    try:
        relay.run()
    except StopLoop:
        pass

    assert sleeps == [22.0]
