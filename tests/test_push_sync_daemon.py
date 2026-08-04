"""Synthetic checks for the provenance-preserving Mac-to-Render relay."""

from __future__ import annotations


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
