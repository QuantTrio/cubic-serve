"""Real progress renews rank-local inactivity; total audit duration is not capped."""

from collections import deque
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from cubic_serve.runtime import performance as serving


@pytest.mark.parametrize("scale", [1.0, 60.0, 120.0])
@pytest.mark.parametrize("section", ["graph_measurement", "target_numerical_audit", "provider_numerical_audit"])
def test_real_progress_can_complete_beyond_twenty_minutes_without_total_cutoff(scale, section):
    class Channel:
        def __init__(self):
            self.items = deque()

        def recv(self):
            return self.items.popleft()

    controller = serving._ProcessController.__new__(serving._ProcessController)
    channels = (Channel(), Channel())
    controller._results = channels
    controller._processes = (SimpleNamespace(exitcode=None),) * 2
    controller._startup_progress = serving._StartupProgress(2)
    clock = [0.0]
    schedule = deque([(3, "base", 10, 10), (8, section, 1, 9),
                      (16, section, 2, 9), (24, "graph_capture", 1, 9),
                      (32, "ready", 0, 0)])

    def wait(pending, timeout):
        assert set(pending) == set(channels)
        when, section, completed, total = schedule[0]
        when *= scale
        if when > clock[0] + timeout:
            clock[0] += timeout
            return []
        schedule.popleft()
        clock[0] = float(when)
        for rank, channel in enumerate(channels):
            if section == "ready":
                item = serving._Ready(rank, 128, 2048, 256, "graphs", "states", None, (), ())
            else:
                unit = ("bytes" if section == "base" else "witnesses"
                        if section in {"target_numerical_audit", "provider_numerical_audit"}
                        else "buckets")
                item = serving._LoadProgress(rank, section, completed, total, unit)
            channel.items.append(item)
        return list(channels)

    with patch.object(serving, "wait", wait), patch.object(
        serving.time, "monotonic", lambda: clock[0],
    ):
        assert len(controller._receive_ready(10.0 * scale)) == 2
        assert clock[0] == 32.0 * scale


@pytest.mark.parametrize('exitcode,signal_name', [(-11, 'SIGSEGV'), (-9, 'SIGKILL'), (7, None), (0, None), (None, None)])
def test_closed_startup_channel_reports_worker_exit_without_guessing_cause(
    exitcode, signal_name, monkeypatch,
):
    class ClosedChannel:
        def recv(self):
            raise EOFError

    channel = ClosedChannel()
    controller = serving._ProcessController.__new__(serving._ProcessController)
    controller._results = (channel,)
    controller._processes = (SimpleNamespace(pid=3536, exitcode=exitcode),)
    controller._startup_progress = serving._StartupProgress(1)
    monkeypatch.setattr(serving, 'wait', lambda *a, **k: [channel])
    with pytest.raises(serving.Qwen3_5TP2PP2ServingError) as failure:
        controller._receive_ready(10)
    text = str(failure.value)
    assert 'rank 0 graph startup channel closed' in text
    assert 'rank=0 pid=3536' in text and f'exitcode={exitcode}' in text
    if signal_name:
        assert signal_name in text
    if exitcode == -11:
        assert 'PYTHONFAULTHANDLER=1' in text and 'OOM' not in text
    elif exitcode == -9:
        assert 'Check the kernel log' in text
    else:
        assert 'SIGSEGV' not in text and 'SIGKILL' not in text
    assert isinstance(failure.value.__cause__, EOFError)


def test_process_death_before_channel_event_reports_correct_rank(monkeypatch):
    controller = serving._ProcessController.__new__(serving._ProcessController)
    controller._results = (object(), object())
    controller._processes = (SimpleNamespace(pid=111, exitcode=None),
                             SimpleNamespace(pid=222, exitcode=-11))
    controller._startup_progress = serving._StartupProgress(2)
    monkeypatch.setattr(serving, 'wait', lambda *a, **k: [])
    with pytest.raises(serving.Qwen3_5TP2PP2ServingError, match=r'rank=1 pid=222 exitcode=-11 \(SIGSEGV\)'):
        controller._receive_ready(10)
