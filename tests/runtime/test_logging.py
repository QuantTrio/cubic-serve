"""Display precision must not change numerical data or opaque log text."""


import logging
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

from cubic_serve.runtime.logging import _InterceptHandler


def _logging_process(tmp_path, code):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "",
           "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src")}
    return subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env,
                          capture_output=True, text=True, check=True, timeout=30)


def test_default_serve_keeps_console_output_without_creating_files(tmp_path):
    result = _logging_process(tmp_path, '''
import logging
from cubic_serve import cli
from cubic_serve.runtime.logging import finish_log_startup
def serve(args):
    logging.info("startup-visible")
    finish_log_startup()
    logging.error("runtime-visible")
cli._serve_qwen3_5 = serve
assert cli.main(["serve", "unused", "--execution-mode", "eager"]) == 0
''')
    assert "startup-visible" in result.stderr
    assert "runtime-visible" in result.stderr
    assert not list(tmp_path.iterdir())


def test_file_logging_switches_after_summary_and_collects_workers(tmp_path):
    result = _logging_process(tmp_path, '''
import logging,subprocess,sys
from cubic_serve.runtime.logging import parse_log_config,serve_logging,finish_log_startup,configure_logging
with serve_logging(parse_log_config('{"dir":"./logs"}')):
    print("startup-parent")
    subprocess.run([sys.executable,"-c","import sys;print('startup-worker');print('startup-worker-error',file=sys.stderr)"],check=True)
    logging.info("final-startup-summary")
    finish_log_startup()
    finish_log_startup()
    configure_logging()
    print("runtime-parent")
    logging.error("runtime-error")
    subprocess.run([sys.executable,"-c","print('runtime-worker-中文')"],check=True)
''')
    console = result.stdout + result.stderr
    for text in ("startup-parent", "startup-worker", "startup-worker-error", "final-startup-summary"):
        assert text in console
    for text in ("runtime-parent", "runtime-error", "runtime-worker-中文"):
        assert text not in console
    log = (tmp_path / "logs/cubic-serve.log").read_text()
    for text in ("startup-parent", "startup-worker-error", "final-startup-summary",
                 "runtime-parent", "runtime-error", "runtime-worker-中文"):
        assert log.count(text) == 1
    assert "cubic-log-ready" not in log and "\x00" not in log


def test_console_always_and_restarts_append(tmp_path):
    code = '''
from cubic_serve.runtime.logging import parse_log_config,serve_logging,finish_log_startup
with serve_logging(parse_log_config('{"dir":"./logs","console":"always"}')):
    print("append-startup")
    finish_log_startup()
    print("append-runtime")
'''
    for _ in range(2):
        result = _logging_process(tmp_path, code)
        assert "append-runtime" in result.stdout
    log = (tmp_path / "logs/cubic-serve.log").read_text()
    assert log.count("append-startup") == log.count("append-runtime") == 2


def test_size_rotation_preserves_all_records(tmp_path):
    _logging_process(tmp_path, '''
from cubic_serve.runtime.logging import parse_log_config,serve_logging,finish_log_startup
with serve_logging(parse_log_config('{"dir":"./logs","rotation":200,"retention":null}')):
    finish_log_startup()
    for n in range(300):
        print(f"record-{n:03d} " + "x"*80, flush=True)
''')
    files = list((tmp_path / "logs").glob("*.log"))
    assert len(files) > 1
    text = "".join(p.read_text() for p in files)
    for n in range(300):
        assert text.count(f"record-{n:03d} ") == 1


def test_time_rotation_and_file_count_retention(tmp_path):
    _logging_process(tmp_path, '''
import time
from cubic_serve.runtime.logging import parse_log_config,serve_logging
with serve_logging(parse_log_config('{"dir":"./logs","rotation":"1 ms","retention":2}')):
    for n in range(7):
        print(f"timed-record-{n}",flush=True)
        time.sleep(.02)
''')
    files = list((tmp_path / "logs").glob("*.log"))
    assert 1 < len(files) <= 3
    assert "timed-record-6" in "".join(p.read_text() for p in files)


def test_age_retention_removes_old_rotated_logs(tmp_path):
    directory = tmp_path / "logs"
    directory.mkdir()
    old = directory / "cubic-serve.2000-01-01_00-00-00_000000.log"
    old.write_text("old log")
    os.utime(old, (1, 1))
    _logging_process(tmp_path, '''
from cubic_serve.runtime.logging import parse_log_config,serve_logging
with serve_logging(parse_log_config('{"dir":"./logs","rotation":20,"retention":"1 day"}')):
    print("trigger-rotation-"*5,flush=True)
''')
    assert not old.exists()


def test_split_utf8_and_ready_marker_remain_lossless(tmp_path):
    result = _logging_process(tmp_path, '''
import os
from cubic_serve.runtime import logging as logs
read = os.read
logs.os.read = lambda fd,size: read(fd,min(size,1))
with logs.serve_logging(logs.parse_log_config('{"dir":"./logs"}')):
    print("启动完成",flush=True)
    logs.finish_log_startup()
    print("生成完成",flush=True)
''')
    assert "启动完成" in result.stdout and "生成完成" not in result.stdout
    assert (tmp_path / "logs/cubic-serve.log").read_text() == "启动完成\n生成完成\n"


def test_failed_startup_preserves_console_file_and_original_exception(tmp_path):
    result = _logging_process(tmp_path, '''
from cubic_serve.runtime.logging import parse_log_config,serve_logging
try:
    with serve_logging(parse_log_config('{"dir":"./logs"}')):
        print("startup-failed",flush=True)
        raise RuntimeError("original failure")
except RuntimeError as error:
    assert str(error)=="original failure"
else:
    raise AssertionError("exception swallowed")
''')
    assert "startup-failed" in result.stdout
    assert "startup-failed" in (tmp_path / "logs/cubic-serve.log").read_text()


def test_disk_failure_reports_error_and_drains_output_to_console(tmp_path):
    result = _logging_process(tmp_path, '''
from loguru._file_sink import FileSink
from cubic_serve.runtime.logging import parse_log_config,serve_logging,finish_log_startup
def fail(*args):
    raise OSError("disk full")
FileSink.write=fail
try:
    with serve_logging(parse_log_config('{"dir":"./logs"}')):
        finish_log_startup()
        print("x"*150000,flush=True)
        print("continued-after-failure",flush=True)
except RuntimeError as error:
    assert str(error)=="file logging failed"
else:
    raise AssertionError("disk failure hidden")
''')
    assert "file logging failed" in result.stderr
    assert "continued-after-failure" in result.stdout


def test_invalid_file_config_fails_before_model_loading(tmp_path):
    result = _logging_process(tmp_path, '''
from cubic_serve import cli
def forbidden(args):
    raise AssertionError("model should not be loaded")
cli._serve_qwen3_5=forbidden
assert cli.main(["serve","unused","--execution-mode","eager","--log-config",'{"dir":"./logs","rotation":"1 potato"}'])==2
''')
    assert "--log-config" in result.stderr
    assert not list(tmp_path.iterdir())


def test_log_config_rejects_invalid_fields_and_nonpositive_intervals(tmp_path):
    import pytest
    from cubic_serve.runtime.logging import parse_log_config

    for value in ("null", "[]", "{", "{}", '{"dir":3}',
                  '{"dir":"logs","unknown":1}', '{"dir":"logs","console":"invalid"}'):
        with pytest.raises(ValueError):
            parse_log_config(value)
    for key, invalid in (("rotation", True), ("rotation", 0), ("rotation", "0 seconds"),
                         ("rotation", "-1 day"), ("retention", False), ("retention", -1)):
        with pytest.raises(ValueError):
            parse_log_config(json.dumps({"dir": str(tmp_path), key: invalid}))


def test_repeated_logging_scopes_release_threads_and_descriptors(tmp_path):
    _logging_process(tmp_path, '''
import os,threading
from cubic_serve.runtime.logging import parse_log_config,serve_logging,finish_log_startup
before=len(os.listdir('/proc/self/fd'))
for _ in range(5):
    with serve_logging(parse_log_config('{"dir":"./{logs}"}')):
        print("before-ready")
        finish_log_startup()
        print("after-ready")
assert len(os.listdir('/proc/self/fd'))==before
assert not any(t.name=="cubic-log-relay" for t in threading.enumerate())
''')
    assert (tmp_path / "{logs}/cubic-serve.log").read_text().count("after-ready") == 5


def test_log_close_does_not_wait_for_an_inherited_worker_descriptor(tmp_path):
    _logging_process(tmp_path, '''
import subprocess,sys,threading
from cubic_serve.runtime.logging import parse_log_config,serve_logging
child=None
try:
    with serve_logging(parse_log_config('{"dir":"./logs"}')):
        child=subprocess.Popen([sys.executable,"-c","import sys;sys.stdin.buffer.read()"],stdin=subprocess.PIPE)
        print("before-close",flush=True)
    assert child.poll() is None
    assert not any(t.name=="cubic-log-relay" for t in threading.enumerate())
finally:
    if child is not None:
        child.stdin.close()
        assert child.wait(timeout=10)==0
''')
    assert (tmp_path / "logs/cubic-serve.log").read_text() == "before-close\n"


def test_log_close_reports_retention_failure(tmp_path):
    result = _logging_process(tmp_path, '''
from loguru._file_sink import Retention
from cubic_serve.runtime.logging import parse_log_config,serve_logging
def fail(*args):
    raise OSError("retention failure")
Retention.retention_age=fail
try:
    with serve_logging(parse_log_config('{"dir":"./logs","rotation":null}')):
        print("kept-on-disk",flush=True)
except RuntimeError as error:
    assert str(error)=="file logging failed"
else:
    raise AssertionError("retention failure hidden")
''')
    assert "file logging failed during close" in result.stderr
    assert (tmp_path / "logs/cubic-serve.log").read_text() == "kept-on-disk\n"


def test_graph_progress_counts_real_work_and_estimates_remaining_time(monkeypatch, caplog):
    from cubic_serve.runtime import logging as presentation

    times = iter((100., 110., 130., 200., 208.))
    monkeypatch.setattr(presentation.time, "monotonic", lambda: next(times))
    report = presentation.startup_graph_progress(0, "Preparing MTP graphs")
    with caplog.at_level(logging.INFO, logger="cubic_serve.startup"):
        report(0, 4, phase="rounds")
        report(1, 4, phase="rounds")
        report(4, 4, phase="rounds")
        report(0, 2, phase="prompt graphs")
        report(1, 2, phase="prompt graphs")
    messages = [record.getMessage() for record in caplog.records]
    assert messages[0].startswith("Preparing MTP speculative decode config: [")
    assert messages[3].startswith("Preparing MTP prefill graphs: [")
    assert "0/4 0% | 0s | ETA --" in messages[0]
    assert "1/4 25% | 10s | ETA ~30s" in messages[1]
    assert "[####################] 4/4 100%" in messages[2]
    assert "ETA ~0s" in messages[2]
    assert "0/2 0% | 0s | ETA --" in messages[3]
    assert "1/2 50% | 8s | ETA ~8s" in messages[4]


def test_graph_progress_never_completes_failed_work_or_reports_other_ranks(caplog):
    import pytest
    from cubic_serve.runtime import logging as presentation

    error = RuntimeError("original startup failure")
    report = presentation.startup_graph_progress(0, "Calibrating MTP graph memory")
    with caplog.at_level(logging.INFO, logger="cubic_serve.startup"):
        with pytest.raises(RuntimeError) as caught:
            report(0, 3, phase="rounds")
            report(1, 3, phase="rounds")
            raise error
        assert caught.value is error
        assert "100%" not in caplog.text
        caplog.clear()
        presentation.startup_graph_progress(3, "MTP")(0, 5, phase="rounds")
        report(0, 0, phase="empty")
        assert not caplog.records


def test_startup_summary_and_capture_progress_are_not_repeated_for_ranks(caplog):
    from cubic_serve.runtime import logging as presentation

    with caplog.at_level(logging.INFO, logger="cubic_serve.startup"):
        for rank in range(4):
            presentation.startup_cache_loaded(rank, "fingerprint-value")
            presentation.startup_sampling_qualified(rank, 8, 21, masked=True)
            presentation.startup_draft_portfolio(rank, 3, ((8, 3),))
            presentation.startup_progress(rank, "graph_capture", 1, 8)
    messages = [record.getMessage() for record in caplog.records]
    assert messages == ["Startup cache loaded fingerprint=fingerprint-value",
                        "Capturing CUDA graphs: 1/8"]
    assert all("cases=" not in message and "engineering_cases" not in message
               for message in messages)


def test_startup_diagnostics_remain_available_at_debug(caplog):
    from cubic_serve.runtime.logging import startup_sampling_qualified

    with caplog.at_level(logging.DEBUG, logger="cubic_serve.startup"):
        startup_sampling_qualified(1, 8, 21, masked=True)
    assert "rank=1" in caplog.text and "cases=21" in caplog.text


def test_diagnostic_events_preserve_error_identity_and_traceback(caplog):
    from cubic_serve.runtime.logging import diagnostic_event

    owner = logging.getLogger("cubic_serve.worker.test")
    with caplog.at_level(logging.ERROR, logger=owner.name):
        try:
            raise RuntimeError("private backend diagnostic")
        except RuntimeError as error:
            diagnostic_event(owner, "performance.continuous_batch_dispatcher_failed")
            assert caplog.records[-1].exc_info[1] is error
    assert caplog.records[-1].name == owner.name
    assert caplog.records[-1].getMessage() == "continuous batch dispatcher failed"


def test_diagnostic_event_warnings_are_not_hidden(caplog):
    from cubic_serve.runtime.logging import diagnostic_event

    owner = logging.getLogger("cubic_serve.worker.test")
    with caplog.at_level(logging.WARNING, logger=owner.name):
        diagnostic_event(owner, "quantized_qualification.mtp_composition_receipt_unavailable_s",
                         "missing implementation")
    assert caplog.records[-1].levelno == logging.WARNING
    assert "missing implementation" in caplog.records[-1].getMessage()


def _render(message, *args):
    record = logging.LogRecord("test", logging.INFO, __file__, 1, message, args, None)
    with patch("cubic_serve.runtime.logging.logger") as logger:
        _InterceptHandler().emit(record)
    return logger.opt.return_value.log.call_args.args[1], record


def test_nested_storage_errors_have_bounded_precision_and_keep_original_values():
    errors = (("full", 0.2727425923462147),)
    rendered, record = _render(
        "native_error=%.6f baseline_error=%.6f validated_storage_errors=%s",
        errors[0][1], 0.096526123456, errors,
    )
    assert rendered == (
        "native_error=0.272743 baseline_error=0.096526 "
        "validated_storage_errors=(('full', 0.272743),)"
    )
    assert record.args[2] is errors
    assert errors[0][1] == 0.2727425923462147


def test_mapping_and_dataclass_arguments_share_display_precision():
    @dataclass
    class Policy:
        alpha: float

    policy = Policy(0.1234567890123)
    rendered, _ = _render("data=%s policy=%s", {"values": [policy.alpha]}, policy)
    assert "data={'values': [0.123457]}" in rendered
    assert "Policy(alpha=0.123457)" in rendered
    assert policy.alpha == 0.1234567890123


def test_explicit_numeric_formats_and_small_nonzero_values():
    rendered, _ = _render("duration=%.3f tiny=%s big=%s", 1.23456789, 1.23456789e-12, 1.23456789e20)
    assert rendered == "duration=1.235 tiny=1.234568e-12 big=1.234568e+20"


def test_mapping_substitution_and_opaque_strings_are_preserved():
    opaque = "/weights/model-0.2727425923462147 36.103.198.225"
    rendered, _ = _render("error=%(error)s path=%(path)s", {"error": 0.123456789, "path": opaque})
    assert rendered == f"error=0.123457 path={opaque}"
    assert _render(opaque)[0] == opaque


def test_serving_memory_breakdown_keeps_state_reserve_and_host_bytes_separate(caplog):
    from types import SimpleNamespace

    import torch

    from cubic_serve.runtime.quantized_graph import _log_memory_residency
    from cubic_serve.runtime.cache import KVLayout

    # The summary consumes already measured ledgers; it must not query or copy a device.
    layout = KVLayout((0,), 32, 1, 32)
    arena = SimpleNamespace(
        layout=layout, backed_bytes=80, states={"gdn": torch.empty(8, dtype=torch.bfloat16)},
        spec=SimpleNamespace(state_slot_capacity=4, scratch_capacity=16,
                             scratch_arena_ids=("graph",)),
    )
    draft = SimpleNamespace(
        backed_bytes=36, states={"tail": torch.empty(2, dtype=torch.bfloat16)},
        spec=SimpleNamespace(scratch_capacity=32, scratch_arena_ids=("graph",)),
    )
    weights = SimpleNamespace(storage_bytes=dict(device=1000, pinned_host=200, pageable_host=300))
    with caplog.at_level("INFO", logger="cubic_serve.runtime.quantized_graph"):
        _log_memory_residency(
            rank=1, device="cuda:5", weights=weights, arena=arena, draft_arena=draft,
            startup_free_bytes=2000, final_free_bytes=500, reserve_bytes=400,
            page_bytes=320, draft_page_bytes=160, max_running_requests=4, maximum=16,
        )
    assert "weights_device_bytes=1000 weights_pinned_host_bytes=200" in caplog.text
    assert "kv_page_backing_bytes=96 state_bytes=20 other_device_bytes=384" in caplog.text
    assert "free_bytes=500 required_reserve_bytes=400" in caplog.text
    assert "target_bytes_per_token=10.000 draft_bytes_per_token=5.000" in caplog.text
    assert "max_running_requests=4 graph_max_M=16 state_owner_slots=4" in caplog.text
    assert "target_scratch_slots=16 draft_scratch_slots=32" in caplog.text
