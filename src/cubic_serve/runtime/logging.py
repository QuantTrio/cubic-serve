"""Presentation of diagnostics for Cubic Serve and its dependencies.

Startup wording, levels and rank suppression belong here. This module must not
import execution components or control their state; cache dependencies stop at
this presentation boundary. Startup reporters accept only display metadata.
"""

from __future__ import annotations

import logging
import codecs
import json
import os
import re
import sys
import threading
import time
import uuid
from contextlib import contextmanager, suppress
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any

from loguru import logger

_FORMAT = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}"
_STARTUP = logging.getLogger("cubic_serve.startup")
_CONSOLE_SINK_ID: int | None = None
_FINISH_LOG_STARTUP = None


@dataclass(frozen=True)
class LogConfig:
    directory: Path
    rotation: str | int | None = "100 MB"
    retention: str | int | None = "7 days"
    console: str = "startup"


def parse_log_config(value: str | None) -> LogConfig | None:
    if value is None:
        return None
    try:
        options = json.loads(value)
    except (TypeError, ValueError) as error:
        raise ValueError("--log-config must be a JSON object") from error
    if not isinstance(options, dict) or set(options) - {"dir", "rotation", "retention", "console"}:
        raise ValueError("--log-config supports dir, rotation, retention and console")
    directory = options.get("dir")
    if not isinstance(directory, str) or not directory.strip():
        raise ValueError("--log-config requires a non-empty dir")
    for key in ("rotation", "retention"):
        rule = options.get(key)
        if rule is None:
            continue
        if type(rule) is int and rule >= (1 if key == "rotation" else 0):
            continue
        if isinstance(rule, str):
            if key == "rotation" and rule in {"hourly", "daily", "weekly", "monthly", "yearly"}:
                continue
            match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*[A-Za-z]+", rule.strip())
            if match and float(match[1]) > 0:
                continue
        raise ValueError(f"--log-config {key} must be a positive size/duration, integer or null")
    console = options.get("console", "startup")
    if console not in ("startup", "always"):
        raise ValueError("--log-config console must be startup or always")
    return LogConfig(Path(directory).expanduser().resolve(),
                     options.get("rotation", "100 MB"),
                     options.get("retention", "7 days"), console)


def finish_log_startup() -> None:
    """Switch console routing only after the HTTP startup summary is emitted."""
    if _FINISH_LOG_STARTUP is not None:
        _FINISH_LOG_STARTUP()


@contextmanager
def serve_logging(config: LogConfig | None):
    """One frontend writer owns rotation for all inherited stdout/stderr."""
    global _FINISH_LOG_STARTUP
    configure_logging()
    if config is None:
        yield
        return
    path = str(config.directory / "cubic-serve.log").replace("{", "{{").replace("}", "}}")
    try:
        sink_id = logger.add(
            path, rotation=config.rotation, retention=config.retention, mode="a",
            encoding="utf-8", colorize=False, format="{message}",
            filter=lambda record: record["extra"].get("_file_output", False),
            catch=False, backtrace=False, diagnose=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"--log-config: {error}") from error
    file_logger = logger.bind(_file_output=True).opt(raw=True)
    sys.stdout.flush()
    sys.stderr.flush()
    stdout_fd, stderr_fd = os.dup(1), os.dup(2)
    stdout_blocking = os.get_blocking(stdout_fd)
    read_fd, write_fd = os.pipe()
    os.set_blocking(stdout_fd, False)
    token = b"\x00cubic-log:" + uuid.uuid4().hex.encode()
    marker, stop_marker = token + b":ready\x00", token + b":stop\x00"
    markers = (marker, stop_marker)
    failures = []

    def relay():
        startup = True
        pending = b""
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

        def emit(data, *, final=False):
            text = decoder.decode(data, final=final)
            if not failures:
                try:
                    if text:
                        file_logger.info(text)
                except Exception as error:
                    failures.append(error)
                    with suppress(OSError):
                        os.write(stderr_fd, b"error: file logging failed; continuing on console\n")
            if startup or config.console == "always" or failures:
                with suppress(BlockingIOError, BrokenPipeError):
                    view = memoryview(data)
                    while view:
                        view = view[os.write(stdout_fd, view):]

        try:
            stopped = False
            while chunk := os.read(read_fd, 64 * 1024):
                data = pending + chunk
                pending = b""
                while True:
                    positions = [(data.find(control), control) for control in markers]
                    found = [(position, control) for position, control in positions if position >= 0]
                    if not found:
                        break
                    position, control = min(found)
                    emit(data[:position])
                    data = data[position + len(control):]
                    if control == stop_marker:
                        stopped = True
                        break
                    startup = False
                if stopped:
                    break
                # Hold only a possible split control marker, never an unbounded line.
                for length in range(min(len(data), len(marker) - 1), 0, -1):
                    if any(data.endswith(control[:length]) for control in markers):
                        pending, data = data[-length:], data[:-length]
                        break
                emit(data)
            emit(pending, final=True)
        finally:
            try:
                logger.remove(sink_id)
            except Exception as error:
                if not failures:
                    failures.append(error)
                    with suppress(OSError):
                        os.write(stderr_fd, b"error: file logging failed during close\n")
            finally:
                os.close(read_fd)

    previous_stdout, previous_stderr = sys.stdout, sys.stderr
    previous_unbuffered = os.environ.get("PYTHONUNBUFFERED")
    previous_ready = _FINISH_LOG_STARTUP
    ready_sent = False

    def ready():
        nonlocal ready_sent
        if not ready_sent:
            sys.stdout.flush()
            sys.stderr.flush()
            os.write(2, marker)
            ready_sent = True

    relay_thread = threading.Thread(target=relay, name="cubic-log-relay", daemon=True)
    relay_thread.start()
    tee_stdout = tee_stderr = None
    failed = False
    try:
        os.dup2(write_fd, 1)
        os.dup2(write_fd, 2)
        os.close(write_fd)
        tee_stdout = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8")
        tee_stderr = os.fdopen(os.dup(2), "w", buffering=1, encoding="utf-8")
        sys.stdout, sys.stderr = tee_stdout, tee_stderr
        os.environ["PYTHONUNBUFFERED"] = "1"
        _FINISH_LOG_STARTUP = ready
        # Loguru must follow the redirected stderr rather than its former stream.
        configure_logging()
        yield
    except BaseException:
        failed = True
        raise
    finally:
        _FINISH_LOG_STARTUP = previous_ready
        sys.stdout.flush()
        sys.stderr.flush()
        # An ordered end marker drains prior output without waiting for other
        # processes to close inherited descriptors.
        os.write(2, stop_marker)
        sys.stdout, sys.stderr = previous_stdout, previous_stderr
        if tee_stdout is not None:
            tee_stdout.close()
        if tee_stderr is not None:
            tee_stderr.close()
        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
        if previous_unbuffered is None:
            os.environ.pop("PYTHONUNBUFFERED", None)
        else:
            os.environ["PYTHONUNBUFFERED"] = previous_unbuffered
        configure_logging()
        relay_thread.join()
        os.set_blocking(1, stdout_blocking)
        os.close(stdout_fd)
        os.close(stderr_fd)
        if failures and not failed:
            raise RuntimeError("file logging failed") from failures[0]


def startup_graph_progress(rank: int, label: str):
    """Display completed startup work; ETA uses completed work in this phase.

    This observer creates no thread and never advances without a completion
    notification. A failed build cannot print a successful final count.
    """
    phase_started = None
    last_phase = None

    def report(completed: int, total: int, *, phase: str) -> None:
        nonlocal phase_started, last_phase
        if rank != 0 or total == 0:
            return
        now = time.monotonic()
        if phase != last_phase:
            phase_started, last_phase = now, phase
        elapsed = now - phase_started
        remaining = f"~{elapsed * (total - completed) / completed:.0f}s" if completed else "--"
        filled = 20 * completed // total
        bar = "#" * filled + "-" * (20 - filled)
        display_phase = {"prompt graphs": "prefill graphs",
                         "rounds": "speculative decode config"}.get(phase, phase)
        action = {"Calibrating MTP graph memory": "Calibrating MTP",
                  "Preparing MTP graphs": "Preparing MTP"}.get(label, label)
        _STARTUP.info("%s %s: [%s] %d/%d %.0f%% | %.0fs | ETA %s",
                      action, display_phase, bar, completed, total,
                      100 * completed / total, elapsed, remaining)

    return report


def startup_cache_loaded(rank: int, fingerprint: str) -> None:
    if rank == 0:
        _STARTUP.info("Startup cache loaded fingerprint=%s", fingerprint)


def startup_progress(
    rank: int, section: str, completed: int, total: int, *, capacity: int | None = None,
) -> None:
    if section == "graph_capture":
        if rank == 0:
            if capacity is None:
                _STARTUP.info("Capturing CUDA graphs: %d/%d", completed, total)
            else:
                _STARTUP.info("Capturing CUDA graphs: M%d %d/%d", capacity, completed, total)
    else:
        section = {"graph_measurement": "graph_calibration"}.get(section, section)
        _STARTUP.info("Qualification rank=%d phase=%s completed=%d/%d",
                      rank, section, completed, total)


def startup_sampling_qualified(rank: int, capacity: int, cases: int, *, masked: bool) -> None:
    _STARTUP.debug("Sampling graph ready rank=%d M=%d masked=%s cases=%d",
                   rank, capacity, masked, cases)


def startup_graph_captured(
    mode: str, rank: int, capacity: int, build: float, eager: float, capture: float, check: float,
) -> None:
    _STARTUP.debug("%s graph rank=%d M=%d build_s=%.2f eager_s=%.2f capture_s=%.2f check_s=%.2f",
                   mode, rank, capacity, build, eager, capture, check)


def startup_graphs_ready(rank: int) -> None:
    if rank == 0:
        _STARTUP.info("Execution graphs ready")


def startup_graph_capture_started(rank: int) -> None:
    if rank == 0:
        _STARTUP.info("Capturing CUDA graphs")


def startup_state_ready(rank: int, duration: float) -> None:
    _STARTUP.debug("Graph state ready rank=%d duration_s=%.2f", rank, duration)


def startup_sampling_started(rank: int) -> None:
    if rank == 0:
        _STARTUP.info("Preparing sampling")


def startup_sampling_ready(rank: int, duration: float) -> None:
    _STARTUP.debug("Sampling ready rank=%d duration_s=%.2f", rank, duration)


def startup_draft_portfolio(rank: int, depth: int, reviews: tuple[tuple[int, int], ...]) -> None:
    _STARTUP.debug("Draft graphs ready rank=%d K=%d reviews=%s", rank, depth, reviews)


def serving_status(
    owner: logging.Logger, active_peak: int, kv_usage_percent: float,
    prefix_hits: int, prefix_queries: int, prefill_tokens: int, seconds: float,
    tokens: int, accepted: int, proposed: int, rounds: int,
) -> None:
    fields = [
        f"[STATUS] n_req={active_peak}",
        f"kv={kv_usage_percent:.1f}%",
        f"prefix={100.0 * prefix_hits / prefix_queries:.1f}%" if prefix_queries else "prefix=0.0%",
        f"prefill={prefill_tokens / seconds:.1f} tok/s" if seconds > 0 else "prefill=0.0 tok/s",
    ]
    if proposed:
        fields.extend((f"draft={proposed / rounds:.1f}", f"accept={accepted / rounds:.1f}"))
    fields.append(f"decode={tokens / seconds:.1f} tok/s" if seconds > 0 else "decode=0.0 tok/s")
    owner.info(" ".join(fields), stacklevel=2)


def calibration_k_plan(
    owner: logging.Logger, dense: bool, n: int, k: int, bits: int, multiprocessors: int,
    partitions: int, rows: list[int], costs: dict[int, list[float]], errors: dict[int, float],
) -> None:
    cost_text = "{" + ", ".join(
        f"{p}: [{', '.join(f'{value:.4f}' for value in values)}]" for p, values in costs.items()
    ) + "}"
    error_text = "{" + ", ".join(f"{p}: {value:.2f}" for p, value in errors.items()) + "}"
    owner.info("Marlin %s K plan N=%d K=%d bits=%d SM=%d partitions=%d "
               "rows=%s costs_ms=%s time_deviation_pct=%s",
               "dense" if dense else "routed", n, k, bits, multiprocessors, partitions,
               rows, cost_text, error_text, stacklevel=2)


class _LogFloat(float):
    """Bound implicit display precision without rounding the underlying value."""

    def __repr__(self) -> str:
        return format(self, ".6e" if 0 < abs(self) < 1e-6 or abs(self) >= 1e9 else ".6f")

    __str__ = __repr__


class _LogDataclass:
    def __init__(self, value: object) -> None:
        self.value = value

    def __repr__(self) -> str:
        members = ", ".join(
            f"{field.name}={_display_argument(getattr(self.value, field.name))!r}"
            for field in fields(self.value) if field.repr
        )
        return f"{type(self.value).__qualname__}({members})"

    __str__ = __repr__


def _display_argument(value: object) -> object:
    """Format typed log arguments only; never rewrite identifiers or messages."""
    if isinstance(value, float):
        return _LogFloat(value)
    if isinstance(value, tuple):
        return tuple(_display_argument(item) for item in value)
    if isinstance(value, list):
        return [_display_argument(item) for item in value]
    if isinstance(value, dict):
        return {key: _display_argument(item) for key, item in value.items()}
    if is_dataclass(value) and not isinstance(value, type):
        return _LogDataclass(value)
    return value


def format_weight_progress(completed: int, total: int, unit: str = "tensors") -> str:
    """Shared weight-load bar; counters describe actual loaded payloads."""
    filled = min(20, 20 * completed // total)
    amount = (f"{completed / (1 << 30):.2f}/{total / (1 << 30):.2f} GiB"
              if unit == "bytes" else f"{completed}/{total} tensors")
    return (f"[{'#' * filled}{'-' * (20 - filled)}] "
            f"{100.0 * completed / total:5.1f}% {amount}")


class _InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        message = str(record.msg)
        if record.args:
            message %= _display_argument(record.args)
        if record.name == "uvicorn.error" and record.levelno == logging.INFO and (
            record.msg == "Started server process [%d]"
            or message in (
                "Waiting for application startup.",
                "ASGI 'lifespan' protocol appears unsupported.",
                "Application startup complete.",
            )
        ):
            level = "DEBUG"
        logger.opt(exception=record.exc_info).log(level, message)


def configure_logging() -> None:
    """Route application, rank, and Uvicorn records through local-time Loguru."""
    global _CONSOLE_SINK_ID
    if _CONSOLE_SINK_ID is None:
        logger.remove()
    else:
        with suppress(ValueError):
            logger.remove(_CONSOLE_SINK_ID)
    _CONSOLE_SINK_ID = logger.add(
        sys.stderr,
        level="INFO",
        format=_FORMAT,
        colorize=sys.stderr.isatty(),
        backtrace=False,
        diagnose=False,
        filter=lambda record: not record["extra"].get("_file_output", False),
    )
    handler = _InterceptHandler()
    logging.basicConfig(handlers=(handler,), level=logging.INFO, force=True)
    logging.captureWarnings(True)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        dependency = logging.getLogger(name)
        dependency.handlers.clear()
        dependency.propagate = True


# Event producers own the data; presentation owns templates and severity.
_DIAGNOSTIC_EVENTS: dict[str, tuple[str, str]] = {
    "state_transfer.byte_proof": (
        "debug", "State transfer checked fields=%d owners=%d cases=%d seconds=%.3f",
    ),
    "state_transfer.cached": (
        "debug", "State transfer ready graph=%s cached=True fields=%d owners=%d",
    ),
    "state_transfer.prepared": (
        "debug", "State transfer ready graph=%s cached=False fields=%d owners=%d cold_s=%.3f",
    ),
    'performance.bucket_migration_audit_failed_before_nccl_teardown': (
        'exception',
        'bucket migration audit failed before NCCL teardown rank=%d'
    ),
    'performance.cache_ready_rank_d_device_s_pages': (
        'warning',
        'cache ready rank=%d device=%s pages=%d arena=%.3f GiB state_slots=%d scr'
        'atch_slots=%d reserve=%.3f GiB final_free=%.3f GiB'
    ),
    'performance.capturing_graph_rank_d_c_d': (
        'warning',
        'capturing graph rank=%d C%d'
    ),
    'performance.capturing_mtp_graph_rank_d_c_d': (
        'warning',
        'capturing MTP graph rank=%d C%d'
    ),
    'performance.continuous_batch_dispatcher_failed': (
        'exception',
        'continuous batch dispatcher failed'
    ),
    'performance.draft_proposal_calibration_rank_d_physical_m': (
        'info',
        'draft proposal calibration rank=%d physical_m=%d logical_m=%d cost_ms=%.'
        '3f'
    ),
    'performance.draft_s': (
        'warning',
        '[DRAFT] %s'
    ),
    'performance.draft_verifier_qualified_rank_d_physical_m': (
        'info',
        'draft verifier qualified rank=%d physical_m=%d target_m=%d'
    ),
    'performance.dynamic_draft_policy_s': (
        'info',
        'dynamic draft policy=%s'
    ),
    'performance.grammar_sampling_scratch_rank_d_capacity_d': (
        'info',
        'grammar sampling scratch rank=%d capacity=%d device=%.1f MiB pinned=%.1f'
        ' MiB'
    ),
    'performance.graph_and_rank_staging_memory_rank_d': (
        'info',
        'graph and rank staging memory rank=%d total=%.1f MiB transient=%.1f MiB '
        'buckets=%d'
    ),
    'performance.graph_cold_audit_ready_rank_d_c': (
        'warning',
        'graph cold audit ready rank=%d C%d'
    ),
    'performance.operator_config_digest_s_value_s': (
        'info',
        'operator config digest=%s value=%s'
    ),
    'performance.operator_strategy_rank_d_model_s_device': (
        'info',
        'operator strategy rank=%d model=%s device=%s config_content_digest=%s'
    ),
    'performance.padding_scratch_binding_failed_rank_d_c': (
        'error',
        'padding scratch binding failed rank=%d C%d M%d checks=%s'
    ),
    'performance.qualification_cache_hit_rank_d_check_migration': (
        'warning',
        'qualification cache hit rank=%d check=migration_cow'
    ),
    'performance.qualification_cache_partial_miss_rank_d_check': (
        'warning',
        'qualification cache partial miss rank=%d check=migration_cow; rerunning '
        'on all ranks'
    ),
    'performance.qualification_cache_write_failed_rank_d': (
        'warning',
        'qualification cache write failed rank=%d'
    ),
    'performance.reasoning_budget_scratch_rank_d_device_f': (
        'info',
        'reasoning budget scratch rank=%d device=%.3f MiB pinned=%.3f MiB'
    ),
    'performance.startup_check_begin_phase_s': (
        'info',
        'startup check begin phase=%s'
    ),
    'performance.startup_check_complete_phase_s_duration_s': (
        'info',
        'startup check complete phase=%s duration_s=%.2f'
    ),
    'performance.startup_check_failed_rank_d_phase_s': (
        'error',
        'startup check failed rank=%d phase=%s duration_s=%.2f'
    ),
    'performance.startup_probe_rank_d_allocator_cleanup_complete': (
        'warning',
        'startup probe rank=%d allocator cleanup complete'
    ),
    'performance.startup_probe_rank_d_body_build_begin': (
        'warning',
        'startup probe rank=%d body build begin'
    ),
    'performance.startup_probe_rank_d_body_build_complete': (
        'warning',
        'startup probe rank=%d body build complete'
    ),
    'performance.startup_probe_rank_d_bootstrap_warm_complete': (
        'warning',
        'startup probe rank=%d bootstrap warm complete'
    ),
    'performance.startup_probe_rank_d_cache_plan_complete': (
        'warning',
        'startup probe rank=%d cache plan complete'
    ),
    'performance.startup_probe_rank_d_cleanup_barrier_complete': (
        'warning',
        'startup probe rank=%d cleanup barrier complete'
    ),
    'performance.startup_probe_rank_d_context_reduce_complete': (
        'warning',
        'startup probe rank=%d context reduce complete'
    ),
    'performance.startup_probe_rank_d_final_body_build': (
        'warning',
        'startup probe rank=%d final body build begin'
    ),
    'performance.startup_probe_rank_d_final_body_build_2': (
        'warning',
        'startup probe rank=%d final body build complete'
    ),
    'performance.startup_probe_rank_d_gc_begin': (
        'warning',
        'startup probe rank=%d gc begin'
    ),
    'performance.startup_probe_rank_d_gc_complete': (
        'warning',
        'startup probe rank=%d gc complete'
    ),
    'performance.startup_probe_rank_d_warm_c_d': (
        'warning',
        'startup probe rank=%d warm C%d begin'
    ),
    'performance.startup_probe_rank_d_warm_c_d_2': (
        'warning',
        'startup probe rank=%d warm C%d complete'
    ),
    'performance.wave_audit_local_verdict_failed_rank_d': (
        'error',
        'wave audit local verdict failed rank=%d M%d code=%d detail=%s'
    ),
    'performance.worker_batch_failed_before_nccl_teardown_rank': (
        'error',
        'worker batch failed before NCCL teardown rank=%d request=%s\n%s'
    ),
    'performance.worker_bootstrap_entered_rank_d': (
        'info',
        'worker bootstrap entered rank=%d'
    ),
    'performance.worker_continuation_failed_before_nccl_teardown_rank': (
        'error',
        'worker continuation failed before NCCL teardown rank=%d request=%s\n%s'
    ),
    'performance.worker_metadata_prepared_bytes_d_duration_s': (
        'info',
        'worker metadata prepared bytes=%d duration_s=%.3f'
    ),
    'performance.worker_metadata_restored_rank_d_bytes_d': (
        'info',
        'worker metadata restored rank=%d bytes=%d duration_s=%.3f'
    ),
    'performance.worker_processes_spawned_count_d_duration_s': (
        'info',
        'worker processes spawned count=%d duration_s=%.2f'
    ),
    'quantized_graph.cache_allocation_complete_rank_d_duration_s': (
        'info',
        'cache allocation complete rank=%d duration_s=%.2f'
    ),
    'quantized_graph.cache_allocator_correction_rank_d_pages_d': (
        'warning',
        'cache allocator correction rank=%d pages=%d deficit=%.1f MiB rounding=%.'
        '1f MiB'
    ),
    'quantized_graph.cache_memory_admission_rank_d_free_f': (
        'info',
        'cache memory admission rank=%d free=%.1f MiB reserve=%.1f MiB graphs=%.1'
        'f MiB transient=%.1f MiB state_and_scratch=%.1f MiB page_bytes=%d local_'
        'pages=%d minimum_pages=%d'
    ),
    'quantized_graph.shared_cache_measured_startup': (
        'info',
        'shared KV/state pool uses calibrated portfolio allocation rank=%d'
    ),
    'quantized_graph.cuda_virtual_memory_unavailable_using_measured_startup': (
        'warning',
        'CUDA virtual memory unavailable; using calibrated startup allocation rank='
        '%d'
    ),
    'quantized_graph.fixed_address_kv_admission_rank_d_pages': (
        'info',
        'fixed-address KV admission rank=%d pages=%d added=%.1f MiB transient=%.1'
        'f MiB context=%d duration_s=%.2f'
    ),
    'quantized_graph.fixed_address_kv_bootstrap_rank_d_admitted': (
        'info',
        'fixed-address KV bootstrap rank=%d admitted_pages=%d virtual_pages=%d'
    ),
    'quantized_graph.graph_memory_budget_rank_d_resident_f': (
        'warning',
        'graph memory budget rank=%d resident=%.1f MiB transient=%.1f MiB allocat'
        'or_slack=%.1f MiB released_active=%.1f MiB released_physical=%.1f MiB'
    ),
    'quantized_graph.graph_memory_final_rank_d_free_f': (
        'warning',
        'graph memory final rank=%d free=%.1f MiB reserve=%.1f MiB released=%.1f '
        'MiB runtime_reserve=%.1f MiB input_workspace=%.1f MiB surplus=%.1f MiB'
    ),
    'quantized_graph.graph_memory_measurement_complete_rank_d_duration': (
        'info',
        'graph memory calibration complete rank=%d duration_s=%.2f'
    ),
    'quantized_graph.graph_memory_measurement_rank_d_m_d': (
        'info',
        'graph memory calibration rank=%d M%d build_s=%.2f capture_s=%.2f'
    ),
    'quantized_graph.independent_numerical_audit_passed_rank_d_m': (
        'warning',
        'independent numerical audit passed rank=%d M%d'
    ),
    'quantized_graph.independent_numerical_qualification_complete_rank_d_duration': (
        'info',
        'independent numerical qualification complete rank=%d duration_s=%.2f'
    ),
    'quantized_graph.initializing_execution_workers_and_cuda_communication': (
        'info',
        'Initializing execution workers and CUDA communication'
    ),
    'quantized_graph.kernel_preparation_complete_rank_d_post_weight': (
        'info',
        'kernel preparation complete rank=%d post_weight_s=%.2f pipeline_s=%.2f'
    ),
    'quantized_graph.operator_local_text_numerical_checks_complete_rank': (
        'info',
        'operator-local text numerical checks complete rank=%d'
    ),
    'quantized_graph.post_weight_startup_complete_rank_d_duration': (
        'info',
        'post-weight startup complete rank=%d duration_s=%.2f'
    ),
    'quantized_graph.preparing_kernels_and_measuring_graph_memory': (
        'info',
        'Preparing kernels and calibrating graph memory'
    ),
    'quantized_graph.s_captured_rank_d_m_d_route': (
        'warning',
        '%s captured rank=%d M%d route_fold=%s cost_us=%.2f->%.2f'
    ),
    'quantized_graph.text_precision_activation_s_attention_s_softmax': (
        'info',
        'Text precision activation=%s attention=%s softmax=%s recurrent_state=%s'
    ),
    'quantized_graph.typed_cross_m_identity_checks_complete_rank': (
        'info',
        'typed cross-M identity checks complete rank=%d'
    ),
    'quantized_graph.weight_loading_complete_rank_d_duration_s': (
        'info',
        'weight loading complete rank=%d duration_s=%.2f binding_wait_s=%.2f'
    ),
    'quantized_graph.worker_cuda_communication_initialized_rank_d_duration': (
        'info',
        'worker CUDA communication initialized rank=%d duration_s=%.2f'
    ),
    'quantized_qualification.mtp_composition_proof_rank_d_cases_d': (
        'info',
        'MTP composition proof rank=%d cases=%d receipt=%s duration_s=%.2f'
    ),
    'quantized_qualification.mtp_composition_qualification_rank_d_full_receipt': (
        'info',
        'MTP composition qualification rank=%d full_receipt=%s'
    ),
    'quantized_qualification.mtp_composition_receipt_unavailable_s': (
        'warning',
        'MTP composition receipt unavailable: %s'
    ),
    'quantized_qualification.startup_evidence_prepared_rank_d_operations_d': (
        'info',
        'startup evidence prepared rank=%d operations=%d'
    ),
}


def diagnostic_event(owner: logging.Logger, event: str, *args: object, **kwargs: Any) -> None:
    method, template = _DIAGNOSTIC_EVENTS[event]
    kwargs.setdefault("stacklevel", 2)
    getattr(owner, method)(template, *args, **kwargs)
