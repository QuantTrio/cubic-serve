from __future__ import annotations

import asyncio
import json
import unittest
from unittest.mock import patch

import httpx

from tools.benchmarks.chat_stream import (
    DeviceSample,
    DeviceSampler,
    DeviceSamplingWindow,
    RequestResult,
    RequestSpec,
    arrival_offsets,
    build_parser,
    long_generation_guardrail,
    parse_nvidia_smi_csv,
    run,
    run_level,
    stream_request,
    summarize_device_sampling,
    summarize_level,
)


class _SSEStream(httpx.AsyncByteStream):
    def __init__(self, chunks: tuple[bytes, ...]) -> None:
        self._chunks = chunks

    async def __aiter__(self):  # type: ignore[no-untyped-def]
        for chunk in self._chunks:
            yield chunk


def _event(value: object) -> bytes:
    return b"data: " + json.dumps(value).encode() + b"\n\n"


class ChatStreamBenchmarkTests(unittest.TestCase):
    def test_video_and_audio_benchmark_use_the_common_public_input_protocol(self) -> None:
        payloads = []
        async def handler(request):
            payloads.append(json.loads(request.content))
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                  stream=_SSEStream((
                _event({"choices": [{"delta": {"content": "ok"}}]})
                + _event({"choices": [], "usage": {"prompt_tokens": 70, "completion_tokens": 1}})
                + b"data: [DONE]\n\n",
            )))
        async def invoke():
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                return await stream_request(
                    client, api_url="http://server/v1/chat/completions", model="fixture",
                    spec=RequestSpec(0, 0, "describe", 1024, media_data_urls=(
                        ("video", "data:video/mp4;base64,dmlkZW8="),
                        ("audio", "data:audio/wav;base64,YXVkaW8="))),
                    reasoning_effort="xhigh")
        result = asyncio.run(invoke())
        self.assertTrue(result.success, result.error)
        content = payloads[0]["messages"][0]["content"]
        self.assertEqual([part["type"] for part in content], ["text", "video_url", "audio_url"])
        self.assertEqual(content[1]["video_url"]["url"], "data:video/mp4;base64,dmlkZW8=")
        self.assertEqual(content[2]["audio_url"]["url"], "data:audio/wav;base64,YXVkaW8=")

    def test_image_benchmark_sends_real_media_through_public_chat(self) -> None:
        payloads = []
        async def handler(request):
            payloads.append(json.loads(request.content))
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                  stream=_SSEStream((
                _event({"choices": [{"delta": {"content": "red"}}]})
                + _event({"choices": [], "usage": {"prompt_tokens": 70, "completion_tokens": 1}})
                + b"data: [DONE]\n\n",
            )))
        async def invoke():
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                return await stream_request(
                    client, api_url="http://server/v1/chat/completions", model="fixture",
                    spec=RequestSpec(0, 0, "describe", 1024,
                                     image_data_urls=("data:image/png;base64,cGl4ZWxz",)),
                    reasoning_effort="xhigh",
                )
        result = asyncio.run(invoke())
        self.assertTrue(result.success, result.error)
        self.assertEqual(payloads[0]["messages"][0]["content"], [
            {"type": "text", "text": "describe"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,cGl4ZWxz"}},
        ])
        self.assertEqual(payloads[0]["max_completion_tokens"], 1024)

    def test_paired_namespace_keeps_http_bodies_independent_of_output_filename(self) -> None:
        requests = []
        body = (
            _event({"choices": [{"delta": {"content": "x"}}]})
            + _event({"choices": [], "usage": {"prompt_tokens": 2, "completion_tokens": 1}})
            + b"data: [DONE]\n\n"
        )

        async def handler(request: httpx.Request) -> httpx.Response:
            requests.append(json.loads(request.content))
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, stream=_SSEStream((body,))
            )

        client = httpx.AsyncClient
        with (
            patch("tools.benchmarks.chat_stream.httpx.AsyncClient", side_effect=lambda **kwargs:
                  client(transport=httpx.MockTransport(handler), **kwargs)),
            patch("tools.benchmarks.chat_stream.DeviceSampler", return_value=None),
            patch("tools.benchmarks.chat_stream.secrets.token_hex",
                  side_effect=AssertionError("explicit namespace must not depend on randomness")),
        ):
            for filename in ("reference-c4.json", "candidate-c4.json"):
                args = build_parser().parse_args([
                    "--model", "fixture", "--concurrency", "1,4",
                    "--max-output-tokens", "1", "--prompt", "decode",
                    "--prompt-namespace", "paired-corpus", "--output", filename,
                ])
                report = asyncio.run(run(args))
                self.assertEqual(report["request"]["prompt_namespace"], "paired-corpus")
                self.assertEqual(report["request"]["sampling_seed_base"], 1)
                self.assertEqual(len(report["levels"]), 2)
                self.assertTrue(all(level["failed"] == 0 for level in report["levels"]))
        self.assertEqual(len(requests), 10)
        self.assertEqual(requests[:5], requests[5:])
        self.assertEqual([r["seed"] for r in requests[:5]], [1, 2, 3, 4, 5])
        self.assertEqual(len({r["messages"][0]["content"] for r in requests[:5]}), 5)

    def test_explicit_namespace_rejects_empty_or_shared_prefix_state(self) -> None:
        for extra in (("--prompt-namespace", ""),
                      ("--prompt-namespace", "paired", "--prefix-cache-state", "shared")):
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, "prompt-namespace"):
                asyncio.run(run(build_parser().parse_args(extra)))
        self.assertIsNone(build_parser().parse_args([]).prompt_namespace)

    def test_long_generation_guardrail_rejects_tail_collapse(self) -> None:
        stable = RequestResult(
            0, 0, 0, 1, tuple(index * 0.1 for index in range(10)), 1, 10, 10, 200, True, None
        )
        collapsed = RequestResult(
            1,
            0,
            0,
            3,
            (0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 1.5, 2.0, 2.5),
            1,
            10,
            10,
            200,
            True,
            None,
        )

        accepted = long_generation_guardrail(
            (stable,), minimum_tokens=10, edge_tokens=3, minimum_tail_ratio=0.7
        )
        rejected = long_generation_guardrail(
            (collapsed,), minimum_tokens=10, edge_tokens=3, minimum_tail_ratio=0.7
        )

        self.assertTrue(accepted["passed"])
        self.assertFalse(rejected["passed"])

    def test_nvidia_smi_parser_preserves_missing_portable_metrics(self) -> None:
        samples = parse_nvidia_smi_csv(
            "0, GPU-a, General GPU, 120.5, 87, [N/A], 1500, 900\n",
            observed_at=4.0,
        )

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].index, 0)
        self.assertEqual(samples[0].power_w, 120.5)
        self.assertEqual(samples[0].gpu_utilization_percent, 87)
        self.assertIsNone(samples[0].memory_activity_percent)

    def test_device_summary_is_per_gpu_windowed_and_relative_to_idle(self) -> None:
        def sample(at: float, index: int, power: float, gpu: float, memory: float) -> DeviceSample:
            return DeviceSample(
                at,
                index,
                f"GPU-{index}",
                "General GPU",
                power,
                gpu,
                memory,
                1500,
                900,
            )

        baseline = DeviceSamplingWindow(
            (sample(9.0, 0, 50, 0, 0), sample(9.1, 0, 70, 10, 20)),
            query_count=2,
            failed_query_count=0,
            errors=(),
        )
        request = DeviceSamplingWindow(
            (
                sample(10.1, 0, 150, 80, 60),
                sample(10.9, 0, 170, 100, 80),
                sample(12.1, 0, 999, 0, 0),
            ),
            query_count=3,
            failed_query_count=0,
            errors=(),
        )

        summary = summarize_device_sampling(
            baseline=baseline,
            request=request,
            origin=10,
            ended_at=12,
            interval_s=0.25,
            requested_device_indices=(0,),
        )

        self.assertEqual(summary["status"], "ok")
        devices = summary["devices"]
        assert isinstance(devices, list)
        device = devices[0]
        assert isinstance(device, dict)
        active = device["request_window"]
        delta = device["mean_delta_from_idle"]
        assert isinstance(active, dict)
        assert isinstance(delta, dict)
        self.assertEqual(active["sample_count"], 2)
        self.assertEqual(active["power_w"], {"mean": 160, "p95": 169, "peak": 170})
        self.assertEqual(delta["power_w"], 100)
        self.assertEqual(delta["gpu_utilization_percentage_points"], 85)

    def test_continuous_usage_counts_hidden_tokens_without_text_retokenization(self) -> None:
        chunks = tuple(_event({"choices": [{"delta": {"content": text}}],
                               "usage": {"prompt_tokens": 7, "completion_tokens": count}})
                       for text, count in (("", 1), ("字", 2), ("", 3), ("tail", 3)))
        chunks += (_event({"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 3}}), b"data: [DONE]\n\n")
        async def handler(request):
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=_SSEStream(chunks))
        async def invoke():
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                return await stream_request(client, api_url="http://server/v1/chat/completions", model="fixture", spec=RequestSpec(0, 0, "hello", 3), reasoning_effort=None)
        result = asyncio.run(invoke())
        self.assertTrue(result.success, result.error)
        self.assertEqual(len(result.output_arrivals), 3)
        self.assertEqual(result.output_tokens, 3)

    def test_arrival_offsets_cover_burst_fixed_and_seeded_poisson(self) -> None:
        self.assertEqual(arrival_offsets(3, pattern="burst", request_rate=None, seed=1), (0, 0, 0))
        self.assertEqual(arrival_offsets(3, pattern="fixed", request_rate=2, seed=1), (0, 0.5, 1))
        self.assertEqual(
            arrival_offsets(4, pattern="poisson", request_rate=3, seed=9),
            arrival_offsets(4, pattern="poisson", request_rate=3, seed=9),
        )

    def test_stream_request_uses_public_chat_sse_and_usage_token_count(self) -> None:
        seen_payload: dict[str, object] = {}
        chunks = (
            b": ping\n\n" + _event({"choices": [{"delta": {"role": "assistant", "content": ""}}]}),
            _event({"choices": [{"delta": {"reasoning_content": "想"}}]})[:19],
            _event({"choices": [{"delta": {"reasoning_content": "想"}}]})[19:]
            + _event({"choices": [{"delta": {"content": "答"}}]}),
            _event({"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 2}})
            + b"data: [DONE]\n\n",
        )

        async def handler(request: httpx.Request) -> httpx.Response:
            seen_payload.update(json.loads(request.content))
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream; charset=utf-8"},
                stream=_SSEStream(chunks),
            )

        async def invoke() -> RequestResult:
            transport = httpx.MockTransport(handler)
            async with httpx.AsyncClient(transport=transport) as client:
                return await stream_request(
                    client,
                    api_url="http://server/v1/chat/completions",
                    model="fixture",
                    spec=RequestSpec(0, 0, "hello", 8),
                    reasoning_effort="xhigh",
                )

        result = asyncio.run(invoke())

        self.assertTrue(result.success, result.error)
        self.assertEqual((result.input_tokens, result.output_tokens), (7, 2))
        self.assertEqual(len(result.output_arrivals), 2)
        self.assertEqual(result.content_characters, 2)
        self.assertEqual(seen_payload["stream"], True)
        self.assertEqual(seen_payload["stream_options"], {"include_usage": True, "continuous_usage_stats": True})
        self.assertEqual(seen_payload["reasoning_effort"], "xhigh")
        self.assertEqual(seen_payload["max_completion_tokens"], 8)

    def test_stream_request_keeps_usage_authoritative_when_chunks_bundle_tokens(self) -> None:
        body = (
            _event({"choices": [{"delta": {"content": "two tokens"}}]})
            + _event({"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 2}})
            + b"data: [DONE]\n\n"
        )

        async def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=_SSEStream((body,)),
            )

        async def invoke() -> RequestResult:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                return await stream_request(
                    client,
                    api_url="http://server/v1/chat/completions",
                    model="fixture",
                    spec=RequestSpec(0, 0, "hello", 2),
                    reasoning_effort=None,
                )

        result = asyncio.run(invoke())
        self.assertTrue(result.success, result.error)
        self.assertEqual(result.output_tokens, 2)
        self.assertEqual(len(result.output_arrivals), 1)
        summary = summarize_level(
            (result,),
            concurrency=1,
            origin=result.started_at,
            ended_at=result.ended_at,
            window_s=1,
        )
        self.assertFalse(summary["window_timeline_exact"])
        windows = summary["windows"]
        assert isinstance(windows, list)
        self.assertEqual(
            sum(window["output_tokens"] for window in windows if isinstance(window, dict)),
            2,
        )

    def test_summary_reports_per_request_and_window_token_throughput(self) -> None:
        origin = 10.0
        results = (
            RequestResult(0, 0, 10.0, 12.0, (10.5, 11.0, 11.5), 4, 3, 3, 200, True, None),
            RequestResult(1, 0, 10.0, 12.0, (10.6, 11.1, 11.6), 4, 3, 3, 200, True, None),
        )

        summary = summarize_level(
            results, concurrency=2, origin=origin, ended_at=12.0, window_s=1.0
        )

        self.assertTrue(summary["canonical_valid"])
        self.assertEqual(summary["max_active_requests"], 2)
        self.assertEqual(summary["total_output_tokens"], 6)
        self.assertEqual(
            summary["batch_makespan_output_tokens_per_second_diagnostic"], 3
        )
        windows = summary["windows"]
        requests = summary["requests"]
        self.assertIsInstance(windows, list)
        self.assertIsInstance(requests, list)
        assert isinstance(windows, list)
        assert isinstance(requests, list)
        self.assertEqual(
            [window["output_tokens"] for window in windows if isinstance(window, dict)],
            [2, 4],
        )
        first = requests[0]
        assert isinstance(first, dict)
        self.assertEqual(first["ttft_ms"], 500)
        self.assertEqual(first["tpot_ms"], 500)
        self.assertAlmostEqual(first["decode_tokens_per_second"], 2)

    def test_staggered_prefill_is_an_extra_labeled_public_request(self) -> None:
        prompts: list[str] = []
        body = (
            _event({"choices": [{"delta": {"content": "x"}}]})
            + _event({"choices": [], "usage": {"prompt_tokens": 2, "completion_tokens": 1}})
            + b"data: [DONE]\n\n"
        )

        async def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            prompts.append(payload["messages"][0]["content"])
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=_SSEStream((body,)),
            )

        async def invoke() -> dict[str, object]:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                return await run_level(
                    client,
                    api_url="http://server/v1/chat/completions",
                    model="fixture",
                    prompt="decode",
                    max_output_tokens=1,
                    concurrency=2,
                    request_multiplier=1,
                    arrival_pattern="burst",
                    request_rate=None,
                    seed=1,
                    reasoning_effort="xhigh",
                    window_s=1,
                    staggered_prefill_prompt="long-prefill",
                    staggered_prefill_at_s=0,
                )

        summary = asyncio.run(invoke())
        self.assertEqual(sorted(prompts), ["decode", "decode", "long-prefill"])
        requests = summary["requests"]
        assert isinstance(requests, list)
        self.assertEqual(
            [request["kind"] for request in requests if isinstance(request, dict)],
            ["regular", "regular", "staggered_prefill"],
        )

    def test_prompt_namespace_isolates_every_request_prefix(self) -> None:
        prompts: list[str] = []
        body = (
            _event({"choices": [{"delta": {"content": "x"}}]})
            + _event({"choices": [], "usage": {"prompt_tokens": 2, "completion_tokens": 1}})
            + b"data: [DONE]\n\n"
        )

        async def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            prompts.append(payload["messages"][0]["content"])
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=_SSEStream((body,)),
            )

        async def invoke() -> None:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                await run_level(
                    client,
                    api_url="http://server/v1/chat/completions",
                    model="fixture",
                    prompt="decode",
                    max_output_tokens=1,
                    concurrency=2,
                    request_multiplier=1,
                    arrival_pattern="burst",
                    request_rate=None,
                    seed=1,
                    reasoning_effort="xhigh",
                    window_s=1,
                    prompt_namespace="run-a",
                )

        asyncio.run(invoke())
        self.assertEqual(len(set(prompts)), 2)
        self.assertTrue(all(prompt.endswith("\ndecode") for prompt in prompts))

    def test_unavailable_device_sampling_does_not_change_request_result(self) -> None:
        body = (
            _event({"choices": [{"delta": {"content": "x"}}]})
            + _event({"choices": [], "usage": {"prompt_tokens": 2, "completion_tokens": 1}})
            + b"data: [DONE]\n\n"
        )

        async def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=_SSEStream((body,)),
            )

        def unavailable(
            device_indices: tuple[int, ...] | None,
        ) -> tuple[DeviceSample, ...]:
            del device_indices
            raise FileNotFoundError("nvidia-smi")

        async def invoke() -> dict[str, object]:
            sampler = DeviceSampler(device_indices=(0,), interval_s=0.001, query=unavailable)
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                return await run_level(
                    client,
                    api_url="http://server/v1/chat/completions",
                    model="fixture",
                    prompt="decode",
                    max_output_tokens=1,
                    concurrency=1,
                    request_multiplier=1,
                    arrival_pattern="burst",
                    request_rate=None,
                    seed=1,
                    reasoning_effort="xhigh",
                    window_s=1,
                    device_sampler=sampler,
                    device_idle_s=0.001,
                )

        summary = asyncio.run(invoke())
        self.assertTrue(summary["canonical_valid"])
        telemetry = summary["device_telemetry"]
        assert isinstance(telemetry, dict)
        self.assertEqual(telemetry["status"], "unavailable")
        self.assertEqual(telemetry["devices"], [])
        self.assertTrue(telemetry["errors"])


if __name__ == "__main__":
    unittest.main()
