from __future__ import annotations

import logging
import unittest

import torch

from cubic_serve.runtime.profiling import CudaStepProfiler, StepProfile


class StepProfileTest(unittest.TestCase):
    def test_summary_keeps_values_separate_from_milliseconds(self) -> None:
        profile = StepProfile(
            "controller",
            anchor_metric="commit_ms",
            sample_steps=2,
            schema="test",
            logger=logging.getLogger(__name__),
        )
        profile.record_value("target_rows", 128)
        profile.record("commit_ms", 1.0)
        profile.record_value("target_rows", 256)
        profile.record("commit_ms", 2.0)

        summary = profile.summary()

        self.assertEqual(summary["values"]["target_rows"]["sum"], 384.0)
        self.assertEqual(summary["values"]["target_rows"]["p50"], 128.0)
        self.assertNotIn("target_rows", summary["metrics"])
        self.assertEqual(summary["metrics"]["commit_ms"]["sum_ms"], 3.0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_inflight_timing_does_not_wait_or_read_incomplete_events(self) -> None:
        profile = StepProfile(
            "worker", anchor_metric="commit_ms", sample_steps=2,
            schema="test", logger=logging.getLogger(__name__),
        )
        device = torch.device("cuda", torch.cuda.current_device())
        profiler = CudaStepProfiler(profile, device, ("async_ms",))
        stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(stream):
            profiler.begin_cuda("async_ms")
            torch.cuda._sleep(100_000_000)
            profiler.end_cuda("async_ms")
        profiler.flush_cuda(("async_ms",))
        self.assertFalse(stream.query(), "timing collection synchronized the work")
        summary = profile.summary()
        self.assertNotIn("async_ms", summary["metrics"])
        self.assertEqual(summary["values"]["async_ms_incomplete_samples"]["sum"], 1)
        stream.synchronize()
        profiler.flush_cuda(("async_ms",))
        self.assertEqual(profile.summary()["metrics"]["async_ms"]["count"], 1)


if __name__ == "__main__":
    unittest.main()
