from __future__ import annotations

import base64
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from cubic_serve.inputs.chat import ChatProcessorError, MediaResolver
from cubic_serve.inputs.normalize import ChatNormalizationError, normalize_openai_chat_completions

_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAACXBIWXMAAAABAAAAAQBPJcTW"
    "AAAAKUlEQVR4nO3NsQkAAAzDsPz/c6FPJJvAs5VLpm3vAAAAAAAAAAAAoNgDp+T0Ltwd/VwAAAAASUVORK5CYII="
)


def _request(part: dict[str, object]):
    request, _ = normalize_openai_chat_completions(
        {
            "model": "fixture",
            "max_tokens": 1,
            "messages": [{"role": "user", "content": [part]}],
        },
        request_id="media-request",
    )
    return request


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg required")
class MediaResolverTests(unittest.TestCase):
    def test_inline_png_is_realized_from_its_actual_bytes(self) -> None:
        request = _request(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{_PNG}"},
            }
        )
        with patch("cubic_serve.inputs.chat.subprocess.run") as run:
            media = MediaResolver().realize(request)

        self.assertEqual(len(media), 1)
        self.assertEqual(tuple(media[0].frames.shape), (1, 32, 32, 3))
        self.assertEqual(media[0].part.media.byte_length, len(base64.b64decode(_PNG)))
        run.assert_not_called()

    def test_file_video_requires_and_stays_below_the_allowed_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "clip.mp4"
            subprocess.run(
                (
                    "ffmpeg",
                    "-y",
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=blue:s=32x32:r=2:d=2",
                    str(video),
                ),
                check=True,
            )
            request = _request({"type": "video_url", "video_url": {"url": video.as_uri()}})
            with self.assertRaisesRegex(ChatProcessorError, "allowed-local-media-path"):
                MediaResolver().realize(request)
            with self.assertRaisesRegex(ChatProcessorError, "byte size"):
                MediaResolver(allowed_local_media_path=root, max_bytes=1).realize(request)

            media = MediaResolver(allowed_local_media_path=root).realize(request)
            self.assertEqual(tuple(media[0].frames.shape), (4, 32, 32, 3))
            self.assertEqual(media[0].frame_indices, (0, 1, 2, 3))

    def test_missing_local_media_is_an_invalid_request(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = _request(
                {
                    "type": "video_url",
                    "video_url": {"url": (root / "missing.mp4").as_uri()},
                }
            )
            with self.assertRaisesRegex(ChatProcessorError, "does not exist"):
                MediaResolver(allowed_local_media_path=root).realize(request)

    def test_inline_mp4_is_realized_from_its_actual_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "clip.mp4"
            subprocess.run(
                (
                    "ffmpeg",
                    "-y",
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=blue:s=32x32:r=2:d=2",
                    str(video),
                ),
                check=True,
            )
            encoded = base64.b64encode(video.read_bytes()).decode("ascii")
            request = _request(
                {
                    "type": "video_url",
                    "video_url": {"url": f"data:video/mp4;base64,{encoded}"},
                }
            )

            media = MediaResolver().realize(request)

            self.assertEqual(tuple(media[0].frames.shape), (4, 32, 32, 3))
            self.assertEqual(media[0].frame_indices, (0, 1, 2, 3))
            self.assertEqual(media[0].part.media.byte_length, video.stat().st_size)

    def test_http_is_explicitly_fail_closed(self) -> None:
        request = _request({"type": "image_url", "image_url": {"url": "https://example.com/a.png"}})
        with self.assertRaisesRegex(ChatProcessorError, "not implemented"):
            MediaResolver().realize(request)

    def test_inline_media_is_bounded_before_base64_decode(self) -> None:
        request = _request(
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,QUJDRA=="},
            }
        )
        with self.assertRaisesRegex(ChatProcessorError, "encoded byte limit"):
            MediaResolver(max_bytes=3).realize(request)

    def test_mime_type_must_match_the_declared_modality(self) -> None:
        with self.assertRaisesRegex(ChatNormalizationError, "must describe image"):
            _request(
                {
                    "type": "image_url",
                    "image_url": {"url": "data:video/mp4;base64,QUJD"},
                }
            )

    def test_decoded_pixel_budget_is_checked_before_capture(self) -> None:
        request = _request(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{_PNG}"},
            }
        )
        with self.assertRaisesRegex(ChatProcessorError, "pixel limit"):
            MediaResolver(max_decoded_pixels=32 * 32 - 1).realize(request)

    def test_invalid_probe_metadata_is_an_invalid_request(self) -> None:
        request = _request(
            {
                "type": "video_url",
                "video_url": {"url": "data:video/mp4;base64,QUJDRA=="},
            }
        )
        probe_outputs = (
            b'{"streams":[]}',
            b'{"streams":[{"width":32,"height":32,"avg_frame_rate":"0/1"}]}',
        )
        for output in probe_outputs:
            with (
                self.subTest(output=output),
                patch(
                    "cubic_serve.inputs.chat.subprocess.run",
                    return_value=SimpleNamespace(stdout=output),
                ),
                self.assertRaisesRegex(ChatProcessorError, "could not be decoded"),
            ):
                MediaResolver().realize(request)

    def test_audio_enters_wire_contract_but_decoder_rejects_it(self) -> None:
        request = _request(
            {
                "type": "audio_url",
                "audio_url": {"url": "data:audio/wav;base64,UklGRg=="},
            }
        )
        self.assertEqual(request.messages[0].content_parts[0].modality, "audio")
        with self.assertRaisesRegex(ChatProcessorError, "audio"):
            MediaResolver().realize(request)

    def test_complete_video_coverage_and_low_fps_keep_tail_and_real_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "red-blue.mp4"
            subprocess.run(("ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                            "color=red:s=32x32:r=2:d=3", "-f", "lavfi", "-i",
                            "color=blue:s=32x32:r=2:d=3", "-filter_complex",
                            "[0:v][1:v]concat=n=2:v=1:a=0", str(video)), check=True)
            request = _request({"type": "video_url", "video_url": {"url": video.as_uri()}})
            with self.assertRaisesRegex(ChatProcessorError, "complete video sampling"):
                MediaResolver(allowed_local_media_path=root, max_frames=4).realize(request)
            result = MediaResolver(allowed_local_media_path=root, max_frames=4,
                                   video_fps=0.5).realize(request)[0]
            self.assertEqual(tuple(result.frames.shape), (4, 32, 32, 3))
            self.assertEqual(result.frame_indices, (0, 1, 2, 2))
            self.assertEqual(result.frames_per_second, 0.5)
            self.assertEqual(result.frame_timestamps_seconds, (0.0, 2.0, 4.0, 4.0))
            self.assertEqual(result.part.media.duration_ms, 6000)
            self.assertGreater(result.frames[0, :, :, 0].float().mean(), 240)
            self.assertGreater(result.frames[-1, :, :, 2].float().mean(), 240)
            from cubic_serve.models.qwen3_5.multimodal import Qwen3_5VideoMetadata
            self.assertEqual(Qwen3_5VideoMetadata(result.frame_indices, 0.5).timestamps(2),
                             (1.0, 4.0))

    def test_noncommensurate_source_fps_retains_selected_pts_instead_of_nominal_clock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "three-fps.mp4"
            subprocess.run(("ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                            "color=red:s=32x32:r=3:d=2", str(video)), check=True)
            request = _request({"type": "video_url", "video_url": {"url": video.as_uri()}})
            result = MediaResolver(allowed_local_media_path=root, max_frames=4).realize(request)[0]
            self.assertEqual(result.frame_timestamps_seconds, (0.0, 2/3, 1.0, 5/3))
            from cubic_serve.models.qwen3_5.multimodal import Qwen3_5VideoMetadata
            clock = Qwen3_5VideoMetadata(result.frame_indices, result.frames_per_second,
                                       result.frame_timestamps_seconds)
            self.assertEqual(tuple(f"<{t:.1f} seconds>" for t in clock.timestamps(2)),
                             ("<0.3 seconds>", "<1.3 seconds>"))
            self.assertNotEqual(clock.timestamps(2), Qwen3_5VideoMetadata(
                result.frame_indices, result.frames_per_second).timestamps(2))

    def test_unknown_or_underreported_video_duration_cannot_admit_a_prefix(self) -> None:
        request = _request({"type": "video_url",
                            "video_url": {"url": "data:video/mp4;base64,QUJDRA=="}})
        for duration in (None, "1"):
            stream = {"width": 2, "height": 2, "avg_frame_rate": "2/1"}
            if duration is not None:
                stream["duration"] = duration
            import json
            probe = SimpleNamespace(stdout=json.dumps({"streams": [stream]}).encode())
            # The fifth sentinel frame means the four-frame sampling is incomplete.
            decode = SimpleNamespace(stdout=bytes(5 * 2 * 2 * 3))
            with self.subTest(duration=duration), patch(
                "cubic_serve.inputs.chat.subprocess.run", side_effect=(probe, decode)
            ) as run, self.assertRaisesRegex(ChatProcessorError, "complete video sampling"):
                MediaResolver(max_frames=4).realize(request)
            self.assertEqual(run.call_args.args[0][-7:-5], ("-frames:v", "5"))


if __name__ == "__main__":
    unittest.main()
