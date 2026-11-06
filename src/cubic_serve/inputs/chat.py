"""Common chat input contracts and bounded media ingress."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import mimetypes
import re
import subprocess
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from io import BytesIO
from pathlib import Path
from typing import Literal, Protocol
from urllib.parse import unquote, urlsplit

import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from cubic_serve.contracts.model import WeightVariantMeta
from cubic_serve.contracts.request import (
    CanonicalChatRequest,
    ContentPart,
    InputPlan,
    MediaDescriptor,
    MediaSource,
    MediaSourceKind,
)
from cubic_serve.inputs.normalize import MAX_MEDIA_BYTES

ReasoningEffort = Literal["xhigh", "medium", "low"]


class ChatProcessorError(ValueError):
    """The local checkpoint cannot produce one exact supported input plan."""

    def __init__(self, message: str, *, public_message: str | None = None) -> None:
        super().__init__(message)
        self.public_message = public_message or "request input could not be processed"


@dataclass(frozen=True, slots=True)
class DecodedMedia:
    """Trusted RGB frames produced by the common bounded media ingress."""

    part: ContentPart
    frames: torch.Tensor
    frame_indices: tuple[int, ...]
    frames_per_second: float
    frame_timestamps_seconds: tuple[float, ...] = ()


class MediaResolver:
    """Resolve inline or explicitly authorized local media without network I/O."""

    def __init__(
        self,
        *,
        allowed_local_media_path: str | Path | None = None,
        max_bytes: int = MAX_MEDIA_BYTES,
        max_frames: int = 768,
        max_decoded_pixels: int = 64 << 20,
        video_fps: float = 2.0,
    ) -> None:
        if (isinstance(video_fps, bool) or not math.isfinite(video_fps) or video_fps <= 0):
            raise ChatProcessorError("video sampling FPS must be finite and positive")
        if type(max_frames) is not int or max_frames < 4:
            raise ChatProcessorError("video max frames must be an integer of at least four")
        root = None
        if allowed_local_media_path is not None:
            root = Path(allowed_local_media_path).resolve(strict=True)
            if not root.is_dir():
                raise ChatProcessorError("allowed local media path must be a directory")
        self._root = root
        self._max_bytes = max_bytes
        self._max_frames = max_frames
        self._max_decoded_pixels = max_decoded_pixels
        self._video_fps = video_fps

    def realize(self, request: CanonicalChatRequest) -> tuple[DecodedMedia, ...]:
        realized: list[DecodedMedia] = []
        for message in request.messages:
            for part in message.content_parts:
                if part.modality == "text":
                    continue
                if part.modality not in {"image", "video"} or part.media is None:
                    raise ChatProcessorError(
                        f"media decoder does not support {part.modality!r} input"
                    )
                payload, path, mime_type = self._read(part.media, part.modality)
                try:
                    realized.append(
                        self._decode(
                            ordinal=part.ordinal,
                            modality=part.modality,
                            mime_type=mime_type,
                            payload=payload,
                            path=path,
                        )
                    )
                finally:
                    payload = b""
        return tuple(realized)

    def _read(self, source: MediaSource, modality: str) -> tuple[bytes, Path | None, str]:
        if source.kind is MediaSourceKind.BASE64:
            assert source.base64_data is not None and source.mime_type is not None
            self._validate_mime(modality, source.mime_type)
            encoded_limit = 4 * ((self._max_bytes + 2) // 3)
            if not source.base64_data or len(source.base64_data) > encoded_limit:
                raise ChatProcessorError("inline media exceeds the encoded byte limit")
            try:
                payload = base64.b64decode(source.base64_data, validate=True)
            except ValueError as error:
                raise ChatProcessorError("inline media contains invalid base64") from error
            return self._bounded(payload), None, source.mime_type
        assert source.uri is not None
        parsed = urlsplit(source.uri)
        if parsed.scheme.lower() in {"http", "https"}:
            raise ChatProcessorError("HTTP(S) media loading is not implemented")
        if parsed.scheme.lower() != "file" or self._root is None:
            raise ChatProcessorError("local media requires file:// and --allowed-local-media-path")
        try:
            path = Path(unquote(parsed.path)).resolve(strict=True)
        except OSError as error:
            raise ChatProcessorError("local media file does not exist") from error
        try:
            path.relative_to(self._root)
        except ValueError as error:
            raise ChatProcessorError("local media path is outside the allowed root") from error
        if not path.is_file():
            raise ChatProcessorError("local media path must be a regular file")
        mime_type = mimetypes.guess_type(path.name)[0] or f"{modality}/unknown"
        self._validate_mime(modality, mime_type)
        try:
            size = path.stat().st_size
            if size <= 0 or size > self._max_bytes:
                raise ChatProcessorError("media byte size is outside the admitted range")
            with path.open("rb") as file:
                payload = file.read(self._max_bytes + 1)
        except OSError as error:
            raise ChatProcessorError("local media could not be read") from error
        return self._bounded(payload), path, mime_type

    @staticmethod
    def _validate_mime(modality: str, mime_type: str) -> None:
        if not mime_type.lower().startswith(f"{modality}/"):
            raise ChatProcessorError("media MIME type differs from its request modality")

    def _bounded(self, payload: bytes) -> bytes:
        if not payload or len(payload) > self._max_bytes:
            raise ChatProcessorError("media byte size is outside the admitted range")
        return payload

    def _decode(
        self,
        *,
        ordinal: int,
        modality: str,
        mime_type: str,
        payload: bytes,
        path: Path | None,
    ) -> DecodedMedia:
        if modality == "image":
            return self._decode_image(
                ordinal=ordinal,
                mime_type=mime_type,
                payload=payload,
            )
        suffix = path.suffix if path is not None else ".bin"
        with tempfile.NamedTemporaryFile(suffix=suffix) as temporary:
            media_path = path
            if media_path is None:
                temporary.write(payload)
                temporary.flush()
                media_path = Path(temporary.name)
            try:
                probe = subprocess.run(
                    (
                        "ffprobe",
                        "-v",
                        "error",
                        "-select_streams",
                        "v:0",
                        "-show_entries",
                        "stream=width,height,avg_frame_rate,nb_frames,duration:format=duration",
                        "-of",
                        "json",
                        str(media_path),
                    ),
                    check=True,
                    capture_output=True,
                    timeout=30,
                )
                metadata = json.loads(probe.stdout)
                stream = metadata["streams"][0]
                width, height = int(stream["width"]), int(stream["height"])
                frame_pixels = width * height
                if frame_pixels <= 0 or frame_pixels > self._max_decoded_pixels:
                    raise ChatProcessorError("decoded media exceeds the frame or pixel limit")
                rate = float(Fraction(stream.get("avg_frame_rate") or "1"))
                if not math.isfinite(rate) or rate <= 0:
                    raise ValueError("media frame rate must be positive")
                duration = None
                for value in (stream.get("duration"), metadata.get("format", {}).get("duration")):
                    if value not in (None, "N/A"):
                        candidate = float(value)
                        if not math.isfinite(candidate) or candidate <= 0:
                            raise ValueError("media duration must be finite and positive")
                        duration = candidate
                        break
                if duration is None and stream.get("nb_frames") not in (None, "N/A"):
                    duration = int(stream["nb_frames"]) / rate
                    if duration <= 0:
                        raise ValueError("media frame count must be positive")
                decode_command = [
                    "ffmpeg",
                    "-v",
                    "info",
                    "-nostats",
                    "-i",
                    str(media_path),
                    "-map",
                    "0:v:0",
                ]
                if modality == "image":
                    decode_command.extend(("-frames:v", "1"))
                else:
                    frame_limit = min(
                        self._max_frames,
                        self._max_decoded_pixels // frame_pixels,
                    )
                    if frame_limit < 4:
                        raise ChatProcessorError("decoded media exceeds the frame or pixel limit")
                    rate = self._video_fps
                    source_count = (int(stream["nb_frames"])
                                    if stream.get("nb_frames") not in (None, "N/A") else None)
                    expected_frames = (math.ceil(duration * rate)
                                       if duration is not None else None)
                    if expected_frames is not None and source_count is not None:
                        expected_frames = min(expected_frames, source_count)
                    if expected_frames is not None and expected_frames > frame_limit:
                        raise ChatProcessorError(
                            f"complete video sampling at {rate:g} FPS exceeds the "
                            f"{frame_limit}-frame/pixel limit; lower --video-fps or "
                            "increase --video-max-frames within the pixel budget")
                    # One extra decoded frame detects absent/incorrect duration
                    # metadata. It is never admitted or returned as a prefix.
                    # Choose the first source frame in each cadence interval.
                    # Preserve source PTS; do not synthesize CFR duplicates.
                    select = (rf"setpts=PTS-STARTPTS,select=isnan(prev_selected_t)+"
                              rf"gt(floor(t*{rate:.12g})\,floor(prev_selected_t*{rate:.12g})),"
                              "showinfo")
                    decode_command.extend(("-vf", select, "-vsync", "0", "-frames:v",
                                           str(frame_limit + 1)))
                decode_command.extend(("-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"))
                decoded_process = subprocess.run(
                    tuple(decode_command),
                    check=True,
                    capture_output=True,
                    timeout=120,
                )
                decoded = decoded_process.stdout
                frame_log = getattr(decoded_process, "stderr", b"")
                time_base = re.search(rb"config in time_base:\s*(\d+)/(\d+)", frame_log)
                timestamps = ()
                if time_base is not None:
                    tick = Fraction(int(time_base[1]), int(time_base[2]))
                    timestamps = tuple(float(int(value) * tick) for value in re.findall(
                        rb"\bn:\s*\d+\s+pts:\s*(-?\d+)\s+pts_time:", frame_log))
            except ChatProcessorError:
                raise
            except (
                FileNotFoundError,
                subprocess.SubprocessError,
                IndexError,
                KeyError,
                ValueError,
                ZeroDivisionError,
                json.JSONDecodeError,
            ) as error:
                raise ChatProcessorError("media could not be decoded by ffmpeg") from error
        frame_bytes = width * height * 3
        if frame_bytes <= 0 or len(decoded) % frame_bytes:
            raise ChatProcessorError("decoded media has an invalid RGB layout")
        frame_count = len(decoded) // frame_bytes
        if modality == "video" and frame_count > frame_limit:
            raise ChatProcessorError(
                "complete video sampling exceeds the frame/pixel limit; "
                "lower --video-fps or increase --video-max-frames within the pixel budget")
        if frame_count <= 0 or frame_count * width * height > self._max_decoded_pixels:
            raise ChatProcessorError("decoded media exceeds the frame or pixel limit")
        if modality == "image" and frame_count != 1:
            raise ChatProcessorError("image input must decode to exactly one frame")
        if (len(timestamps) < frame_count or any(not math.isfinite(t) or t < 0
                                               for t in timestamps[:frame_count]) or
                tuple(sorted(timestamps[:frame_count])) != timestamps[:frame_count]):
            raise ChatProcessorError("decoded video lacks ordered source frame timestamps")
        timestamps = timestamps[:frame_count]
        frames = torch.from_numpy(
            np.frombuffer(decoded, dtype=np.uint8).copy().reshape(frame_count, height, width, 3)
        )
        frame_indices = tuple(range(frame_count))
        if modality == "video" and frame_count < 4:
            frames = torch.cat((frames, frames[-1:].expand(4 - frame_count, -1, -1, -1)))
            frame_indices += (frame_count - 1,) * (4 - frame_count)
            timestamps += (timestamps[-1],) * (4 - frame_count)
            frame_count = 4
        digest = f"b2s256:{hashlib.blake2s(payload).hexdigest()}"
        duration_ms = max(1, round(1000 * (duration or len(set(frame_indices)) / rate)))
        descriptor = MediaDescriptor(
            digest,
            mime_type.lower(),
            len(payload),
            width,
            height,
            frame_count if modality == "video" else None,
            duration_ms if modality == "video" else None,
        )
        return DecodedMedia(
            ContentPart(ordinal, modality, media=descriptor),
            frames,
            frame_indices,
            rate,
            timestamps,
        )

    def _decode_image(
        self,
        *,
        ordinal: int,
        mime_type: str,
        payload: bytes,
    ) -> DecodedMedia:
        """Decode one bounded image in-process; video alone uses ffmpeg."""

        self._validate_mime("image", mime_type)
        try:
            with Image.open(BytesIO(payload)) as source:
                detected_mime = Image.MIME.get(source.format or "")
                if detected_mime is None or not detected_mime.lower().startswith("image/"):
                    raise ChatProcessorError("encoded payload is not a supported image")
                width, height = source.size
                if width <= 0 or height <= 0 or width * height > self._max_decoded_pixels:
                    raise ChatProcessorError("decoded media exceeds the frame or pixel limit")
                image = ImageOps.exif_transpose(source).convert("RGB")
                frames = torch.from_numpy(np.asarray(image, dtype=np.uint8).copy()).unsqueeze(0)
        except ChatProcessorError:
            raise
        except (OSError, UnidentifiedImageError, ValueError) as error:
            raise ChatProcessorError("media could not be decoded as an image") from error
        height, width = int(frames.shape[1]), int(frames.shape[2])
        digest = f"b2s256:{hashlib.blake2s(payload).hexdigest()}"
        descriptor = MediaDescriptor(digest, detected_mime.lower(), len(payload), width, height)
        return DecodedMedia(
            ContentPart(ordinal, "image", media=descriptor),
            frames,
            (0,),
            1.0,
        )


def resolve_chat_template_kwargs(
    request: CanonicalChatRequest,
    *,
    default_enable_thinking: bool,
) -> tuple[dict[str, object], bool]:
    """Resolve effective public template kwargs and the parser's thinking mode."""

    if not isinstance(request, CanonicalChatRequest):
        raise TypeError("Qwen chat-template kwargs require a CanonicalChatRequest")
    if type(default_enable_thinking) is not bool:
        raise TypeError("default_enable_thinking must be a bool")
    kwargs = request.chat_template_kwargs.as_dict()
    enable_thinking = kwargs.get("enable_thinking", default_enable_thinking)
    if type(enable_thinking) is not bool:
        raise ChatProcessorError("chat template enable_thinking must be a boolean")
    kwargs["enable_thinking"] = enable_thinking
    # Tool definitions are prompt semantics, independently of whether the next
    # answer is constrained or tool_choice disabled tool generation.
    if request.tools.canonical_payload != "[]":
        kwargs["tools"] = request.tools.as_list()
    return kwargs, enable_thinking


class ChatInputProcessor(Protocol):
    """Input realization required by either chat execution backend."""

    def encoded_vocabulary(self, *, vocab_size: int) -> tuple[str | bytes, ...]: ...

    def build_chat_input_plan(
        self,
        request: CanonicalChatRequest,
        *,
        weight_variant: WeightVariantMeta,
        add_generation_prompt: bool = True,
        enable_thinking: bool = True,
        reasoning_effort: ReasoningEffort = "xhigh",
        preserve_thinking: bool = True,
    ) -> InputPlan: ...


__all__ = (
    "ChatInputProcessor",
    "ChatProcessorError",
    "ReasoningEffort",
    "resolve_chat_template_kwargs",
)
