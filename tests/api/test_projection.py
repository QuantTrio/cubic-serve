from __future__ import annotations

import unittest
from dataclasses import replace

from cubic_serve.api import (
    ProjectionError,
    project_anthropic_message,
    project_anthropic_message_stream,
    project_openai_chat_completion,
    project_openai_chat_completion_stream,
)
from cubic_serve.contracts.request import TokenEvent, TokenEventKind

Payload = tuple[tuple[str, str], ...]
EventSpec = tuple[TokenEventKind, str | None, Payload, int | None]


def _sequence(*specs: EventSpec) -> tuple[TokenEvent, ...]:
    return tuple(
        TokenEvent(
            request_id="request-1",
            sequence_no=index,
            kind=kind,
            text=text,
            payload_items=payload,
            token_id=token_id,
        )
        for index, (kind, text, payload, token_id) in enumerate(specs)
    )


def _full_events() -> tuple[TokenEvent, ...]:
    return _sequence(
        (TokenEventKind.REASONING_DELTA, "think ", (), None),
        (TokenEventKind.REASONING_DELTA, "now", (), None),
        (TokenEventKind.TEXT_DELTA, "answer", (), None),
        (TokenEventKind.TEXT_DELTA, "", (), None),
        (
            TokenEventKind.TOOL_CALL_DELTA,
            '{"city":',
            (("id", "call-0"), ("index", "0"), ("name", "weather")),
            None,
        ),
        (TokenEventKind.TOOL_CALL_DELTA, '"Paris"}', (("index", "0"),), None),
        (
            TokenEventKind.TOOL_CALL_DELTA,
            "{}",
            (("id", "call-1"), ("index", "1"), ("name", "clock")),
            None,
        ),
        (
            TokenEventKind.USAGE,
            None,
            (("input_tokens", "11"), ("output_tokens", "7")),
            None,
        ),
        (TokenEventKind.FINISH, None, (("reason", "tool_calls"),), None),
    )


def _simple_events(reason: str, stop_sequence: str | None = None) -> tuple[TokenEvent, ...]:
    finish = (("reason", reason),)
    if stop_sequence is not None:
        finish = (("reason", reason), ("stop_sequence", stop_sequence))
    return _sequence(
        (TokenEventKind.TEXT_DELTA, "ok", (), None),
        (
            TokenEventKind.USAGE,
            None,
            (("input_tokens", "3"), ("output_tokens", "1")),
            None,
        ),
        (TokenEventKind.FINISH, None, finish, None),
    )


class ProjectionTests(unittest.TestCase):
    def test_reasoning_usage_is_identical_in_final_stream_and_non_stream(self) -> None:
        for count in (0, 3):
            events = list(_full_events())
            events[-2] = replace(
                events[-2],
                payload_items=(*events[-2].payload_items, ("reasoning_tokens", str(count))),
            )
            response = project_openai_chat_completion(events, model="fixture", created=0)
            chunks = project_openai_chat_completion_stream(
                events, model="fixture", created=0, include_usage=True
            )
            self.assertEqual(chunks[-1]["usage"], response["usage"])
            self.assertEqual(
                response["usage"]["completion_tokens_details"], {"reasoning_tokens": count}
            )

    def test_invalid_reasoning_usage_fails_projection(self) -> None:
        for count in ("-1", "8", "1.5", "true"):
            events = list(_full_events())
            events[-2] = replace(
                events[-2], payload_items=(*events[-2].payload_items, ("reasoning_tokens", count))
            )
            with self.subTest(count=count), self.assertRaises(ProjectionError):
                project_openai_chat_completion(events, model="fixture", created=0)

    def test_tool_heartbeats_do_not_split_argument_blocks_and_length_keeps_raw_prefix(self) -> None:
        first = (
            TokenEventKind.TOOL_CALL_DELTA,
            '{"city":"Tokyo",',
            (("id", "call-0"), ("index", "0"), ("name", "weather")),
            None,
        )
        events = _sequence(
            first,
            (TokenEventKind.TEXT_DELTA, "", (), None),
            (TokenEventKind.TOOL_CALL_DELTA, '"count":', (("index", "0"),), None),
            (TokenEventKind.USAGE, None, (("input_tokens", "1"), ("output_tokens", "9")), None),
            (TokenEventKind.FINISH, None, (("reason", "length"),), None),
        )
        response = project_openai_chat_completion(events, model="fixture", created=7)
        self.assertEqual(response["choices"][0]["finish_reason"], "length")
        call = response["choices"][0]["message"]["tool_calls"][0]
        self.assertEqual(call["function"]["arguments"], '{"city":"Tokyo","count":')
        response = project_anthropic_message(events, model="fixture")
        self.assertEqual(response["stop_reason"], "max_tokens")
        self.assertEqual(response["content"][0]["input"], {"city": "Tokyo"})

    def test_openai_non_stream_preserves_every_canonical_channel(self) -> None:
        response = project_openai_chat_completion(_full_events(), model="fixture", created=123)

        self.assertEqual(response["id"], "request-1")
        self.assertEqual(response["object"], "chat.completion")
        self.assertEqual(
            response["usage"],
            {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
        )
        choice = response["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertEqual(
            choice["message"],
            {
                "role": "assistant",
                "content": "answer",
                "reasoning_content": "think now",
                "tool_calls": [
                    {
                        "id": "call-0",
                        "type": "function",
                        "function": {
                            "name": "weather",
                            "arguments": '{"city":"Paris"}',
                        },
                    },
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "clock", "arguments": "{}"},
                    },
                ],
            },
        )

    def test_openai_stream_folds_to_the_same_non_stream_values(self) -> None:
        events = _full_events()
        response = project_openai_chat_completion(events, model="fixture", created=123)
        chunks = project_openai_chat_completion_stream(
            events, model="fixture", created=123, include_usage=True
        )

        self.assertEqual(
            chunks[0]["choices"][0]["delta"], {"role": "assistant", "content": ""}
        )
        self.assertEqual(chunks[-2]["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(chunks[-1]["choices"], [])
        self.assertEqual(chunks[-1]["usage"], response["usage"])

        deltas = [chunk["choices"][0]["delta"] for chunk in chunks[1:-2]]
        self.assertEqual(
            "".join(delta.get("content", "") for delta in deltas),
            response["choices"][0]["message"]["content"],
        )
        self.assertEqual(
            "".join(delta.get("reasoning_content", "") for delta in deltas),
            response["choices"][0]["message"]["reasoning_content"],
        )
        tool_deltas = [delta["tool_calls"][0] for delta in deltas if "tool_calls" in delta]
        self.assertEqual(
            "".join(delta["function"]["arguments"] for delta in tool_deltas[:2]),
            response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],
        )

        without_usage = project_openai_chat_completion_stream(
            events, model="fixture", created=123, include_usage=False
        )
        self.assertEqual(without_usage, chunks[:-1])
        self.assertEqual(without_usage[-1]["choices"][0]["finish_reason"], "tool_calls")

        with self.assertRaises(ProjectionError):
            project_openai_chat_completion_stream(
                events, model="fixture", created=123, include_usage=1  # type: ignore[arg-type]
            )

    def test_anthropic_non_stream_preserves_ordered_blocks(self) -> None:
        response = project_anthropic_message(_full_events(), model="fixture")

        self.assertEqual(response["stop_reason"], "tool_use")
        self.assertEqual(response["usage"], {"input_tokens": 11, "output_tokens": 7})
        self.assertEqual(
            response["content"],
            [
                {"type": "thinking", "thinking": "think now"},
                {"type": "text", "text": "answer"},
                {
                    "type": "tool_use",
                    "id": "call-0",
                    "name": "weather",
                    "input": {"city": "Paris"},
                },
                {
                    "type": "tool_use",
                    "id": "call-1",
                    "name": "clock",
                    "input": {},
                },
            ],
        )

    def test_anthropic_stream_has_balanced_typed_blocks_and_terminal_usage(self) -> None:
        events = _full_events()
        response = project_anthropic_message(events, model="fixture")
        chunks = project_anthropic_message_stream(events, model="fixture")

        self.assertEqual(chunks[0]["type"], "message_start")
        self.assertEqual(chunks[-2]["type"], "message_delta")
        self.assertEqual(chunks[-2]["usage"], {"output_tokens": 7})
        self.assertEqual(chunks[-1], {"type": "message_stop"})
        starts = [chunk for chunk in chunks if chunk["type"] == "content_block_start"]
        stops = [chunk for chunk in chunks if chunk["type"] == "content_block_stop"]
        self.assertEqual([chunk["index"] for chunk in starts], [0, 1, 2, 3])
        self.assertEqual([chunk["index"] for chunk in stops], [0, 1, 2, 3])
        self.assertEqual(
            [chunk["content_block"]["type"] for chunk in starts],
            [block["type"] for block in response["content"]],
        )
        deltas = [chunk["delta"] for chunk in chunks if chunk["type"] == "content_block_delta"]
        self.assertEqual(
            [delta["type"] for delta in deltas],
            [
                "thinking_delta",
                "thinking_delta",
                "text_delta",
                "text_delta",
                "input_json_delta",
                "input_json_delta",
                "input_json_delta",
            ],
        )

    def test_finish_reasons_map_explicitly_for_both_protocols(self) -> None:
        cases = (
            ("stop", None, "stop", "end_turn", None),
            ("stop", "END", "stop", "stop_sequence", "END"),
            ("length", None, "length", "max_tokens", None),
        )
        for reason, sequence, openai_reason, anthropic_reason, anthropic_sequence in cases:
            with self.subTest(reason=reason, sequence=sequence):
                events = _simple_events(reason, sequence)
                openai = project_openai_chat_completion(events, model="fixture", created=0)
                anthropic = project_anthropic_message(events, model="fixture")
                self.assertEqual(openai["choices"][0]["finish_reason"], openai_reason)
                self.assertEqual(anthropic["stop_reason"], anthropic_reason)
                self.assertEqual(anthropic["stop_sequence"], anthropic_sequence)
                if sequence is not None:
                    self.assertEqual(openai["choices"][0]["stop_reason"], sequence)

    def test_invalid_lifecycle_and_event_fields_fail_closed(self) -> None:
        valid = _simple_events("stop")
        duplicate_usage = _sequence(
            (TokenEventKind.TEXT_DELTA, "ok", (), None),
            (TokenEventKind.USAGE, None, (("input_tokens", "1"), ("output_tokens", "1")), None),
            (TokenEventKind.USAGE, None, (("input_tokens", "1"), ("output_tokens", "1")), None),
            (TokenEventKind.FINISH, None, (("reason", "stop"),), None),
        )
        raw_token = _sequence(
            (TokenEventKind.TOKEN, None, (), 7),
            (TokenEventKind.USAGE, None, (("input_tokens", "1"), ("output_tokens", "1")), None),
            (TokenEventKind.FINISH, None, (("reason", "stop"),), None),
        )
        bad_payload = _sequence(
            (TokenEventKind.TEXT_DELTA, "ok", (("unknown", "x"),), None),
            (TokenEventKind.USAGE, None, (("input_tokens", "01"), ("output_tokens", "1")), None),
            (TokenEventKind.FINISH, None, (("reason", "stop"),), None),
        )
        cases = (
            (),
            valid[:-1],
            (valid[-1], valid[-2]),
            duplicate_usage,
            raw_token,
            bad_payload,
            (replace(valid[0], sequence_no=2), *valid[1:]),
            (valid[0], replace(valid[1], request_id="other"), valid[2]),
        )
        for events in cases:
            with self.subTest(events=events), self.assertRaises(ProjectionError):
                project_openai_chat_completion(events, model="fixture", created=0)
        with self.assertRaises(ProjectionError):
            project_openai_chat_completion(valid, model=" ", created=0)
        with self.assertRaises(ProjectionError):
            project_openai_chat_completion(valid, model="fixture", created=True)

    def test_invalid_tool_delta_combinations_fail_closed(self) -> None:
        usage = (
            TokenEventKind.USAGE,
            None,
            (("input_tokens", "1"), ("output_tokens", "1")),
            None,
        )
        tool_finish = (TokenEventKind.FINISH, None, (("reason", "tool_calls"),), None)
        first = (
            TokenEventKind.TOOL_CALL_DELTA,
            "{}",
            (("id", "call-0"), ("index", "0"), ("name", "tool")),
            None,
        )
        missing_metadata = _sequence(
            (TokenEventKind.TOOL_CALL_DELTA, "{}", (("index", "0"),), None),
            usage,
            tool_finish,
        )
        invalid_json = _sequence(
            (first[0], "{", first[2], None),
            usage,
            tool_finish,
        )
        reopened = _sequence(
            first,
            (TokenEventKind.TEXT_DELTA, "middle", (), None),
            (TokenEventKind.TOOL_CALL_DELTA, "", (("index", "0"),), None),
            usage,
            tool_finish,
        )
        finish_mismatch = _sequence(
            first,
            usage,
            (TokenEventKind.FINISH, None, (("reason", "stop"),), None),
        )
        for events in (missing_metadata, invalid_json, reopened, finish_mismatch):
            with self.subTest(events=events), self.assertRaises(ProjectionError):
                project_anthropic_message(events, model="fixture")


if __name__ == "__main__":
    unittest.main()
