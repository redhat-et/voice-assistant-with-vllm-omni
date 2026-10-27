# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
#
# Vendored from vllm_omni.clients.duplex (vllm-omni v0.30.0), trimmed to the
# session/transport core (dropped the EventCollector/WAV/metrics CLI-test
# tooling this agent doesn't use). `pip install vllm-omni` would pull in its
# whole GPU-serving dependency tree (diffusers, accelerate, torch via
# transformers, ...) just for this client, which only needs stdlib + base64 +
# websockets -- not appropriate for a lightweight voice agent container.
# Update this file by re-vendoring from vllm_omni/clients/duplex.py if the
# wire protocol changes upstream.
"""Public asynchronous client for the vLLM-Omni full-duplex Realtime API.

:class:`DuplexClientBase` holds the transport-agnostic session client (typed
events, response demultiplexing, incremental playback acking, the session
handshake). :class:`DuplexClient` connects it to ``/v1/realtime?duplex=1``
(the normative duplex contract) over WebSocket with transparent session
resume on transport drops; :class:`vllm_omni.clients.inline_duplex.InlineDuplexClient`
drives an in-process :class:`~vllm_omni.entrypoints.duplex_omni.DuplexOmni`
with exactly the same usage.

Example::

    from vllm_omni.clients.duplex import DuplexClient
    from vllm_omni.clients.minicpmo_4_5 import create_duplex_session_config

    cfg = create_duplex_session_config(ref_audio=audio_data_url(wav_bytes))
    async with DuplexClient("ws://localhost:8099", model=model, config=cfg) as client:
        await client.stream_pcm(pcm16)
        await client.commit()
        async for response in client.responses():
            if response.decision == "listen":
                continue
            async for chunk in response.audio():
                play(chunk)
                await client.ack_playback(response.played_ms)
            break

Media capture/playback, VAD, and resampling stay in the application; this
module moves bytes and events.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Sequence
from dataclasses import dataclass, field
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

import base64

__all__ = [
    "DUPLEX_FIRST_UNIT_MS",
    "DUPLEX_UNIT_MS",
    "PCM16_BYTES_PER_SAMPLE",
    "PCM16_SAMPLE_RATE",
    "AudioFormat",
    "AudioDelta",
    "ConnectionResumed",
    "DuplexClient",
    "DuplexClientBase",
    "DuplexClientError",
    "DuplexConnectionError",
    "DuplexEvent",
    "DuplexProtocolError",
    "DuplexSessionClosedError",
    "ErrorEvent",
    "ListenDecision",
    "ReconnectPolicy",
    "ResponseCreated",
    "ResponseDone",
    "ResponseHandle",
    "SessionConfig",
    "SessionCreated",
    "SessionResumed",
    "SessionUpdated",
    "SpeakDecision",
    "WebSocketTransport",
    "TextDelta",
    "TranscriptDelta",
    "duplex_unit_boundary_ms",
    "wrap_event",
]

# The default duplex input wire format (16 kHz mono PCM16), kept as plain
# constants for callers doing byte/duration arithmetic outside AudioFormat.
PCM16_SAMPLE_RATE = 16_000
PCM16_BYTES_PER_SAMPLE = 2

# The native PCM reservation closes every 1000 ms, including the first unit.
# Stage0 adds the first mel window's padding internally; the camera frame must
# arrive with the reservation, before that internal padding is applied.
DUPLEX_FIRST_UNIT_MS = 1000
DUPLEX_UNIT_MS = 1000


def duplex_unit_boundary_ms(unit_index: int) -> int:
    """Cumulative appended audio, in ms, that closes model unit ``unit_index``."""
    return DUPLEX_FIRST_UNIT_MS + max(0, int(unit_index)) * DUPLEX_UNIT_MS


# Internal transport/backpressure bounds (promote to constructor parameters
# only if a real consumer needs to tune them).
_MAX_BUFFERED_EVENTS = 1024
_MAX_FRAME_BYTES = 64 << 20


def _bare_base64(data: str) -> str:
    """Strip a ``data:<mime>;base64,`` prefix; the wire carries bare base64."""
    if data.startswith("data:"):
        _, _, payload = data.partition(";base64,")
        if payload:
            return payload
    return data


def _put_drop_oldest(queue: asyncio.Queue, item: object) -> None:
    """Deliver ``item`` without blocking, dropping the oldest entry when full.

    Every internal feed path uses this: a consumer that stops draining its
    queue loses its oldest buffered events instead of parking the single
    reader task, which must keep dispatching (and acking) for the rest of
    the session.
    """
    try:
        queue.put_nowait(item)
    except asyncio.QueueFull:
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:  # pragma: no cover - racing consumer
            pass
        queue.put_nowait(item)


# ---------------------------------------------------------------------------
# Errors


class DuplexClientError(Exception):
    """Base class for all duplex client errors."""


class DuplexConnectionError(DuplexClientError):
    """The WebSocket transport could not be established or failed."""


class DuplexProtocolError(DuplexClientError):
    """The server rejected a request with an ``error`` event."""

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        event_id: str | None = None,
        raw: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.event_id = event_id
        self.raw = dict(raw or {})


class DuplexSessionClosedError(DuplexClientError):
    """The session ended and the client cannot continue."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"duplex session closed: {reason}")
        self.reason = reason


# ---------------------------------------------------------------------------
# Configuration

_BYTES_PER_SAMPLE = {"pcm16": 2, "pcm_f32le": 4}


@dataclass(frozen=True)
class AudioFormat:
    """One PCM wire format: encoding name plus sample rate."""

    encoding: str
    sample_rate_hz: int

    @property
    def bytes_per_sample(self) -> int:
        try:
            return _BYTES_PER_SAMPLE[self.encoding]
        except KeyError:
            raise ValueError(f"Unknown audio encoding: {self.encoding!r}") from None

    def duration_ms(self, byte_count: int) -> float:
        return byte_count * 1000.0 / (self.sample_rate_hz * self.bytes_per_sample)

    def byte_count(self, duration_ms: float) -> int:
        samples = round(self.sample_rate_hz * duration_ms / 1000.0)
        return int(samples) * self.bytes_per_sample


@dataclass
class SessionConfig:
    """Model-agnostic duplex session configuration.

    Model-specific knobs ride in ``extra_body``; each duplex model ships a
    ``create_duplex_session_config`` preset in its client module (e.g.
    ``vllm_omni.clients.minicpmo_4_5``), so the client itself stays
    model-neutral.
    """

    modalities: tuple[str, ...] = ("audio", "text")
    input_audio: AudioFormat = AudioFormat("pcm16", 16_000)
    output_audio: AudioFormat = AudioFormat("pcm16", 24_000)
    instructions: str | None = None
    ref_audio: str | None = None
    voice: str | None = None
    auto_response: bool = True
    temperature: float | None = None
    overlap_policy: str | None = None
    playback_commit_policy: str | None = None
    turn_detection: dict[str, object] | None = None
    idle_timeout_s: float | None = None
    extra_body: dict[str, object] = field(default_factory=dict)

    def to_session_payload(self, *, model: str) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": model,
            "modalities": list(self.modalities),
            "input_audio_format": self.input_audio.encoding,
            "output_audio_format": self.output_audio.encoding,
            # The encodings alone do not pin the rates (presets differ from the
            # server defaults), so declare both explicitly: the server reads
            # the input rate from the top-level field and the output rate from
            # the audio.output block.
            "sample_rate_hz": self.input_audio.sample_rate_hz,
            "audio": {
                "input": {"sample_rate_hz": self.input_audio.sample_rate_hz},
                "output": {"sample_rate_hz": self.output_audio.sample_rate_hz},
            },
            "turn_detection": self.turn_detection,
        }
        if self.instructions is not None:
            payload["instructions"] = self.instructions
        if self.ref_audio is not None:
            payload["ref_audio"] = self.ref_audio
        if self.voice is not None:
            payload["voice"] = self.voice
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.overlap_policy is not None:
            payload["overlap_policy"] = self.overlap_policy
        if self.playback_commit_policy is not None:
            payload["playback_commit_policy"] = self.playback_commit_policy
        if self.idle_timeout_s is not None:
            payload["idle_timeout_s"] = self.idle_timeout_s
        extra_body: dict[str, object] = {"auto_response": self.auto_response}
        extra_body.update(self.extra_body)
        payload["extra_body"] = extra_body
        return payload


@dataclass(frozen=True)
class ReconnectPolicy:
    """Jittered exponential backoff for ``session.resume`` reconnects."""

    max_attempts: int = 5
    backoff_s: tuple[float, float] = (0.25, 4.0)

    def delay(self, attempt: int) -> float:
        low, high = self.backoff_s
        ceiling = min(high, low * (2.0 ** max(0, attempt)))
        return random.uniform(low, max(low, ceiling))


# ---------------------------------------------------------------------------
# Typed events


class DuplexEvent:
    """Thin typed wrapper over one server event dict."""

    __slots__ = ("raw",)

    def __init__(self, raw: dict[str, object]) -> None:
        self.raw = raw

    @property
    def type(self) -> str:
        value = self.raw.get("type")
        return value if isinstance(value, str) else ""

    @property
    def event_id(self) -> str | None:
        value = self.raw.get("event_id")
        return value if isinstance(value, str) else None

    @property
    def session_id(self) -> str | None:
        value = self.raw.get("session_id")
        return value if isinstance(value, str) else None

    @property
    def response_id(self) -> str | None:
        value = self.raw.get("response_id")
        if isinstance(value, str):
            return value
        response = self.raw.get("response")
        if isinstance(response, dict):
            value = response.get("id")
            if isinstance(value, str):
                return value
        return None

    @property
    def item_id(self) -> str | None:
        value = self.raw.get("item_id")
        return value if isinstance(value, str) else None

    @property
    def server_event_seq(self) -> int | None:
        value = self.raw.get("server_event_seq")
        return value if isinstance(value, int) else None

    @property
    def sample_rate_hz(self) -> int | None:
        value = self.raw.get("sample_rate_hz")
        return value if isinstance(value, int) and value > 0 else None

    @property
    def audio(self) -> bytes | None:
        delta = self.raw.get("delta") or self.raw.get("audio")
        if not isinstance(delta, str) or not delta:
            return None
        try:
            return base64.b64decode(delta)
        except ValueError:
            return None

    @property
    def text(self) -> str | None:
        for key in ("text", "delta", "transcript"):
            value = self.raw.get(key)
            if isinstance(value, str):
                return value
        return None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}(type={self.type!r})"


class SessionCreated(DuplexEvent):
    @property
    def session(self) -> dict[str, object]:
        value = self.raw.get("session")
        return value if isinstance(value, dict) else {}

    @property
    def resume_token(self) -> str | None:
        value = self.raw.get("resume_token")
        return value if isinstance(value, str) else None


class SessionResumed(SessionCreated):
    pass


class SessionUpdated(DuplexEvent):
    """An acknowledged replacement of the public session configuration."""

    @property
    def session(self) -> dict[str, object]:
        value = self.raw.get("session")
        return value if isinstance(value, dict) else {}


class SessionClosed(DuplexEvent):
    pass


class SessionExpired(DuplexEvent):
    pass


class ResponseCreated(DuplexEvent):
    pass


class ResponseDone(DuplexEvent):
    pass


class AudioDelta(DuplexEvent):
    pass


class TranscriptDelta(DuplexEvent):
    pass


class TextDelta(DuplexEvent):
    pass


class ListenDecision(DuplexEvent):
    pass


class SpeakDecision(DuplexEvent):
    pass


class ConnectionResumed(DuplexEvent):
    """Synthetic client event: the transport dropped and was resumed."""


class ErrorEvent(DuplexEvent):
    @property
    def _error(self) -> dict[str, object] | None:
        value = self.raw.get("error")
        return value if isinstance(value, dict) else None

    @property
    def code(self) -> str | None:
        error = self._error
        value = (error or self.raw).get("code")
        return value if isinstance(value, str) else None

    @property
    def message(self) -> str:
        error = self._error
        if error is not None:
            value = error.get("message")
            if isinstance(value, str):
                return value
        value = self.raw.get("error")
        return value if isinstance(value, str) else json.dumps(self.raw)

    @property
    def related_event_id(self) -> str | None:
        error = self._error
        value = error.get("event_id") if error is not None else None
        return value if isinstance(value, str) else None

    def to_exception(self) -> DuplexProtocolError:
        return DuplexProtocolError(
            self.message,
            code=self.code,
            event_id=self.related_event_id,
            raw=self.raw,
        )


_EVENT_TYPES: dict[str, type[DuplexEvent]] = {
    "session.created": SessionCreated,
    "session.resumed": SessionResumed,
    "session.updated": SessionUpdated,
    "session.closed": SessionClosed,
    "session.expired": SessionExpired,
    "response.created": ResponseCreated,
    "response.done": ResponseDone,
    "response.output_audio.delta": AudioDelta,
    "response.output_audio_transcript.delta": TranscriptDelta,
    "response.text.delta": TextDelta,
    "response.output_text.delta": TextDelta,
    "response.listen": ListenDecision,
    "response.speak": SpeakDecision,
    "connection.resumed": ConnectionResumed,
    "error": ErrorEvent,
}

_AUDIO_DELTA_TYPES = frozenset(name for name, event_cls in _EVENT_TYPES.items() if event_cls is AudioDelta)

_FATAL_RESUME_CODES = frozenset(
    {
        "invalid_session_resume",
        "session_resume_expired",
        "unsupported_session_resume",
        "invalid_resume_token",
        "session_resume_conflict",
    }
)


def wrap_event(raw: dict[str, object]) -> DuplexEvent:
    event_type = raw.get("type")
    event_cls = _EVENT_TYPES.get(event_type) if isinstance(event_type, str) else None
    return (event_cls or DuplexEvent)(raw)


@dataclass(frozen=True)
class _ClosedMarker:
    reason: str
    expected: bool


@dataclass(frozen=True)
class _ErrorMarker:
    error: ErrorEvent


# ---------------------------------------------------------------------------
# Response demultiplexing


class ResponseHandle:
    """One ``response.created`` → terminal lifecycle, demultiplexed.

    A handle reaches :meth:`DuplexClient.responses` only once its decision is
    known: when the response starts producing output (``decision`` is
    ``"speak"`` or about to become it) or when a terminal listen finishes it
    (``decision == "listen"``, no audio). Standalone listen decisions surface
    as already-finished handles too, so a turn loop stays uniform.
    """

    def __init__(
        self,
        *,
        response_id: str | None,
        output_format: AudioFormat,
        max_buffered_events: int,
        decision: str | None = None,
        created_event: ResponseCreated | None = None,
    ) -> None:
        self.response_id = response_id
        self.decision = decision
        self.transcript = ""
        self.text = ""
        self.played_ms = 0.0
        self.created_event = created_event
        self.done_event: DuplexEvent | None = None
        self._output_format = output_format
        self._audio_queue: asyncio.Queue[tuple[bytes, int | None] | None] = asyncio.Queue(maxsize=max_buffered_events)
        self._done = asyncio.Event()
        self._announced = False

    @property
    def finished(self) -> bool:
        return self._done.is_set()

    async def wait(self, timeout_s: float | None = None) -> None:
        if timeout_s is None:
            await self._done.wait()
        else:
            await asyncio.wait_for(self._done.wait(), timeout_s)

    async def audio(self) -> AsyncIterator[bytes]:
        """Yield decoded output-audio chunks until the response finishes.

        ``played_ms`` advances as chunks are yielded, so it reflects what the
        application has taken for playback — feed it to
        :meth:`DuplexClient.ack_playback`. A handle whose audio is never
        drained drops its oldest buffered chunks once the buffer fills; it
        never stalls the client's reader.
        """
        while True:
            item = await self._audio_queue.get()
            if item is None:
                return
            chunk, sample_rate_hz = item
            rate = sample_rate_hz or self._output_format.sample_rate_hz
            self.played_ms += len(chunk) * 1000.0 / (rate * self._output_format.bytes_per_sample)
            yield chunk

    def _feed(self, event: DuplexEvent) -> None:
        if self._done.is_set():
            return
        if isinstance(event, AudioDelta):
            chunk = event.audio
            if chunk:
                _put_drop_oldest(self._audio_queue, (chunk, event.sample_rate_hz))
        elif isinstance(event, TranscriptDelta):
            self.transcript += event.text or ""
        elif isinstance(event, TextDelta):
            self.text += event.text or ""
        elif isinstance(event, SpeakDecision):
            self.decision = "speak"
        elif isinstance(event, ListenDecision):
            # A terminal listen on a response that already spoke is the
            # model yielding the turn; it must not downgrade the decision.
            if self.decision is None:
                self.decision = "listen"

    def _finish(self, event: DuplexEvent | None) -> None:
        if self._done.is_set():
            return
        self.done_event = event
        self._done.set()
        # A full queue cannot block _finish: the sentinel is delivered by a
        # non-blocking put with a drop-oldest fallback.
        _put_drop_oldest(self._audio_queue, None)


# ---------------------------------------------------------------------------
# Client


class WebSocketTransport(ABC):
    """Minimal WebSocket surface the client needs (satisfied by ``websockets``)."""

    @abstractmethod
    async def send(self, data: str) -> None: ...

    @abstractmethod
    async def recv(self) -> str | bytes: ...

    @abstractmethod
    async def close(self) -> None: ...


ConnectFn = Callable[[str], Awaitable["WebSocketTransport"]]


class DuplexClientBase(ABC):
    """Transport-agnostic duplex session client.

    Everything a caller touches (typed events, response demux, input helpers,
    playback acks, the ``session.update`` handshake) lives here. A subclass
    only supplies the transport: :meth:`_open` establishes it and must deliver
    the server's ``session.created`` payload through :meth:`_dispatch`,
    :meth:`_send_command` delivers one client event, :meth:`_teardown`
    releases the transport.
    """

    def __init__(
        self,
        *,
        model: str,
        config: SessionConfig | None = None,
        handshake_timeout_s: float = 30.0,
    ) -> None:
        self.model = model
        self.config = config or SessionConfig()
        #: Allocated by the server; ``None`` until ``session.created`` arrives.
        self.session_id: str | None = None
        self.session_info: dict[str, object] = {}
        self._handshake_timeout_s = handshake_timeout_s
        self._subscribers: list[asyncio.Queue[DuplexEvent | _ClosedMarker]] = []
        self._response_queue: asyncio.Queue[ResponseHandle | _ClosedMarker | _ErrorMarker] = asyncio.Queue(
            maxsize=_MAX_BUFFERED_EVENTS
        )
        self._responses: dict[str, ResponseHandle] = {}
        self._input_audio_end_ms = 0.0
        self._closing = False
        self._closed = asyncio.Event()
        self._closed_marker: _ClosedMarker | None = None
        self._last_error_reason: str | None = None

    # -- transport hooks ------------------------------------------------------

    @abstractmethod
    async def _open(self) -> None:
        """Establish the transport and start delivering server events via ``_dispatch``.

        Called once from :meth:`__aenter__`; the base then sends the
        ``session.update`` handshake and waits for ``session.created``.
        """

    @abstractmethod
    async def _send_command(self, payload: dict[str, object]) -> None:
        """Deliver one client event (already stamped with ``event_id``)."""

    @abstractmethod
    async def _teardown(self) -> None:
        """Release the transport; must be idempotent."""

    # -- lifecycle ----------------------------------------------------------

    async def __aenter__(self) -> DuplexClientBase:
        handshake_queue = self._add_subscriber()
        try:
            await self._open()
            await self.send(
                {
                    "type": "session.update",
                    "session": self.config.to_session_payload(model=self.model),
                }
            )
            created = await self._wait_on_queue(
                handshake_queue,
                ("session.created",),
                timeout_s=self._handshake_timeout_s,
            )
        except BaseException:
            await self._teardown_quietly()
            raise
        finally:
            self._remove_subscriber(handshake_queue)
        assert isinstance(created, SessionCreated)
        self._adopt_session(created)
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        try:
            if exc_type is None and not self._closed.is_set():
                await self.close()
            elif not self._closed.is_set():
                # An application error (or cancellation) is propagating. Still
                # tell the server the session is over: dropping the transport
                # silently reads as a resumable disconnect and holds the
                # session slot for the full disconnect grace period. Best
                # effort only — the teardown below closes the transport
                # whether or not the close reaches the server.
                self._closing = True
                try:
                    await self.send({"type": "session.close"})
                except DuplexClientError:
                    pass
        finally:
            await self._teardown_quietly()

    async def close(self, *, timeout_s: float = 20.0) -> None:
        """Send ``session.close`` and wait for the server to confirm."""
        if self._closed.is_set():
            return
        self._closing = True
        try:
            await self.send({"type": "session.close"})
        except (DuplexConnectionError, DuplexSessionClosedError):
            self._finalize("closed", expected=True)
            return
        try:
            await asyncio.wait_for(self._closed.wait(), timeout_s)
        except asyncio.TimeoutError:
            self._finalize("close timed out", expected=True)

    async def _teardown_quietly(self) -> None:
        self._closing = True
        if not self._closed.is_set():
            self._finalize("closed", expected=True)
        await self._teardown()

    # -- input ----------------------------------------------------------------

    async def append_audio(
        self,
        pcm: bytes,
        *,
        is_speech: bool | None = None,
        video_frames: Sequence[str] | None = None,
    ) -> None:
        """Append one chunk of input audio in the session's input format.

        A turn is ended with :meth:`commit`, never by a flag on the append.

        ``video_frames`` takes base64 JPEG/PNG strings (one per ~1 s unit for
        omni models). Data URLs (:func:`image_data_url`) are accepted too;
        their prefix is stripped, since the wire contract carries bare base64.
        """
        input_format = self.config.input_audio
        duration_ms = input_format.duration_ms(len(pcm))
        self._input_audio_end_ms += duration_ms
        event: dict[str, object] = {
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(pcm).decode("ascii"),
            "format": input_format.encoding,
            "sample_rate_hz": input_format.sample_rate_hz,
            "duration_ms": round(duration_ms),
            "audio_end_ms": round(self._input_audio_end_ms),
        }
        if is_speech is not None:
            event["is_speech"] = is_speech
        if video_frames:
            event["video_frames"] = [_bare_base64(frame) for frame in video_frames]
        await self.send(event)

    async def stream_pcm(
        self,
        pcm: bytes,
        *,
        chunk_ms: int = 200,
        realtime: bool = True,
        is_speech: bool | None = None,
        video_frames: Sequence[str] | None = None,
        stacked_video_frames: Sequence[str | None] | None = None,
    ) -> int:
        """Slice ``pcm`` into ``chunk_ms`` chunks, pace, append; interleave frames.

        ``video_frames`` holds base64/data-URL JPEG frames in capture order,
        one per second of the clip. Frame ``k`` rides the append that closes
        model unit ``k`` (see :func:`duplex_unit_boundary_ms`), which
        reproduces the official ``streaming_prefill(audio_waveform=<1 s>,
        frame_list=[frame])`` pairing: a second of audio and the picture
        captured during it enter the same unit. Sending a frame before its
        whole-second unit boundary would strand it on an append that cannot
        close the unit yet.

        ``stacked_video_frames`` is the optional parallel track of composites
        (see ``vllm_omni.experimental.fullduplex.video_stacking``): entry ``k``
        tiles the sub-frames captured *inside* unit ``k`` and rides the same
        append right after the base frame. ``None`` entries send the base
        frame alone.

        Returns the number of base frames actually sent (a clip shorter than
        the frame list leaves the tail unsent).
        """
        input_format = self.config.input_audio
        chunk_bytes = max(input_format.byte_count(chunk_ms), input_format.bytes_per_sample)
        frames = list(video_frames or [])
        stacked = list(stacked_video_frames or [])

        def units() -> Iterator[tuple[bytes, list[str] | None]]:
            audio_end_ms = 0.0
            frames_sent = 0
            for offset in range(0, len(pcm), chunk_bytes):
                chunk = pcm[offset : offset + chunk_bytes]
                audio_end_ms += input_format.duration_ms(len(chunk))
                if not frames or audio_end_ms < duplex_unit_boundary_ms(frames_sent):
                    yield chunk, None
                    continue
                # A video advances one frame per unit and holds its last frame
                # when the audio outlives the clip; a still image is a
                # one-element list and therefore repeats.
                index = min(frames_sent, len(frames) - 1)
                composite = stacked[index] if index < len(stacked) else None
                frames_sent += 1
                yield chunk, [frames[index]] if composite is None else [frames[index], composite]

        frames_sent = 0
        for chunk, unit_frames in units():
            if not chunk:
                continue
            await self.append_audio(chunk, is_speech=is_speech, video_frames=unit_frames)
            if unit_frames:
                frames_sent += 1
            if realtime:
                await asyncio.sleep(input_format.duration_ms(len(chunk)) / 1000.0)
        return frames_sent

    async def commit(self, *, final: bool = True, create_response: bool | None = None) -> None:
        event: dict[str, object] = {"type": "input_audio_buffer.commit", "final": final}
        if create_response is not None:
            event["response_create"] = create_response
        await self.send(event)

    # -- interruption ---------------------------------------------------------

    async def cancel_response(self, response_id: str | None = None) -> None:
        """Cancel the active (or a specific) response.

        Together with :meth:`clear_input` this composes a client-forced
        barge-in for sessions whose capabilities advertise
        ``supports_barge_in``; the currently bundled duplex models do not,
        so their overlap handling is model-owned.
        """
        event: dict[str, object] = {"type": "response.cancel"}
        if response_id is not None:
            event["response_id"] = response_id
        await self.send(event)

    async def clear_input(self) -> None:
        await self.send({"type": "input_audio_buffer.clear"})

    # -- playback contract ------------------------------------------------------

    async def ack_playback(
        self,
        played_ms: float,
        *,
        response_id: str | None = None,
        item_id: str | None = None,
        committed_ms: float | None = None,
    ) -> None:
        """Report cumulative playback progress; call periodically while playing."""
        event: dict[str, object] = {
            "type": "playback.ack",
            "played_ms": int(played_ms),
            "committed_ms": int(committed_ms if committed_ms is not None else played_ms),
        }
        if response_id is not None:
            event["response_id"] = response_id
        if item_id is not None:
            event["item_id"] = item_id
        await self.send(event)

    # -- output ----------------------------------------------------------------

    def __aiter__(self) -> AsyncIterator[DuplexEvent]:
        return self.events()

    async def events(self) -> AsyncIterator[DuplexEvent]:
        """Iterate over every server event from now on (typed)."""
        queue = self._add_subscriber()
        try:
            while True:
                item = await queue.get()
                if isinstance(item, _ClosedMarker):
                    if item.expected:
                        return
                    raise DuplexSessionClosedError(item.reason)
                yield item
        finally:
            self._remove_subscriber(queue)

    async def responses(self) -> AsyncIterator[ResponseHandle]:
        """Iterate over response lifecycles (single consumer).

        Raises :class:`DuplexProtocolError` when the server rejects a request
        with an ``error`` event while waiting — a rejected send (an invalid
        video frame, an empty commit) produces no response, so without this
        the wait would never end. Re-enter the iterator to keep consuming
        after handling the error.
        """
        while True:
            item = await self._response_queue.get()
            if isinstance(item, _ClosedMarker):
                if item.expected:
                    return
                raise DuplexSessionClosedError(item.reason)
            if isinstance(item, _ErrorMarker):
                raise item.error.to_exception()
            yield item

    async def wait_for(self, *types: str, timeout_s: float) -> DuplexEvent:
        """Wait for the next event whose type is in ``types``.

        Raises :class:`DuplexProtocolError` if an ``error`` event arrives
        first, and :class:`DuplexSessionClosedError` if the session ends first.
        """
        queue = self._add_subscriber()
        try:
            return await self._wait_on_queue(queue, types, timeout_s=timeout_s)
        finally:
            self._remove_subscriber(queue)

    async def send(self, event: dict[str, object]) -> str:
        """Send one raw client event; returns the (possibly stamped) event_id."""
        if self._closed.is_set():
            marker = self._closed_marker
            raise DuplexSessionClosedError(marker.reason if marker else "closed")
        payload = dict(event)
        payload.setdefault("event_id", f"evt_{uuid4().hex}")
        try:
            await self._send_command(payload)
        except asyncio.CancelledError:
            raise
        except DuplexClientError:
            raise
        except Exception as exc:
            raise DuplexConnectionError(f"send failed: {exc}") from exc
        return payload["event_id"]

    # -- internals ---------------------------------------------------------------

    def _adopt_session(self, event: SessionCreated) -> None:
        # session.created and session.resumed both carry the engine's current
        # session object; keep the last one seen.
        if event.session:
            self.session_info = event.session
        session_id = event.session.get("session_id") or event.session.get("id") or event.session_id
        if isinstance(session_id, str) and session_id:
            self.session_id = session_id

    def _announce(self, handle: ResponseHandle) -> None:
        """Deliver a handle to responses() exactly once, decision known."""
        if not handle._announced:
            handle._announced = True
            _put_drop_oldest(self._response_queue, handle)

    def _add_subscriber(self) -> asyncio.Queue[DuplexEvent | _ClosedMarker]:
        queue: asyncio.Queue[DuplexEvent | _ClosedMarker] = asyncio.Queue(maxsize=_MAX_BUFFERED_EVENTS)
        if self._closed.is_set() and self._closed_marker is not None:
            queue.put_nowait(self._closed_marker)
        else:
            self._subscribers.append(queue)
        return queue

    def _remove_subscriber(self, queue: asyncio.Queue[DuplexEvent | _ClosedMarker]) -> None:
        try:
            self._subscribers.remove(queue)
        except ValueError:
            pass

    async def _wait_on_queue(
        self,
        queue: asyncio.Queue[DuplexEvent | _ClosedMarker],
        types: tuple[str, ...],
        *,
        timeout_s: float,
    ) -> DuplexEvent:
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise asyncio.TimeoutError(f"timed out waiting for {types}")
            item = await asyncio.wait_for(queue.get(), remaining)
            if isinstance(item, _ClosedMarker):
                raise DuplexSessionClosedError(item.reason)
            if isinstance(item, ErrorEvent) and item.type not in types:
                raise item.to_exception()
            if item.type in types:
                return item

    def _accept_event(self, data: dict[str, object]) -> bool:
        """Transport hook: return False to drop ``data`` before dispatch (duplicates)."""
        del data
        return True

    async def _after_dispatch(self, event: DuplexEvent, data: dict[str, object]) -> None:
        """Transport hook run after an event was delivered to subscribers."""
        del event, data

    async def _dispatch(self, data: dict[str, object]) -> None:
        if not self._accept_event(data):
            return
        event = wrap_event(data)

        if isinstance(event, SessionResumed):
            self._adopt_session(event)
        elif isinstance(event, SessionCreated):
            self._adopt_session(event)
        elif isinstance(event, SessionUpdated):
            if event.session:
                self.session_info = event.session
        elif isinstance(event, ResponseCreated):
            response_id = event.response_id
            if response_id and response_id not in self._responses:
                handle = ResponseHandle(
                    response_id=response_id,
                    output_format=self.config.output_audio,
                    max_buffered_events=_MAX_BUFFERED_EVENTS,
                    created_event=event,
                )
                self._responses[response_id] = handle
                # Not yielded yet: the handle reaches responses() once its
                # decision is known — at the first output below, or at its
                # terminal event — so a turn loop can branch on `decision`
                # the moment it receives the handle.
        elif isinstance(event, (AudioDelta, TranscriptDelta, TextDelta, SpeakDecision)):
            handle = self._responses.get(event.response_id or "")
            if handle is not None:
                handle._feed(event)
                self._announce(handle)
        elif isinstance(event, ListenDecision):
            # Only a listen carrying a response_id is that response's
            # terminal decision (the server stamps it exactly there); an
            # id-less listen is a standalone decision beat and must never
            # close an in-flight response.
            handle = self._responses.pop(event.response_id or "", None)
            if handle is not None:
                handle._feed(event)
                handle._finish(event)
                self._announce(handle)
            else:
                listen_handle = ResponseHandle(
                    response_id=event.response_id,
                    output_format=self.config.output_audio,
                    max_buffered_events=_MAX_BUFFERED_EVENTS,
                    decision="listen",
                )
                listen_handle._announced = True
                listen_handle._finish(event)
                _put_drop_oldest(self._response_queue, listen_handle)
        elif isinstance(event, ResponseDone):
            handle = self._responses.pop(event.response_id or "", None)
            if handle is not None:
                if handle.decision is None:
                    handle.decision = "speak"
                handle._finish(event)
                self._announce(handle)
        elif isinstance(event, ErrorEvent):
            # A rejected request produces no response; surface the rejection
            # on the response path too so a consumer waiting in responses()
            # is not left waiting forever (see the responses() docstring).
            _put_drop_oldest(self._response_queue, _ErrorMarker(event))
            # Keep the text for the close that usually follows: a client that
            # is streaming input is inside send(), not draining responses(),
            # and would otherwise only ever see "closed".
            self._last_error_reason = f"{event.code}: {event.message}" if event.code else event.message

        for queue in list(self._subscribers):
            _put_drop_oldest(queue, event)

        if isinstance(event, SessionClosed):
            self._finalize(self._last_error_reason or "closed", expected=True)
        elif isinstance(event, SessionExpired):
            reason = data.get("reason")
            self._finalize(f"expired: {reason}" if reason else "expired", expected=False)

        await self._after_dispatch(event, data)

    def _finalize(self, reason: str, *, expected: bool) -> None:
        if self._closed.is_set():
            return
        marker = _ClosedMarker(reason=reason, expected=expected)
        self._closed_marker = marker
        self._closed.set()
        for handle in self._responses.values():
            handle._finish(None)
        self._responses.clear()
        for queue in (*self._subscribers, self._response_queue):
            _put_drop_oldest(queue, marker)


class DuplexClient(DuplexClientBase):
    """Async client for one duplex session over ``/v1/realtime?duplex=1``."""

    def __init__(
        self,
        url: str,
        *,
        model: str,
        config: SessionConfig | None = None,
        reconnect: ReconnectPolicy | None = ReconnectPolicy(),
        heartbeat_interval_s: float | None = 30.0,
        handshake_timeout_s: float = 30.0,
        connect: ConnectFn | None = None,
    ) -> None:
        super().__init__(model=model, config=config, handshake_timeout_s=handshake_timeout_s)
        self.url = url
        self.resume_token: str | None = None
        self._reconnect = reconnect
        self._heartbeat_interval_s = heartbeat_interval_s
        self._connect_fn: ConnectFn = connect or self._default_connect
        self._ws: WebSocketTransport | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._last_server_event_seq: int | None = None

    # -- transport hooks ------------------------------------------------------

    async def _open(self) -> None:
        self._ws = await self._connect_fn(self._target_url())
        self._reader_task = asyncio.create_task(self._read_loop(), name="duplex-client-reader")

    async def __aenter__(self) -> DuplexClient:
        await super().__aenter__()
        if self._heartbeat_interval_s is not None:
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop(), name="duplex-client-heartbeat")
        return self

    async def _send_command(self, payload: dict[str, object]) -> None:
        if self._ws is None:
            raise DuplexConnectionError("send failed: transport is not connected")
        await self._ws.send(json.dumps(payload))

    async def _teardown(self) -> None:
        for task in (self._heartbeat_task, self._reader_task):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:  # noqa: BLE001
                pass

    # -- internals ---------------------------------------------------------------

    def _target_url(self) -> str:
        parts = urlsplit(self.url)
        path = parts.path if parts.path not in ("", "/") else "/v1/realtime"
        query = dict(parse_qsl(parts.query, keep_blank_values=True))
        query.setdefault("duplex", "1")
        query.setdefault("model", self.model)
        # The client always opens with an explicit session.update (and resumes
        # with an explicit session.resume). Autostart would race that handshake:
        # with a model query param, the server creates a bare default session
        # before reading the first client event, silently dropping ref_audio
        # and the model-specific extra_body. Force it off even when the caller's
        # URL carries autostart=1 — this client cannot operate over it.
        query["autostart"] = "0"
        return urlunsplit((parts.scheme or "ws", parts.netloc, path, urlencode(query), parts.fragment))

    async def _default_connect(self, url: str) -> WebSocketTransport:
        try:
            import websockets
        except ImportError as exc:  # pragma: no cover - websockets is a pinned dep
            raise DuplexConnectionError("The duplex client requires the 'websockets' package") from exc
        try:
            return await websockets.connect(url, max_size=_MAX_FRAME_BYTES)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise DuplexConnectionError(f"connect to {url} failed: {exc}") from exc

    def _adopt_session(self, event: SessionCreated) -> None:
        super()._adopt_session(event)
        if event.resume_token:
            self.resume_token = event.resume_token

    def _accept_event(self, data: dict[str, object]) -> bool:
        seq = data.get("server_event_seq")
        if isinstance(seq, int):
            if self._last_server_event_seq is not None and seq <= self._last_server_event_seq:
                return False  # duplicate delivered by resume replay
            self._last_server_event_seq = seq
        return True

    async def _after_dispatch(self, event: DuplexEvent, data: dict[str, object]) -> None:
        if event.type == "session.resync_required":
            # The server stopped journaling this session's events, so any
            # retained resume credential will be refused on the next
            # session.resume. Drop it now: a later transport failure then
            # finalizes immediately instead of burning reconnect attempts.
            self.resume_token = None

        seq = data.get("server_event_seq")
        if isinstance(seq, int) and not self._closed.is_set() and self._ws is not None:
            try:
                await self._ws.send(json.dumps({"type": "session.event_ack", "server_event_seq": seq}))
            except asyncio.CancelledError:
                raise
            except Exception:
                pass  # the reader will observe the transport failure itself

    async def _read_loop(self) -> None:
        while True:
            try:
                while True:
                    raw = await self._ws.recv()
                    if isinstance(raw, (bytes, bytearray)):
                        continue
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(data, dict):
                        await self._dispatch(data)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if self._closing or self._closed.is_set():
                    self._finalize("closed", expected=True)
                    return
                if await self._attempt_resume():
                    continue
                self._finalize(f"connection lost: {exc}", expected=False)
                return

    async def _attempt_resume(self) -> bool:
        policy = self._reconnect
        if policy is None or not self.resume_token or not self.session_id:
            return False
        for attempt in range(policy.max_attempts):
            await asyncio.sleep(policy.delay(attempt))
            try:
                ws = await self._connect_fn(self._target_url())
            except asyncio.CancelledError:
                raise
            except Exception:
                continue
            try:
                await ws.send(
                    json.dumps(
                        {
                            "type": "session.resume",
                            "session_id": self.session_id,
                            "resume_token": self.resume_token,
                            "last_received_server_event_seq": self._last_server_event_seq or 0,
                        }
                    )
                )
                deadline = time.monotonic() + self._handshake_timeout_s
                pending: list[dict[str, object]] = []
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise asyncio.TimeoutError
                    raw = await asyncio.wait_for(ws.recv(), remaining)
                    if isinstance(raw, (bytes, bytearray)):
                        continue
                    data = json.loads(raw)
                    if not isinstance(data, dict):
                        continue
                    event_type = data.get("type")
                    if event_type == "session.resumed":
                        previous = self._ws
                        self._ws = ws
                        if previous is not None and previous is not ws:
                            await self._close_quietly(previous)
                        await self._dispatch({"type": "connection.resumed", "attempt": attempt})
                        await self._dispatch(data)
                        for entry in pending:
                            await self._dispatch(entry)
                        return True
                    if event_type == "session.resync_required":
                        await self._close_quietly(ws)
                        return False
                    if event_type == "error":
                        code = ErrorEvent(data).code
                        if code in _FATAL_RESUME_CODES:
                            await self._close_quietly(ws)
                            return False
                        await self._close_quietly(ws)
                        break  # transient error: next attempt
                    pending.append(data)
            except asyncio.CancelledError:
                # ws is not yet self._ws, so _teardown will not close it.
                await self._close_quietly(ws)
                raise
            except Exception:
                await self._close_quietly(ws)
                continue
        return False

    @staticmethod
    async def _close_quietly(ws: WebSocketTransport) -> None:
        try:
            await ws.close()
        except Exception:  # noqa: BLE001
            pass

    async def _heartbeat_loop(self) -> None:
        interval = self._heartbeat_interval_s
        assert interval is not None
        while not self._closed.is_set():
            await asyncio.sleep(interval)
            try:
                await self.send({"type": "session.heartbeat"})
            except DuplexConnectionError:
                # Transient transport failure — a tick can land inside the
                # resume window while the old socket is already dead. Keep
                # ticking; the resumed session still needs heartbeats to
                # reset the server's receive timeout. If the session is
                # really over, _finalize sets _closed and the loop exits.
                continue
            except DuplexClientError:
                return

