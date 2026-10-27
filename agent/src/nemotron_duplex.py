from __future__ import annotations

import array
import asyncio
import logging
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlparse

from livekit import rtc
from livekit.agents.llm.chat_context import ChatContext, FunctionCallOutput
from livekit.agents.llm import FunctionCall
from livekit.agents.llm.realtime import (
    GenerationCreatedEvent,
    MessageGeneration,
    RealtimeCapabilities,
    RealtimeModel,
    RealtimeSession,
)
from livekit.agents.llm.tool_context import Tool, ToolChoice, ToolContext
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils.aio import Chan
from livekit.agents.utils.audio import AudioByteStream

from _vllm_omni_duplex_client import DuplexClient, DuplexClientError
from _vllm_omni_nemotron_config import create_duplex_session_config
from vllm_realtime import _AudioStream, _MessageStream, _TextStream

logger = logging.getLogger("nemotron-duplex")

_INPUT_SAMPLE_RATE_HZ = 16_000
_INPUT_FRAME_SAMPLES = 1_280  # 80ms at 16kHz -- the duplex engine's frame-locked unit
_OUTPUT_SAMPLE_RATE_HZ = 22_050


def _pcm16_to_f32le(pcm16: bytes) -> bytes:
    samples = array.array("h")
    samples.frombytes(pcm16)
    floats = array.array("f", (s / 32768.0 for s in samples))
    return floats.tobytes()


@dataclass
class _Generation:
    response_id: str
    audio_stream: _AudioStream
    text_stream: _TextStream
    function_stream: Chan[FunctionCall]


class NemotronDuplexRealtimeModel(RealtimeModel):
    """RealtimeModel for Nemotron VoiceChat's model-native full-duplex mode.

    Unlike OpenAI's Realtime API (and Qwen3-Omni's /v1/realtime, see
    vllm_realtime.py), Nemotron VoiceChat always listens and speaks
    concurrently and decides on its own when to respond -- there is no "turn"
    to detect or interrupt, and no barge-in. The duplex engine also speaks a
    frame-locked wire protocol (16kHz float32 input in exact 80ms/1280-sample
    chunks, 22.05kHz PCM16 output) that LiveKit's generic OpenAI-compatible
    realtime client can't produce -- see _vllm_omni_duplex_client.py, vendored
    from vllm-omni's own tested client for this exact protocol.
    """

    def __init__(self, *, base_url: str, model: str, api_key: str = "EMPTY") -> None:
        super().__init__(
            capabilities=RealtimeCapabilities(
                message_truncation=False,
                turn_detection=True,
                user_transcription=False,
                auto_tool_reply_generation=True,
                audio_output=True,
                manual_function_calls=False,
                supports_overlapping_speech=True,
            )
        )
        # VLLM_BASE_URL is conventionally http(s)://host:port/v1 in this app
        # (it also serves plain /v1/chat/completions). The vendored duplex
        # client needs an explicit ws(s):// scheme (unlike the OpenAI client,
        # it does not convert http(s)) and only defaults an *empty* path to
        # /v1/realtime -- a /v1 path is left alone, not treated as "needs
        # /realtime appended" the way the OpenAI client's URL builder does.
        if base_url.startswith("https://"):
            base_url = "wss://" + base_url[len("https://") :]
        elif base_url.startswith("http://"):
            base_url = "ws://" + base_url[len("http://") :]
        if base_url.rstrip("/").endswith("/v1"):
            base_url = base_url.rstrip("/") + "/realtime"
        self._base_url = base_url
        self._model_name = model
        self._api_key = api_key
        self._sessions: set[NemotronDuplexRealtimeSession] = set()

    @property
    def model(self) -> str:
        return self._model_name

    @property
    def provider(self) -> str:
        return urlparse(self._base_url).hostname or "vllm-omni"

    def session(self, *, turn_detection_disabled: bool = False) -> NemotronDuplexRealtimeSession:
        if turn_detection_disabled:
            logger.warning(
                "turn_detection_disabled has no effect: Nemotron VoiceChat full-duplex "
                "has no client-driven turn mode to fall back to"
            )
        sess = NemotronDuplexRealtimeSession(self)
        self._sessions.add(sess)
        return sess

    async def aclose(self) -> None:
        for sess in list(self._sessions):
            await sess.aclose()
        self._sessions.clear()


class NemotronDuplexRealtimeSession(RealtimeSession):
    def __init__(self, realtime_model: NemotronDuplexRealtimeModel) -> None:
        super().__init__(realtime_model)
        self._model = realtime_model
        self._instructions = ""
        self._chat_ctx = ChatContext()
        self._tool_ctx = ToolContext([])
        self._seen_tool_output_ids: set[str] = set()
        self._seen_function_call_ids: set[str] = set()

        self._client: DuplexClient | None = None
        self._run_task: asyncio.Task | None = None
        self._sender_task: asyncio.Task | None = None
        self._input_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._resampler: rtc.AudioResampler | None = None
        self._resampler_input_rate: int | None = None
        self._bytestream = AudioByteStream(
            _INPUT_SAMPLE_RATE_HZ, 1, samples_per_channel=_INPUT_FRAME_SAMPLES
        )
        self._generations: dict[str, _Generation] = {}
        self._pending_generate_reply: list[asyncio.Future[GenerationCreatedEvent]] = []
        self._closed = False

    @property
    def chat_ctx(self) -> ChatContext:
        return self._chat_ctx

    @property
    def tools(self) -> ToolContext:
        return self._tool_ctx

    async def update_instructions(self, instructions: str) -> None:
        self._instructions = instructions

    async def update_tools(self, tools: list[Tool]) -> None:
        self._tool_ctx = ToolContext(tools)

    async def update_chat_ctx(self, chat_ctx: ChatContext) -> None:
        new_tool_results = [
            item
            for item in chat_ctx.items
            if isinstance(item, FunctionCallOutput) and item.call_id not in self._seen_tool_output_ids
        ]
        self._chat_ctx = chat_ctx
        if not new_tool_results:
            return
        self._ensure_started()
        for item in new_tool_results:
            self._seen_tool_output_ids.add(item.call_id)
            assert self._client is not None
            try:
                await self._client.send(
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": item.output or "",
                        },
                    }
                )
            except DuplexClientError:
                logger.exception("Failed to send tool output to Nemotron VoiceChat")

    def update_options(self, *, tool_choice: NotGivenOr[ToolChoice | None] = NOT_GIVEN) -> None:
        pass

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        if self._closed:
            return
        self._ensure_started()
        if self._resampler is None or self._resampler_input_rate != frame.sample_rate:
            self._resampler = rtc.AudioResampler(frame.sample_rate, _INPUT_SAMPLE_RATE_HZ, num_channels=1)
            self._resampler_input_rate = frame.sample_rate
        for resampled in self._resampler.push(frame):
            for chunk in self._bytestream.push(bytes(resampled.data)):
                self._input_queue.put_nowait(_pcm16_to_f32le(bytes(chunk.data)))

    def push_video(self, frame: rtc.VideoFrame) -> None:
        pass

    def generate_reply(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        tools: NotGivenOr[list[Tool]] = NOT_GIVEN,
    ) -> asyncio.Future[GenerationCreatedEvent]:
        # Nemotron VoiceChat decides on its own when to speak (e.g. an opening
        # greeting from `instructions`); there is no "generate now" command to
        # send. Resolved by the next response.created the session dispatches.
        self._ensure_started()
        fut: asyncio.Future[GenerationCreatedEvent] = asyncio.get_event_loop().create_future()
        self._pending_generate_reply.append(fut)
        return fut

    def commit_audio(self) -> None:
        pass

    def clear_audio(self) -> None:
        while not self._input_queue.empty():
            try:
                self._input_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._bytestream = AudioByteStream(
            _INPUT_SAMPLE_RATE_HZ, 1, samples_per_channel=_INPUT_FRAME_SAMPLES
        )

    def interrupt(self) -> None:
        # The native duplex deployment does not support barge-in (see
        # vllm-omni's NemotronLabs-VoiceChat recipe); the model owns pacing.
        pass

    def truncate(
        self,
        *,
        message_id: str,
        modalities: list[Literal["text", "audio"]],
        audio_end_ms: int,
        audio_transcript: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        pass

    def _ensure_started(self) -> None:
        if self._run_task is None and not self._closed:
            self._run_task = asyncio.create_task(self._run(), name="nemotron-duplex-session")

    async def _run(self) -> None:
        config = create_duplex_session_config(
            instructions=self._instructions or None,
            tools=self._tool_ctx.parse_function_tools("openai") or None,
        )
        client = DuplexClient(self._model._base_url, model=self._model._model_name, config=config)
        self._client = client
        try:
            async with client:
                self._sender_task = asyncio.create_task(self._send_loop(), name="nemotron-duplex-sender")
                await self._consume_events()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Nemotron VoiceChat duplex session failed")
        finally:
            if self._sender_task is not None:
                self._sender_task.cancel()
            for response_id in list(self._generations):
                self._finish_generation(response_id)
            for fut in self._pending_generate_reply:
                if not fut.done():
                    fut.set_exception(RuntimeError("Nemotron VoiceChat duplex session ended"))
            self._pending_generate_reply.clear()

    async def _send_loop(self) -> None:
        assert self._client is not None
        while True:
            chunk = await self._input_queue.get()
            if chunk is None:
                return
            try:
                await self._client.append_audio(chunk)
            except DuplexClientError:
                return

    async def _consume_events(self) -> None:
        assert self._client is not None
        async for event in self._client.events():
            raw = event.raw
            event_type = raw.get("type")
            if event_type == "response.created":
                self._start_generation(event.response_id)
            elif event_type == "response.output_audio.delta":
                self._on_audio_delta(event.response_id, event.audio, event.sample_rate_hz)
            elif event_type == "response.output_audio_transcript.delta":
                self._on_text_delta(event.response_id, event.text)
            elif event_type == "response.output_item.done":
                item = raw.get("item")
                if isinstance(item, dict) and item.get("type") == "function_call":
                    self._on_function_call(event.response_id, item)
            elif event_type == "response.done":
                self._finish_generation(event.response_id)
            elif event_type == "error":
                logger.error("Nemotron VoiceChat duplex error: %s", raw.get("error") or raw)

    def _start_generation(self, response_id: str | None) -> None:
        if not response_id or response_id in self._generations:
            return
        audio_stream = _AudioStream()
        text_stream = _TextStream()
        function_stream: Chan[FunctionCall] = Chan()
        self._generations[response_id] = _Generation(
            response_id=response_id,
            audio_stream=audio_stream,
            text_stream=text_stream,
            function_stream=function_stream,
        )
        modalities_fut: asyncio.Future[list[Literal["text", "audio"]]] = asyncio.get_event_loop().create_future()
        modalities_fut.set_result(["audio", "text"])
        generation = MessageGeneration(
            message_id=f"nemotron-{response_id}",
            text_stream=text_stream,
            audio_stream=audio_stream,
            modalities=modalities_fut,
        )
        event = GenerationCreatedEvent(
            message_stream=_MessageStream(generation),
            function_stream=function_stream,
            user_initiated=False,
        )
        if self._pending_generate_reply:
            fut = self._pending_generate_reply.pop(0)
            if not fut.done():
                fut.set_result(event)
        self.emit("generation_created", event)

    def _on_audio_delta(self, response_id: str | None, chunk: bytes | None, sample_rate_hz: int | None) -> None:
        gen = self._generations.get(response_id or "")
        if gen is None or not chunk:
            return
        rate = sample_rate_hz or _OUTPUT_SAMPLE_RATE_HZ
        gen.audio_stream.push(
            rtc.AudioFrame(data=chunk, sample_rate=rate, num_channels=1, samples_per_channel=len(chunk) // 2)
        )
        # shortcut: no playback.ack. Nemotron VoiceChat doesn't support
        # barge-in (the only thing the ack's truncation contract is for), and
        # acking requires the server to have already registered the response
        # as a history item -- which, for this model's turns, only itself
        # happens inside a *successful* ack. The first ack for a turn almost
        # always loses that race (confirmed live: playback_item_not_found on
        # every turn), so there is no working ack sequence to send here.

    def _on_text_delta(self, response_id: str | None, text: str | None) -> None:
        gen = self._generations.get(response_id or "")
        if gen is None or not text:
            return
        gen.text_stream.push(text)

    def _on_function_call(self, response_id: str | None, item: dict[str, object]) -> None:
        gen = self._generations.get(response_id or "")
        if gen is None:
            return
        call_id = item.get("call_id")
        name = item.get("name")
        arguments = item.get("arguments")
        if not isinstance(call_id, str) or not isinstance(name, str):
            return
        # The server has sent the same completed function_call item twice for
        # one call_id in testing; LiveKit executes a tool per FunctionCall it
        # receives, so forwarding a duplicate re-runs the tool.
        if call_id in self._seen_function_call_ids:
            return
        self._seen_function_call_ids.add(call_id)
        logger.info("Tool call: %s(%s) [%s]", name, str(arguments)[:100], call_id)
        gen.function_stream.send_nowait(
            FunctionCall(call_id=call_id, name=name, arguments=arguments if isinstance(arguments, str) else "{}")
        )

    def _finish_generation(self, response_id: str | None) -> None:
        gen = self._generations.pop(response_id or "", None)
        if gen is None:
            return
        gen.text_stream.close()
        gen.audio_stream.close()
        gen.function_stream.close()

    async def aclose(self) -> None:
        self._closed = True
        self._input_queue.put_nowait(None)
        if self._run_task is not None:
            self._run_task.cancel()
            try:
                await self._run_task
            except (asyncio.CancelledError, Exception):
                pass
        self._model._sessions.discard(self)
