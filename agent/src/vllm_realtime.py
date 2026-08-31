from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import time
import wave
from collections.abc import AsyncIterator
from typing import Literal
from urllib.parse import urlparse

from livekit import rtc
from livekit.agents.utils.aio import Chan
from livekit.agents.llm.chat_context import ChatContext, FunctionCallOutput
from livekit.agents.llm.realtime import (
    GenerationCreatedEvent,
    MessageGeneration,
    RealtimeCapabilities,
    RealtimeModel,
    RealtimeSession,
)
from livekit.agents.llm import FunctionCall
from livekit.agents.llm.tool_context import Tool, ToolChoice, ToolContext
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from openai import AsyncOpenAI
from openai.lib.streaming.chat import ChatCompletionStreamState

logger = logging.getLogger("vllm-realtime")


class _AudioStream:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[rtc.AudioFrame | None] = asyncio.Queue()

    def push(self, frame: rtc.AudioFrame) -> None:
        self._queue.put_nowait(frame)

    def close(self) -> None:
        self._queue.put_nowait(None)

    def __aiter__(self) -> AsyncIterator[rtc.AudioFrame]:
        return self

    async def __anext__(self) -> rtc.AudioFrame:
        frame = await self._queue.get()
        if frame is None:
            raise StopAsyncIteration
        return frame


class _TextStream:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()

    def push(self, text: str) -> None:
        self._queue.put_nowait(text)

    def close(self) -> None:
        self._queue.put_nowait(None)

    def __aiter__(self) -> AsyncIterator[str]:
        return self

    async def __anext__(self) -> str:
        text = await self._queue.get()
        if text is None:
            raise StopAsyncIteration
        return text


class _MessageStream:
    def __init__(self, generation: MessageGeneration) -> None:
        self._generation = generation
        self._sent = False

    def __aiter__(self) -> AsyncIterator[MessageGeneration]:
        return self

    async def __anext__(self) -> MessageGeneration:
        if self._sent:
            raise StopAsyncIteration
        self._sent = True
        return self._generation


def _frames_to_wav_base64(frames: list[rtc.AudioFrame]) -> str:
    if not frames:
        return ""
    pcm = b"".join(bytes(f.data) for f in frames)
    sample_rate = frames[0].sample_rate
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return base64.b64encode(buf.getvalue()).decode()


def _wav_bytes_to_frame(wav_data: bytes) -> rtc.AudioFrame | None:
    buf = io.BytesIO(wav_data)
    try:
        with wave.open(buf, "rb") as w:
            sr = w.getframerate()
            ch = w.getnchannels()
            pcm = w.readframes(w.getnframes())
    except Exception:
        return None
    samples_per_channel = len(pcm) // (2 * ch)
    if samples_per_channel == 0:
        return None
    return rtc.AudioFrame(
        data=pcm, sample_rate=sr, num_channels=ch,
        samples_per_channel=samples_per_channel,
    )


class VLLMRealtimeModel(RealtimeModel):
    def __init__(
        self, *, base_url: str, model: str, api_key: str = "EMPTY",
        speaker: str = "Ethan",
        room: rtc.Room | None = None,
    ) -> None:
        super().__init__(
            capabilities=RealtimeCapabilities(
                message_truncation=False,
                turn_detection=False,
                user_transcription=False,
                auto_tool_reply_generation=True,
                audio_output=True,
                manual_function_calls=False,
            )
        )
        self._base_url = base_url
        self._model_name = model
        self._api_key = api_key
        self._speaker = speaker
        self._room = room
        self._sessions: set[VLLMRealtimeSession] = set()

    @property
    def model(self) -> str:
        return self._model_name

    @property
    def provider(self) -> str:
        return urlparse(self._base_url).hostname or "vllm-omni"

    def session(self, *, turn_detection_disabled: bool = False) -> VLLMRealtimeSession:
        # This model never does server-side turn detection (turn_detection=False
        # above), so there's nothing to disable regardless of the flag's value.
        sess = VLLMRealtimeSession(self)
        self._sessions.add(sess)
        return sess

    async def aclose(self) -> None:
        for sess in list(self._sessions):
            await sess.aclose()
        self._sessions.clear()


class VLLMRealtimeSession(RealtimeSession):
    def __init__(self, realtime_model: VLLMRealtimeModel) -> None:
        super().__init__(realtime_model)
        self._model = realtime_model
        self._client = AsyncOpenAI(
            api_key=realtime_model._api_key, base_url=realtime_model._base_url,
        )
        self._closed = False
        self._chat_ctx = ChatContext()
        self._tool_ctx = ToolContext([])
        self._instructions = ""
        self._conversation: list[dict] = []
        self._audio_buffer: list[rtc.AudioFrame] = []
        self._current_audio_stream: _AudioStream | None = None
        self._current_text_stream: _TextStream | None = None
        self._current_function_stream: Chan[FunctionCall] | None = None
        self._generation_task: asyncio.Task | None = None
        self._interrupted = False
        self._seen_tool_output_ids: set[str] = set()

    @property
    def chat_ctx(self) -> ChatContext:
        return self._chat_ctx

    @property
    def tools(self) -> ToolContext:
        return self._tool_ctx

    async def update_instructions(self, instructions: str) -> None:
        self._instructions = instructions

    async def update_chat_ctx(self, chat_ctx: ChatContext) -> None:
        new_tool_results = [
            item for item in chat_ctx.items
            if isinstance(item, FunctionCallOutput) and item.call_id not in self._seen_tool_output_ids
        ]
        self._chat_ctx = chat_ctx

        if not new_tool_results:
            return

        logger.info("Tool results received, generating follow-up")
        for item in new_tool_results:
            self._seen_tool_output_ids.add(item.call_id)

        tool_messages = []
        call_ids = {r.call_id for r in new_tool_results}
        for item in chat_ctx.items:
            if isinstance(item, FunctionCall) and item.call_id in call_ids:
                tool_messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": item.call_id,
                        "type": "function",
                        "function": {"name": item.name, "arguments": item.arguments},
                    }],
                })
            elif isinstance(item, FunctionCallOutput) and item.call_id in call_ids:
                tool_messages.append({
                    "role": "tool",
                    "tool_call_id": item.call_id,
                    "content": item.output or "",
                })

        if tool_messages:
            self._conversation.extend(tool_messages)
            messages: list[dict] = []
            if self._instructions:
                messages.append({"role": "system", "content": self._instructions})
            messages.extend(self._conversation)
            self._start_generation_task(self._streaming_generation(
                messages, tool_choice="auto",
            ))

    async def update_tools(self, tools: list[Tool]) -> None:
        self._tool_ctx = ToolContext(tools)

    def update_options(self, *, tool_choice: NotGivenOr[ToolChoice | None] = NOT_GIVEN) -> None:
        pass

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        if self._closed:
            return
        self._audio_buffer.append(frame)

    def push_video(self, frame: rtc.VideoFrame) -> None:
        pass

    def generate_reply(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        tools: NotGivenOr[list[Tool]] = NOT_GIVEN,
    ) -> asyncio.Future[GenerationCreatedEvent]:
        fut: asyncio.Future[GenerationCreatedEvent] = asyncio.get_event_loop().create_future()

        frames = list(self._audio_buffer)
        self._audio_buffer.clear()

        tc = tool_choice if isinstance(tool_choice, str) else "auto"
        self._start_generation_task(
            self._run_generation(frames, fut, tool_choice=tc)
        )
        return fut

    def _start_generation_task(self, coro) -> None:
        if self._generation_task and not self._generation_task.done():
            self._generation_task.cancel()
        self._finish_generation()
        self._generation_task = asyncio.create_task(coro)

    def commit_audio(self) -> None:
        pass

    def clear_audio(self) -> None:
        self._audio_buffer.clear()

    def interrupt(self) -> None:
        logger.info("Interrupt requested")
        self._interrupted = True
        if self._generation_task and not self._generation_task.done():
            self._generation_task.cancel()
        self._finish_generation()

    def truncate(
        self, *, message_id: str, modalities: list[Literal["text", "audio"]],
        audio_end_ms: int, audio_transcript: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        pass

    def _create_generation_event(self) -> tuple[GenerationCreatedEvent, Chan[FunctionCall]]:
        self._current_audio_stream = _AudioStream()
        self._current_text_stream = _TextStream()
        self._current_function_stream = Chan()

        modalities_fut: asyncio.Future[list[Literal["text", "audio"]]] = (
            asyncio.get_event_loop().create_future()
        )
        modalities_fut.set_result(["audio", "text"])

        generation = MessageGeneration(
            message_id=f"vllm-msg-{int(time.time())}",
            text_stream=self._current_text_stream,
            audio_stream=self._current_audio_stream,
            modalities=modalities_fut,
        )

        event = GenerationCreatedEvent(
            message_stream=_MessageStream(generation),
            function_stream=self._current_function_stream,
            user_initiated=False,
        )
        return event, self._current_function_stream

    async def _streaming_generation(
        self,
        messages: list[dict],
        *,
        tool_choice: str = "auto",
        publish_tool_call_latency: bool = False,
        generation_start: float | None = None,
    ) -> None:
        event, function_stream = self._create_generation_event()
        self.emit("generation_created", event)

        if generation_start is None:
            generation_start = time.perf_counter()

        tools_param = None
        if tool_choice != "none":
            tools_param = self._tool_ctx.parse_function_tools("openai") or None

        assistant_text = ""
        first_audio = True
        tool_call_detected = False
        try:
            kwargs: dict = {
                "model": self._model._model_name,
                "modalities": ["text", "audio"],
                "messages": messages,
                "stream": True,
                "extra_body": {"speaker": self._model._speaker},
            }
            if tools_param:
                kwargs["tools"] = tools_param
                kwargs["tool_choice"] = tool_choice

            state = ChatCompletionStreamState(input_tools=tools_param or [])
            stream = await self._client.chat.completions.create(**kwargs)

            async for chunk in stream:
                state.handle_chunk(chunk)

                modality = getattr(chunk, "modality", None)
                for choice in chunk.choices:
                    delta = getattr(choice, "delta", None)
                    if not delta:
                        continue

                    if getattr(delta, "tool_calls", None) and not tool_call_detected:
                        tool_call_detected = True

                    content = getattr(delta, "content", None)
                    if not content:
                        continue

                    if modality == "audio" and tool_call_detected:
                        continue
                    elif modality == "audio":
                        if first_audio:
                            first_audio = False
                            ttfa = time.perf_counter() - generation_start
                            logger.info("Time to first audio: %.3fs", ttfa)
                            interrupted = self._interrupted
                            self._interrupted = False
                            try:
                                if self._model._room:
                                    await self._model._room.local_participant.publish_data(
                                        json.dumps({"ttfa": ttfa, "interrupted": interrupted}).encode(),
                                        topic="latency",
                                    )
                            except Exception:
                                logger.warning("Failed to publish latency")
                        frame = _wav_bytes_to_frame(base64.b64decode(content))
                        if frame and self._current_audio_stream:
                            self._current_audio_stream.push(frame)
                    elif modality == "text" or modality is None:
                        assistant_text += content
                        if self._current_text_stream:
                            self._current_text_stream.push(content)

            completion = state.get_final_completion()
            tool_calls = completion.choices[0].message.tool_calls if completion.choices else None

            if tool_calls:
                for tc in tool_calls:
                    logger.info("Tool call: %s(%s) [%s]", tc.function.name, tc.function.arguments[:100], tc.id)
                    if self._current_function_stream:
                        self._current_function_stream.send_nowait(
                            FunctionCall(
                                call_id=tc.id,
                                name=tc.function.name,
                                arguments=tc.function.arguments,
                            )
                        )
                try:
                    if self._model._room:
                        ttfa = time.perf_counter() - generation_start
                        await self._model._room.local_participant.publish_data(
                            json.dumps({"ttfa": ttfa, "tool_call": True}).encode(),
                            topic="latency",
                        )
                        for tc in tool_calls:
                            await self._model._room.local_participant.publish_data(
                                json.dumps({"name": tc.function.name, "arguments": tc.function.arguments}).encode(),
                                topic="tool_call",
                            )
                except Exception:
                    logger.warning("Failed to publish tool call telemetry")

        except asyncio.CancelledError:
            logger.info("Generation cancelled")
        except Exception:
            logger.exception("Chat completion error")
        finally:
            if assistant_text:
                self._conversation.append({"role": "assistant", "content": assistant_text})
            self._finish_generation()

    async def _run_generation(
        self,
        frames: list[rtc.AudioFrame],
        fut: asyncio.Future[GenerationCreatedEvent],
        tool_choice: str = "auto",
    ) -> None:
        wav_b64 = _frames_to_wav_base64(frames)
        if not wav_b64:
            logger.warning("No audio to process")
            if not fut.done():
                fut.set_exception(RuntimeError("No audio to process"))
            return

        logger.info("Generating reply from %d audio frames", len(frames))

        # PoC: conversation history grows unbounded (including base64 audio).
        # Production code should trim older turns or replace audio with transcripts.
        messages: list[dict] = []
        if self._instructions:
            messages.append({"role": "system", "content": self._instructions})
        messages.extend(self._conversation)
        user_message = {
            "role": "user",
            "content": [
                {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{wav_b64}"}},
            ],
        }
        messages.append(user_message)

        event, function_stream = self._create_generation_event()

        if not fut.done():
            fut.set_result(event)
        self.emit("generation_created", event)

        tools_param = None
        if tool_choice != "none":
            tools_param = self._tool_ctx.parse_function_tools("openai") or None

        assistant_text = ""
        generation_start = time.perf_counter()
        first_audio = True
        tool_call_detected = False
        try:
            kwargs: dict = {
                "model": self._model._model_name,
                "modalities": ["text", "audio"],
                "messages": messages,
                "stream": True,
                "extra_body": {"speaker": self._model._speaker},
            }
            if tools_param:
                kwargs["tools"] = tools_param
                kwargs["tool_choice"] = tool_choice

            state = ChatCompletionStreamState(input_tools=tools_param or [])
            stream = await self._client.chat.completions.create(**kwargs)

            async for chunk in stream:
                state.handle_chunk(chunk)

                modality = getattr(chunk, "modality", None)
                for choice in chunk.choices:
                    delta = getattr(choice, "delta", None)
                    if not delta:
                        continue

                    if getattr(delta, "tool_calls", None) and not tool_call_detected:
                        tool_call_detected = True

                    content = getattr(delta, "content", None)
                    if not content:
                        continue

                    if modality == "audio" and tool_call_detected:
                        continue
                    elif modality == "audio":
                        if first_audio:
                            first_audio = False
                            ttfa = time.perf_counter() - generation_start
                            logger.info("Time to first audio: %.3fs", ttfa)
                            interrupted = self._interrupted
                            self._interrupted = False
                            try:
                                if self._model._room:
                                    await self._model._room.local_participant.publish_data(
                                        json.dumps({"ttfa": ttfa, "interrupted": interrupted}).encode(),
                                        topic="latency",
                                    )
                            except Exception:
                                logger.warning("Failed to publish latency")
                        frame = _wav_bytes_to_frame(base64.b64decode(content))
                        if frame and self._current_audio_stream:
                            self._current_audio_stream.push(frame)
                    elif modality == "text" or modality is None:
                        assistant_text += content
                        if self._current_text_stream:
                            self._current_text_stream.push(content)

            completion = state.get_final_completion()
            tool_calls = completion.choices[0].message.tool_calls if completion.choices else None

            if tool_calls:
                try:
                    if self._model._room:
                        ttfa = time.perf_counter() - generation_start
                        await self._model._room.local_participant.publish_data(
                            json.dumps({"ttfa": ttfa, "tool_call": True}).encode(),
                            topic="latency",
                        )
                except Exception:
                    logger.warning("Failed to publish tool call latency")
                self._conversation.append(user_message)
                for tc in tool_calls:
                    logger.info("Tool call: %s(%s) [%s]", tc.function.name, tc.function.arguments[:100], tc.id)
                    if self._current_function_stream:
                        self._current_function_stream.send_nowait(
                            FunctionCall(
                                call_id=tc.id,
                                name=tc.function.name,
                                arguments=tc.function.arguments,
                            )
                        )
                    try:
                        if self._model._room:
                            await self._model._room.local_participant.publish_data(
                                json.dumps({"name": tc.function.name, "arguments": tc.function.arguments}).encode(),
                                topic="tool_call",
                            )
                    except Exception:
                        logger.warning("Failed to publish tool call event")
            else:
                if assistant_text:
                    self._conversation.append(user_message)
                    self._conversation.append({
                        "role": "assistant",
                        "content": assistant_text,
                    })

        except asyncio.CancelledError:
            logger.info("Generation cancelled")
        except Exception:
            logger.exception("Chat completion error")
        finally:
            self._finish_generation()

    def _finish_generation(self) -> None:
        if self._current_text_stream:
            self._current_text_stream.close()
            self._current_text_stream = None
        if self._current_audio_stream:
            self._current_audio_stream.close()
            self._current_audio_stream = None
        if self._current_function_stream:
            self._current_function_stream.close()
            self._current_function_stream = None

    async def aclose(self) -> None:
        self._closed = True
        if self._generation_task and not self._generation_task.done():
            self._generation_task.cancel()
            try:
                await self._generation_task
            except (asyncio.CancelledError, Exception):
                pass
        self._finish_generation()
        await self._client.close()
        self._model._sessions.discard(self)
