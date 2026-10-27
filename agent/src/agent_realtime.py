import asyncio
import json
import logging
import os

from dotenv import load_dotenv
from livekit.agents import AgentServer, AgentSession, AutoSubscribe, JobContext, TurnHandlingOptions, cli, metrics
from livekit.agents.voice.events import FunctionToolsExecutedEvent, MetricsCollectedEvent
from livekit.plugins import openai, silero

from assistant import VoiceAssistant
from nemotron_duplex import NemotronDuplexRealtimeModel

load_dotenv(".env.local")
logger = logging.getLogger("voice-assistant-realtime")

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8091/v1")
VLLM_MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen3-Omni-30B-A3B-Instruct")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")
# Nemotron VoiceChat is a model-native full-duplex model, not a turn-based one
# like Qwen3-Omni -- it needs a different RealtimeModel (see nemotron_duplex.py)
# and no local VAD/turn handling. Detected by model name so existing
# Qwen3-Omni deployments (no code or env changes) keep working unmodified.
IS_NEMOTRON = "nemotron" in VLLM_MODEL.lower()

server = AgentServer()


@server.rtc_session(agent_name="voice-assistant-realtime")
async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    if IS_NEMOTRON:
        session = AgentSession(
            llm=NemotronDuplexRealtimeModel(
                base_url=VLLM_BASE_URL,
                api_key=VLLM_API_KEY,
                model=VLLM_MODEL,
            ),
        )
    else:
        session = AgentSession(
            # vLLM-Omni's /v1/realtime endpoint doesn't support server-side VAD /
            # turn detection, so drive turn-taking locally with Silero VAD instead.
            llm=openai.realtime.RealtimeModel(
                base_url=VLLM_BASE_URL,
                api_key=VLLM_API_KEY,
                model=VLLM_MODEL,
                input_audio_transcription=None,
            ),
            vad=silero.VAD.load(),
            turn_handling=TurnHandlingOptions(
                turn_detection="vad",
                interruption={"mode": "vad"},
            ),
        )

    def publish(payload: dict, topic: str) -> None:
        async def _publish() -> None:
            try:
                await ctx.room.local_participant.publish_data(
                    json.dumps(payload).encode(), topic=topic,
                )
            except Exception:
                logger.warning("Failed to publish %s telemetry", topic)

        asyncio.create_task(_publish())

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent) -> None:
        m = ev.metrics
        if not isinstance(m, metrics.RealtimeModelMetrics) or m.ttft < 0:
            return
        publish({"ttfa": m.ttft, "interrupted": m.cancelled}, "latency")

    @session.on("function_tools_executed")
    def _on_function_tools_executed(ev: FunctionToolsExecutedEvent) -> None:
        for call in ev.function_calls:
            logger.info("Tool call: %s(%s) [%s]", call.name, call.arguments[:100], call.call_id)
            publish({"name": call.name, "arguments": call.arguments}, "tool_call")

    await session.start(
        agent=VoiceAssistant(),
        room=ctx.room,
    )

    logger.info("Voice assistant started, connected to vLLM-Omni realtime API at %s", VLLM_BASE_URL)


if __name__ == "__main__":
    cli.run_app(server)
