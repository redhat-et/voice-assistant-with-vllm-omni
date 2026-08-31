import logging
import os

from dotenv import load_dotenv
from livekit.agents import AgentServer, AgentSession, AutoSubscribe, JobContext, cli
from livekit.plugins import silero

from assistant import VoiceAssistant
from vllm_realtime import VLLMRealtimeModel

load_dotenv(".env.local")
logger = logging.getLogger("voice-assistant-completions")

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8091/v1")
VLLM_MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen3-Omni-30B-A3B-Instruct")

server = AgentServer()


@server.rtc_session(agent_name="voice-assistant-completions")
async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    model = VLLMRealtimeModel(
        base_url=VLLM_BASE_URL,
        model=VLLM_MODEL,
        room=ctx.room,
    )

    session = AgentSession(
        llm=model,
        vad=silero.VAD.load(),
        turn_detection="vad",
    )
    await session.start(
        agent=VoiceAssistant(),
        room=ctx.room,
    )

    logger.info("Voice assistant started, connected to vLLM-Omni chat completions at %s", VLLM_BASE_URL)


if __name__ == "__main__":
    cli.run_app(server)
