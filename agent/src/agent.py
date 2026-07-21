import logging
import os

import httpx
from dotenv import load_dotenv
from livekit.agents import Agent, AgentServer, AgentSession, AutoSubscribe, JobContext, cli
from livekit.agents.llm import function_tool
from livekit.plugins import silero

from vllm_realtime import VLLMRealtimeModel

load_dotenv(".env.local")
logger = logging.getLogger("voice-assistant")

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8091/v1")

WMO_WEATHER_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "foggy", 48: "depositing rime fog",
    51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle",
    61: "slight rain", 63: "moderate rain", 65: "heavy rain",
    71: "slight snowfall", 73: "moderate snowfall", 75: "heavy snowfall",
    77: "snow grains", 80: "slight rain showers", 81: "moderate rain showers",
    82: "violent rain showers", 85: "slight snow showers", 86: "heavy snow showers",
    95: "thunderstorm", 96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
}

server = AgentServer()


@function_tool()
async def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    async with httpx.AsyncClient(timeout=10) as client:
        geo = await client.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": location, "count": 1},
        )
        geo_data = geo.json()
        results = geo_data.get("results")
        if not results:
            return f"Sorry, I could not find a location called {location}."

        place = results[0]
        lat, lon = place["latitude"], place["longitude"]
        name = place.get("name", location)
        country = place.get("country", "")

        weather = await client.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
            },
        )
        current = weather.json().get("current", {})
        temp = current.get("temperature_2m")
        humidity = current.get("relative_humidity_2m")
        wind = current.get("wind_speed_10m")
        code = current.get("weather_code", -1)
        condition = WMO_WEATHER_CODES.get(code, "unknown conditions")

        return (
            f"{name}, {country}: {condition}, "
            f"{temp} degrees Celsius, "
            f"{humidity} percent humidity, "
            f"wind {wind} kilometers per hour."
        )


class VoiceAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a helpful voice assistant. Respond naturally and concisely. You have access to tools — use them when appropriate.",
            tools=[get_weather],
        )


@server.rtc_session(agent_name="voice-assistant")
async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    model = VLLMRealtimeModel(
        base_url=VLLM_BASE_URL,
        model="Qwen/Qwen3-Omni-30B-A3B-Instruct",
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

    logger.info("Voice assistant started, connected to vLLM-Omni at %s", VLLM_BASE_URL)


if __name__ == "__main__":
    cli.run_app(server)
