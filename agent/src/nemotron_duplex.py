from __future__ import annotations

import dataclasses
import logging
from typing import Any
from urllib.parse import urlparse

from livekit.plugins.openai.realtime import RealtimeModel, RealtimeSession
from pydantic import BaseModel

logger = logging.getLogger("nemotron-duplex")

# Nemotron VoiceChat's duplex engine only accepts a connection that asks for
# its model-native full-duplex mode; everything else about the wire protocol
# (session.update, input_audio_buffer.append, etc.) is the same OpenAI
# Realtime API schema the base RealtimeModel already speaks. See
# vllm_omni.model_executor.models.nemotron_voicechat.duplex.plugin
# ._require_native_full_duplex, which rejects a session.update missing this.
_AUTO_RESPONSE_EXTRA_BODY = {"auto_response": True}


class NemotronDuplexRealtimeSession(RealtimeSession):
    """Adds Nemotron VoiceChat's required `extra_body.auto_response` to session.update.

    `_create_session_update_event` is a private extension point on the base
    session, but the only one available: the typed event it normally returns
    has no field for vendor extras. `_send_task` sends a dict as-is (only
    pydantic models go through `.model_dump()`), so returning a plain dict
    here with `extra_body` added is enough to get it onto the wire untouched.

    `extra_body` has to sit *inside* the `session` object, not as a sibling
    of `type`/`session`: the duplex websocket handler's first-message handoff
    (`RealtimeEnvelope.first_message`) does `session_payload = dict(payload["session"])`
    for a `session.update` event and discards everything else in the message,
    so a top-level `extra_body` never reaches `DuplexSessionConfig.from_realtime`.
    """

    def _create_session_update_event(self) -> dict[str, Any]:
        event = super()._create_session_update_event()
        data = (
            event.model_dump(by_alias=True, exclude_unset=True, exclude_defaults=False)
            if isinstance(event, BaseModel)
            else dict(event)
        )
        data.setdefault("session", {})["extra_body"] = _AUTO_RESPONSE_EXTRA_BODY
        return data


class NemotronDuplexRealtimeModel(RealtimeModel):
    """RealtimeModel for Nemotron VoiceChat's model-native full-duplex mode.

    Unlike OpenAI's Realtime API (client-triggered responses, optionally with
    server-side VAD deciding *when* to trigger one), Nemotron VoiceChat always
    listens and speaks concurrently and decides on its own when to respond --
    there is no "turn" to detect or interrupt. `turn_detection=None` keeps the
    base model from sending an OpenAI-style turn_detection config it would
    ignore anyway, and the capabilities are patched so AgentSession never
    tries to drive turns locally (no VAD needed) or hold back / cut the
    assistant's audio against local speech activity.
    """

    def __init__(self, *, base_url: str, model: str, api_key: str = "EMPTY") -> None:
        # The duplex engine lives behind /v1/realtime?duplex=1 -- a plain
        # /v1/realtime connection gets the turn-based path this model can't
        # satisfy. `autostart=0` matters just as much: without it the server
        # opens the session immediately on connect using a default payload
        # (no extra_body), before our session.update below ever arrives, and
        # rejects it the same way a plain connection would. `process_base_url`
        # preserves query params already on base_url, so this is the only
        # place that needs to know about either.
        parsed = urlparse(base_url)
        duplex_base_url = base_url if parsed.query else f"{base_url}?duplex=1&autostart=0"

        super().__init__(
            base_url=duplex_base_url,
            model=model,
            api_key=api_key,
            input_audio_transcription=None,
            turn_detection=None,
        )
        self._capabilities = dataclasses.replace(
            self._capabilities,
            turn_detection=True,
            supports_overlapping_speech=True,
        )

    def session(self, *, turn_detection_disabled: bool = False) -> NemotronDuplexRealtimeSession:
        if turn_detection_disabled:
            logger.warning(
                "turn_detection_disabled has no effect: Nemotron VoiceChat full-duplex "
                "has no client-driven turn mode to fall back to"
            )
        sess = NemotronDuplexRealtimeSession(self, turn_detection_disabled=False)
        self._sessions.add(sess)
        return sess
