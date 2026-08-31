import { AccessToken, RoomAgentDispatch, RoomConfiguration } from "livekit-server-sdk";
import { NextRequest, NextResponse } from "next/server";

const AGENT_NAMES = {
  completions: "voice-assistant-completions",
  realtime: "voice-assistant-realtime",
} as const;

type AgentMode = keyof typeof AGENT_NAMES;

export async function POST(request: NextRequest) {
  const apiKey = process.env.LIVEKIT_API_KEY;
  const apiSecret = process.env.LIVEKIT_API_SECRET;
  const serverUrl = process.env.LIVEKIT_URL;

  if (!apiKey || !apiSecret || !serverUrl) {
    return NextResponse.json(
      { error: "LiveKit credentials not configured" },
      { status: 500 }
    );
  }

  const body = await request.json().catch(() => ({}));
  const mode: AgentMode = body.agentMode === "realtime" ? "realtime" : "completions";
  const agentName = AGENT_NAMES[mode];

  const roomName = `voice-room-${Math.random().toString(36).slice(2, 9)}`;
  const participantName = `user-${Math.random().toString(36).slice(2, 7)}`;

  const at = new AccessToken(apiKey, apiSecret, {
    identity: participantName,
    name: participantName,
  });

  at.addGrant({
    room: roomName,
    roomJoin: true,
    canPublish: true,
    canSubscribe: true,
  });

  at.roomConfig = new RoomConfiguration({
    agents: [new RoomAgentDispatch({ agentName })],
  });

  const token = await at.toJwt();

  return NextResponse.json({
    serverUrl,
    roomName,
    participantName,
    participantToken: token,
  });
}
