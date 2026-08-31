"use client";

import {
  LiveKitRoom,
  RoomAudioRenderer,
  BarVisualizer,
  DisconnectButton,
  useVoiceAssistant,
  useDataChannel,
  useTranscriptions,
} from "@livekit/components-react";
import "@livekit/components-styles";
import { useCallback, useEffect, useRef, useState } from "react";

interface ConnectionDetails {
  serverUrl: string;
  roomName: string;
  participantName: string;
  participantToken: string;
}

type AgentMode = "completions" | "realtime";

const AGENT_MODES: { value: AgentMode; label: string }[] = [
  { value: "completions", label: "Completions API" },
  { value: "realtime", label: "Realtime API" },
];

interface LatencyPoint {
  ms: number;
  interrupted: boolean;
  toolCall: boolean;
}

interface ToolCallEntry {
  name: string;
  arguments: string;
  timestamp: number;
}

const MAX_LATENCY_POINTS = 20;

function LatencyChart({ points }: { points: LatencyPoint[] }) {
  if (points.length === 0) return null;

  const ms = points.map((p) => p.ms);
  const max = Math.max(...ms, 100);
  const h = 160;
  const w = 480;
  const barW = w / MAX_LATENCY_POINTS;
  const normalPoints = points.filter((p) => !p.interrupted && !p.toolCall);
  const normalMs = normalPoints.map((p) => p.ms);

  return (
    <div className="w-full max-w-lg">
      <div
        className="flex items-end justify-between rounded-lg bg-zinc-900 p-4"
        style={{ height: h + 32 }}
      >
        <svg
          width={w}
          height={h}
          viewBox={`0 0 ${w} ${h}`}
          className="w-full"
        >
          {points.map((p, i) => {
            const barH = (p.ms / max) * (h - 20);
            const x = i * barW;
            const color = p.toolCall
              ? "#38bdf8"
              : p.interrupted
                ? "#a78bfa"
                : p.ms < 300
                  ? "#22c55e"
                  : p.ms < 500
                    ? "#eab308"
                    : "#ef4444";
            return (
              <g key={i}>
                <rect
                  x={x + 2}
                  y={h - barH}
                  width={barW - 4}
                  height={barH}
                  rx={3}
                  fill={color}
                  opacity={p.interrupted || p.toolCall ? 0.5 : 0.85}
                />
                {p.interrupted && (
                  <line
                    x1={x + 2}
                    y1={h - barH + 4}
                    x2={x + barW - 4}
                    y2={h - 4}
                    stroke={color}
                    strokeWidth={1}
                    opacity={0.6}
                  />
                )}
                <text
                  x={x + barW / 2}
                  y={h - barH - 4}
                  textAnchor="middle"
                  fontSize={10}
                  fill="#a1a1aa"
                >
                  {Math.round(p.ms)}
                </text>
              </g>
            );
          })}
        </svg>
      </div>
      <div className="mt-2 flex items-center justify-between text-xs text-zinc-500">
        <span>
          Last:{" "}
          <span className="text-zinc-300">
            {Math.round(ms[ms.length - 1])}ms
          </span>
        </span>
        <span>
          Avg:{" "}
          <span className="text-zinc-300">
            {normalMs.length > 0
              ? Math.round(
                  normalMs.reduce((a, b) => a + b, 0) / normalMs.length,
                )
              : "—"}
            ms
          </span>
        </span>
        <span>
          Min:{" "}
          <span className="text-zinc-300">
            {normalMs.length > 0 ? Math.round(Math.min(...normalMs)) : "—"}ms
          </span>
        </span>
        <span>
          <span className="text-sky-400">■</span> tool
          {" "}
          <span className="text-violet-400">■</span> interrupted
        </span>
      </div>
    </div>
  );
}

function ToolCallLog({ entries }: { entries: ToolCallEntry[] }) {
  if (entries.length === 0) return null;

  return (
    <div className="w-full max-w-lg">
      <div className="rounded-lg bg-zinc-900 p-3">
        <p className="mb-2 text-xs font-medium text-zinc-400">Tool Calls</p>
        <div className="flex flex-col gap-1.5 max-h-40 overflow-y-auto">
          {entries.map((entry, i) => (
            <div key={i} className="rounded bg-zinc-800 px-2.5 py-1.5 text-xs">
              <span className="font-mono text-emerald-400">{entry.name}</span>
              <span className="text-zinc-500 ml-1 break-all">
                ({entry.arguments.length > 80
                  ? entry.arguments.slice(0, 80) + "…"
                  : entry.arguments})
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function TranscriptPanel() {
  // Only the assistant's speech is transcribed — neither agent backend
  // transcribes user audio — so every segment here is the assistant's.
  const transcriptions = useTranscriptions();
  const scrollRef = useRef<HTMLDivElement>(null);

  const sorted = [...transcriptions].sort(
    (a, b) => a.streamInfo.timestamp - b.streamInfo.timestamp,
  );
  const lastText = sorted[sorted.length - 1]?.text;

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [sorted.length, lastText]);

  if (sorted.length === 0) return null;

  return (
    <div className="w-full max-w-lg">
      <div
        ref={scrollRef}
        className="flex max-h-56 flex-col gap-2 overflow-y-auto rounded-lg bg-zinc-900 p-4"
      >
        {sorted.map((t) => (
          <p key={t.streamInfo.id} className="text-sm">
            <span className="text-sky-400">Assistant:</span>{" "}
            <span className="text-zinc-200">{t.text}</span>
          </p>
        ))}
      </div>
    </div>
  );
}

function AgentVisualizer() {
  const { state, audioTrack } = useVoiceAssistant();
  const [points, setPoints] = useState<LatencyPoint[]>([]);
  const [toolCalls, setToolCalls] = useState<ToolCallEntry[]>([]);

  useDataChannel("latency", (msg) => {
    try {
      const data = JSON.parse(new TextDecoder().decode(msg.payload));
      if (typeof data.ttfa === "number") {
        setPoints((prev) => [
          ...prev.slice(-MAX_LATENCY_POINTS + 1),
          { ms: data.ttfa * 1000, interrupted: data.interrupted === true, toolCall: data.tool_call === true },
        ]);
      }
    } catch {}
  });

  useDataChannel("tool_call", (msg) => {
    try {
      const data = JSON.parse(new TextDecoder().decode(msg.payload));
      if (typeof data.name === "string") {
        setToolCalls((prev) => [
          ...prev.slice(-9),
          { name: data.name, arguments: data.arguments || "", timestamp: Date.now() },
        ]);
      }
    } catch {}
  });

  return (
    <div className="flex flex-col items-center gap-6">
      <div className="h-48 w-full max-w-md">
        <BarVisualizer state={state} barCount={5} trackRef={audioTrack} />
      </div>
      <p className="text-sm text-zinc-400 capitalize">{state}</p>
      <TranscriptPanel />
      <ToolCallLog entries={toolCalls} />
      <LatencyChart points={points} />
    </div>
  );
}

export default function VoiceAssistant() {
  const [connectionDetails, setConnectionDetails] =
    useState<ConnectionDetails | null>(null);
  const [connecting, setConnecting] = useState(false);
  const [agentMode, setAgentMode] = useState<AgentMode>("completions");

  const handleConnect = useCallback(async () => {
    setConnecting(true);
    try {
      const response = await fetch("/api/token", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ agentMode }),
      });
      if (!response.ok) throw new Error("Failed to get token");
      const details: ConnectionDetails = await response.json();
      setConnectionDetails(details);
    } catch (err) {
      console.error("Connection failed:", err);
      setConnecting(false);
    }
  }, [agentMode]);

  const handleDisconnected = useCallback(() => {
    setConnectionDetails(null);
    setConnecting(false);
  }, []);

  if (!connectionDetails) {
    return (
      <div className="flex flex-col items-center gap-6">
        <div className="flex rounded-full bg-zinc-900 p-1">
          {AGENT_MODES.map((mode) => (
            <button
              key={mode.value}
              onClick={() => setAgentMode(mode.value)}
              className={`rounded-full px-4 py-1.5 text-sm font-medium transition-colors ${
                agentMode === mode.value
                  ? "bg-white text-black"
                  : "text-zinc-400 hover:text-zinc-200"
              }`}
            >
              {mode.label}
            </button>
          ))}
        </div>
        <button
          onClick={handleConnect}
          disabled={connecting}
          className="rounded-full bg-white px-8 py-4 text-lg font-medium text-black transition-opacity hover:opacity-80 disabled:opacity-50"
        >
          {connecting ? "Connecting..." : "Start Conversation"}
        </button>
      </div>
    );
  }

  return (
    <LiveKitRoom
      token={connectionDetails.participantToken}
      serverUrl={connectionDetails.serverUrl}
      connect={true}
      audio={true}
      onDisconnected={handleDisconnected}
      className="flex flex-col items-center gap-8"
    >
      <AgentVisualizer />
      <RoomAudioRenderer />
      <DisconnectButton className="rounded-full border border-zinc-700 px-6 py-3 text-sm text-zinc-300 transition-colors hover:border-red-500 hover:text-red-400">
        End Conversation
      </DisconnectButton>
    </LiveKitRoom>
  );
}
