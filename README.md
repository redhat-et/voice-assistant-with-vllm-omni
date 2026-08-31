# Voice Assistant with vLLM-Omni

Real-time voice assistant powered by [Qwen3-Omni](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) served via [vLLM-Omni](https://github.com/vllm-project/vllm-omni), with a [LiveKit](https://livekit.io/) frontend.

## Architecture

```
┌──────────┐  WebRTC   ┌──────────────┐  dispatch by agent name (from toggle)
│ Browser   │◄─────────►│ LiveKit      │─────────────┬──────────────────────┐
│ (React)   │  toggle   │ Server :7880 │              │                      │
│ :3000     │  picks    └──────────────┘              ▼                      ▼
└──────────┘  agent                     ┌────────────────────────┐  ┌─────────────────────┐
                                        │ agent-completions        │  │ agent-realtime        │
                                        │ /v1/chat/completions     │  │ /v1/realtime          │
                                        └────────────┬─────────────┘  └───────────┬───────────┘
                                                     │                            │
                                                     └────────────┬───────────────┘
                                                                  ▼
                                                       ┌─────────────────────┐
                                                       │ vLLM-Omni Qwen3-Omni │
                                                       │ :8091                │
                                                       └─────────────────────┘
```

The browser captures audio via WebRTC; LiveKit routes it to whichever Python agent the frontend toggle dispatched by name for that session — both agents talk to the same vLLM-Omni/Qwen3-Omni server, just through different APIs:

- **`agent-completions`** uses Silero VAD for turn detection, then sends the user's audio to vLLM-Omni's `/v1/chat/completions` endpoint as a base64-encoded WAV. Conversation history is maintained across turns in the agent process.
- **`agent-realtime`** streams audio to vLLM-Omni's `/v1/realtime` WebSocket endpoint (OpenAI Realtime API-compatible), still driving turn-taking locally with Silero VAD since vLLM-Omni's realtime endpoint has no server-side VAD.

Both return text and spoken audio directly from Qwen3-Omni (no separate STT/TTS). The frontend renders the assistant's spoken text as a live transcript — neither agent transcribes the user's own audio, so only the assistant's side of the conversation appears as text. Both agents also publish time-to-first-audio and tool-call events over LiveKit data channels (topics `latency` and `tool_call`), driving the frontend's latency chart and tool-call log regardless of which one is active — `agent-completions` publishes them from `vllm_realtime.py` directly, `agent-realtime` derives them from `AgentSession`'s generic `metrics_collected`/`function_tools_executed` events.

## Prerequisites

- **GPU server:** 2x NVIDIA H100 (or equivalent), [vLLM-Omni](https://github.com/vllm-project/vllm-omni) installed
- **Local machine:** Python 3.10+, Node.js 18+, pnpm, [livekit-server](https://github.com/livekit/livekit/releases)

Install LiveKit server (macOS):

```bash
brew install livekit
```

## Setup

### 1. GPU Server — Start vLLM-Omni

```bash
./scripts/start-vllm.sh
```

Or manually:

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --omni \
    --host 0.0.0.0 \
    --port 8091
```

Wait for "Application startup complete" before proceeding. Verify:

```bash
curl http://<gpu-server-ip>:8091/v1/models
```

### 2. Local Machine — Configure Environment

```bash
cp .env.example agent/.env.local
cp .env.example frontend/.env.local
```

Edit both `.env.local` files and set `VLLM_BASE_URL` to your GPU server:

```
VLLM_BASE_URL=http://<gpu-server-ip>:8091/v1
```

### 3. Local Machine — Start LiveKit Server

```bash
./scripts/start-livekit.sh
```

Dev mode uses API key `devkey` and secret `secret` (matching `.env.example` defaults).

### 4. Local Machine — Start the Agents

```bash
./scripts/start-agent.sh           # completions API agent (voice-assistant-completions)
./scripts/start-agent-realtime.sh  # realtime API agent (voice-assistant-realtime)
```

Each creates a virtual environment on first run, installs dependencies, and starts its agent in dev mode. Run both (in separate terminals) so the frontend toggle has something to dispatch to on either setting — you only need the one matching your toggle choice if you're just testing a single mode.

### 5. Local Machine — Start the Frontend

```bash
cd frontend
pnpm install
pnpm dev
```

Open http://localhost:3000, pick **Completions API** or **Realtime API**, click **Start Conversation**, and speak.

## OpenShift Deployment

Container images are built automatically via GitHub Actions and pushed to GHCR on every push to main:

- `ghcr.io/redhat-et/voice-assistant-with-vllm-omni/agent:latest`
- `ghcr.io/redhat-et/voice-assistant-with-vllm-omni/frontend:latest`

Deploy to OpenShift:

```bash
oc apply -k deploy/openshift/
```

See `deploy/openshift/` for the full set of manifests (LiveKit, both agents, frontend, vLLM-Omni with GPU scheduling). `agent.yaml` defines two separate Deployments/HPAs — `agent-completions` and `agent-realtime` — running the same container image with a different entrypoint, so both are dispatchable by the frontend toggle in the deployed environment.

## Project Structure

```
├── agent/                   # LiveKit Python agents
│   ├── pyproject.toml
│   └── src/
│       ├── assistant.py      # Shared Agent (instructions, weather tool)
│       ├── agent.py          # Completions-API entrypoint (voice-assistant-completions)
│       ├── agent_realtime.py # Realtime-API entrypoint (voice-assistant-realtime)
│       └── vllm_realtime.py  # RealtimeModel backed by chat completions
├── frontend/               # Next.js web UI
│   ├── app/
│   │   ├── api/token/      # JWT token generation + agent dispatch for LiveKit
│   │   └── page.tsx
│   └── components/
│       └── VoiceAssistant.tsx  # Agent toggle, transcript, latency/tool-call panels
├── deploy/
│   └── openshift/          # Kustomize manifests for OpenShift
├── .github/
│   └── workflows/          # CI: build and push container images
└── scripts/                # Startup scripts
```

## Configuration

All configuration is via environment variables in `.env.local` files:

| Variable | Default | Description |
|---|---|---|
| `LIVEKIT_URL` | `ws://localhost:7880` | LiveKit server WebSocket URL |
| `LIVEKIT_API_KEY` | `devkey` | LiveKit API key |
| `LIVEKIT_API_SECRET` | `secret` | LiveKit API secret |
| `VLLM_BASE_URL` | `http://localhost:8091/v1` | vLLM-Omni HTTP endpoint (serves both `/v1/chat/completions` and `/v1/realtime`) |
| `VLLM_MODEL` | `Qwen/Qwen3-Omni-30B-A3B-Instruct` | Model name passed to vLLM-Omni by both agents |

Which agent a session uses is picked at connect time via the frontend toggle, not an environment variable — it's sent as `agentMode` in the `POST /api/token` request body and mapped to an explicit LiveKit agent dispatch name (`voice-assistant-completions` or `voice-assistant-realtime`).

## Troubleshooting

**Agent can't connect to vLLM-Omni:**
- Ensure vLLM-Omni is running with `--omni` flag
- Check the GPU server firewall allows port 8091
- Verify: `curl http://<gpu-server-ip>:8091/v1/models`

**No audio response:**
- Check browser microphone permissions
- Verify the agent for the toggle mode you picked registered with LiveKit (check its logs for a "registered" message) — if only one of the two agents is running locally, sessions using the other toggle setting will connect but no assistant will ever join
- Ensure `LIVEKIT_URL` in frontend `.env.local` matches the LiveKit server address

**High latency:**
- Expected end-to-end latency is ~1-2s on 2x H100
- Check GPU utilization with `nvidia-smi`
