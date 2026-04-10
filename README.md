# Poker Bot

A No-Limit Texas Hold'em poker assistant with three components:

1. **Training** – Reinforcement learning pipeline (Deep Q-Networks) for training poker agents
2. **Backend** – FastAPI server that serves trained models and runs a real-time CV bluff-detection pipeline
3. **Frontend** – Next.js dashboard that streams your camera via WebRTC and displays live deception-proxy metrics

## Project Structure

```
├── training/
│   ├── src/
│   │   ├── models/          # Neural network definitions (v13-v18)
│   │   └── workers/         # Multiprocessing episode workers
│   ├── notebooks/
│   │   ├── training/        # Training notebooks by version
│   │   └── play_against_bot.ipynb
│   ├── checkpoints/         # Saved model weights (.pt/.pth)
│   ├── results/             # Training curves and evaluation plots
│   └── CHANGELOG.md
├── backend/
│   ├── routers/
│   │   ├── poker.py         # Bot action inference endpoints
│   │   └── cv.py            # Computer vision / bluff-detection endpoints
│   ├── services/
│   │   ├── model_service.py # DQN model loading and inference
│   │   ├── game_service.py  # Game state → observation conversion
│   │   └── cv_service.py    # Frame analysis pipeline
│   ├── models/schemas.py    # Pydantic request / response schemas
│   ├── config.py            # Settings (env vars, model paths)
│   └── main.py              # FastAPI app entry point
├── next-poker-app/          # Next.js 16 frontend
│   └── app/
│       └── page.tsx         # WebRTC camera stream + metrics dashboard
└── requirements.txt
```

## Prerequisites

- **Python 3.10+** with `pip`
- **Node.js 18+** with `npm` (for the frontend)
- Trained model checkpoints in `training/checkpoints/` (e.g. `poker_agent_v18.pt`)

## Local Development

### 1 – Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2 – Start the backend

```bash
uvicorn backend.main:app --reload
```

The API will be available at `http://localhost:8000`.  
Interactive API docs: `http://localhost:8000/docs`

> **Optional env vars**
>
> | Variable | Default | Description |
> |---|---|---|
> | `ENABLE_POKER_ROUTER` | `1` | Set to `0` to disable model-inference endpoints (useful when PyTorch is unavailable) |
> | `ENABLE_POKER_PRELOAD` | `0` | Set to `1` to preload the default model on startup |
> | `MODEL_VERSION` | `v18` | Default model version to use |
> | `MODEL_CHECKPOINT_DIR` | `training/checkpoints` | Path to checkpoint directory |

### 3 – Start the frontend

```bash
cd next-poker-app
npm install
npm run dev
```

The app will be available at `http://localhost:3000`.

> Set `NEXT_PUBLIC_BACKEND_URL` in `next-poker-app/.env.local` if your backend runs on a different host/port (defaults to `http://localhost:8000`).

### 4 – (Optional) Train a new model

```bash
# Run the latest training notebook
jupyter notebook training/notebooks/training/poker_agent_v18.ipynb

# Or play against an existing checkpoint
jupyter notebook training/notebooks/play_against_bot.ipynb
```

## Docker Setup

For consistent local development, you can run the backend using Docker and Docker Compose.

### 1 – Prerequisites
- **Docker** and **Docker Compose V2** are required.
- **Important for Linux Users**: Older versions of `docker-compose` (v1.x) will crash with a `KeyError: 'ContainerConfig'` when handling modern builds. 
  - On Ubuntu 24.04+, install the modern plugin: `sudo apt update && sudo apt install docker-compose-v2`
  - Always use the space command: `docker compose` (not `docker-compose`).

### 2 – Build and Run
From the root directory, run:

```bash
docker compose up --build backend
```

- **Hot Reloading**: The `backend/` directory is mounted into the container via volumes, so changes you make locally will auto-restart the server inside Docker.
- **Baked-in Models**: Trained models, checkpoints, and feature definitions are baked directly into the image to ensure the container is self-contained and ready to run.
- **Ports**: The backend is published at `http://localhost:8000` and Redis is published at `localhost:6379` for local development.

### 3 – Monitoring
```bash
docker compose logs -f backend
```

### 4 – Access the app on your phone with Cloudflare Quick Tunnel

This is the fastest way to test the site on your phone with HTTPS, which is especially helpful for camera and WebRTC flows.

1. Start the backend:

```powershell
docker compose up -d backend
cloudflared tunnel --url http://localhost:8000
```

Copy the backend `https://...trycloudflare.com` URL from the `cloudflared` output.

2. Start the frontend and point it at the backend tunnel:

```powershell
$env:NEXT_PUBLIC_BACKEND_URL="https://YOUR-BACKEND-URL.trycloudflare.com"
docker compose up -d frontend
cloudflared tunnel --url http://localhost:3000
```

3. Open the frontend `https://...trycloudflare.com` URL on your phone.

Keep both `cloudflared` terminals running while you test.

> **Notes**
>
> - Install `cloudflared` first if needed:
>   ```powershell
>   winget install Cloudflare.cloudflared
>   ```
> - The frontend will use `NEXT_PUBLIC_BACKEND_URL` when set. If it is not set, it falls back to `localhost` on your PC or to the current hostname on your local network.
> - Cloudflare Quick Tunnels are best for temporary development and testing, not production hosting.

### 5 – Automate tunnel startup and optionally update a permanent Short.io link

The repo includes a helper script that:

- starts the backend container
- opens a backend Quick Tunnel
- restarts the frontend with `NEXT_PUBLIC_BACKEND_URL` pointing at that backend tunnel
- opens a frontend Quick Tunnel
- optionally creates or updates a permanent Short.io link so the same short URL can point at the new frontend tunnel after each restart

Run it from the repo root:

```powershell
.\scripts\Start-PhoneTunnels.ps1
```

For repeat use, create a local ignored config file first:

```powershell
New-Item -ItemType Directory -Force .local\phone-tunnels | Out-Null
Copy-Item .\scripts\PhoneTunnels.config.example.psd1 .\.local\phone-tunnels\config.psd1
```

Then edit `.local\phone-tunnels\config.psd1` and fill in at least:

```powershell
@{
    ShortIoApiKey = 'your-shortio-api-key'
    ShortIoDomain = 'your-account.short.gy'
    ShortIoPath   = 'poker'
}
```

After that, just run:

```powershell
.\scripts\Start-PhoneTunnels.ps1
```

On the first run, the script creates `https://your-account.short.gy/poker`. On later runs, it reuses the saved Short.io link ID from `.local/phone-tunnels/state.json` and updates the destination automatically.

The script resolves values in this order:

- command-line parameters
- `.local\phone-tunnels\config.psd1`
- environment variables
- built-in defaults

The script prints the backend tunnel URL, frontend tunnel URL, and the permanent Short.io URL if configured.

## Backend API

### Poker endpoints (`/poker`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/poker/health` | Health check; lists available model checkpoints |
| `GET` | `/poker/models` | List all available model versions with metadata |
| `POST` | `/poker/action` | **Primary endpoint** – send the current game state and receive the bot's recommended action, equity estimate, and Q-values |

**`POST /poker/action` – example request body**

```json
{
  "community_cards": [{"rank": "A", "suit": "s"}, {"rank": "K", "suit": "h"}, {"rank": "Q", "suit": "d"}],
  "pot": 150,
  "players": [
    {"position": 0, "stack": 950, "bet": 50, "hole_cards": [{"rank": "A", "suit": "h"}, {"rank": "K", "suit": "s"}], "is_bot": true, "is_active": true},
    {"position": 1, "stack": 1000, "bet": 50, "hole_cards": null, "is_bot": false, "is_active": true}
  ],
  "bot_position": 0,
  "current_bet": 50,
  "big_blind": 10,
  "model_version": "v18"
}
```

**Example response**

```json
{
  "action": "RAISE_MEDIUM",
  "action_id": 3,
  "amount": 75,
  "equity": 0.72,
  "hand_strength_category": "Strong",
  "q_values": {"FOLD": -5.2, "CALL": 3.1, "RAISE_SMALL": 4.5, "RAISE_MEDIUM": 5.8, "RAISE_LARGE": 4.2, "ALL_IN": 1.1}
}
```

Available actions: `FOLD`, `CALL`, `RAISE_SMALL`, `RAISE_MEDIUM`, `RAISE_LARGE`, `ALL_IN`

### CV endpoints (`/cv`)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/cv/analyze` | Analyze a Base64-encoded RGBA frame and return bluff/stress metrics |
| `POST` | `/cv/analyze-raw` | Analyze a raw RGBA byte-stream frame |
| `POST` | `/cv/webrtc/offer` | WebRTC signaling: accept an SDP offer and return an SDP answer |
| `DELETE` | `/cv/session` | Clear per-session baseline state |

Metrics returned include: brightness, motion, edge density, activity zone, pulse BPM (POS algorithm), pulse confidence, skin coverage, stress index, emotion state (`calm` / `focused` / `tense` / `agitated`), bluff risk score, bluff level (`low` / `watch` / `elevated`), bluff delta vs. baseline, and analysis/stream FPS.

#### WebRTC ingest (`POST /cv/webrtc/offer`)

The primary ingest path. The frontend opens a WebRTC peer connection directly to the backend and sends a center-cropped (center 70% of width × height) video stream. A DataChannel (`metadata`) carries accurate per-frame capture timestamps from the frontend to the backend, and CV metrics are returned to the frontend over the same channel.

**Request body**

```json
{
  "sdp": "<SDP offer string>",
  "type": "offer",
  "sessionId": "<uuid>"
}
```

**Response**

```json
{
  "sdp": "<SDP answer string>",
  "type": "answer"
}
```

**DataChannel metadata message (frontend → backend)**

```json
{
  "sessionId": "<uuid>",
  "frameId": 42,
  "captureTs": 1765935300123,
  "streamFps": 29.9,
  "cropWidth": 896,
  "cropHeight": 504
}
```

`captureTs` is a Unix epoch millisecond timestamp derived from `performance.timeOrigin + requestVideoFrameCallback.now`, providing sub-millisecond accuracy for heart-rate estimation.

**DataChannel metrics message (backend → frontend)**

The backend responds to each decoded frame with the full `CvMetrics` JSON (same schema as `/cv/analyze` responses).

## Redis Data Flow & Webcam Metrics

The poker bot utilizes Redis to maintain game states and synchronize live webcam computer vision metrics between streaming players and the main game client.

### Redis Keys
- `poker:session:{session_id}`: Persistent game state tracking for the main poker session.
- `poker:webcam:session:{session_id}`: Stores a dictionary of the session's active opponents mapped by seat position, tracking whether they are connected and assigning them a unique `cv_session_id`.
- `poker:webcam:code:{code}`: A temporary mapping (1-hour TTL) that maps a 6-character join code to a specific `session_id`.
- `poker:webcam:metrics:{cv_session_id}`: Stores the most recently serialized `CvMetrics` JSON produced by the computer vision pipeline. This key has a short 10-second TTL and is continually refreshed by the WebRTC ingest service.

### Metrics Flow
1. **Initiation**: The main Play page hits `POST /session/webcam/generate-code` to generate a code mapping to the current `session_id`.
2. **Joining**: An opponent enters the code on the Join page, calling `POST /session/webcam/join` to get a designated `cv_session_id`.
3. **WebRTC Stream**: The opponent begins a WebRTC session, streaming video tracks directly to the backend `WebRtcIngestService`.
4. **Analysis & Persistence**: The backend `CvService` processes every frame, and the ingest service immediately writes the resulting JSON metrics into the `poker:webcam:metrics:{cv_session_id}` Redis key.
5. **Consumption**: The Play page polls `GET /session/webcam/status/{session_id}` which iterates over the connected opponents, fetches each of their active metric keys from Redis, and bundles the live CV analysis back to the client. This allows the poker bot (and UI) to consume real-time bluffing data without interrupting the game logic.

## Frontend

The Next.js app (`next-poker-app/`) provides a real-time deception-proxy dashboard:

- **WebRTC backend stream** – captures your camera at the highest negotiated resolution and frame rate, applies a center crop (central 70% × 70%) to focus on the face region, and sends the cropped stream directly to the backend via a WebRTC peer connection
- **DataChannel** – opens a `metadata` DataChannel on the same peer connection; the frontend sends accurate per-frame capture timestamps (via `requestVideoFrameCallback`) with each frame, and the backend returns CV metrics on the same channel
- **Live CV metrics panel** – displays all backend CV metrics (pulse, stress, emotion, bluff risk, signal quality, etc.) updated in real time via the DataChannel
- **Bluff-pressure trend chart** – SVG chart showing bluff-risk history over a rolling 30-second window
- **Session management** – automatically creates and clears per-session baseline state on the backend

> **Note:** Visual cues are heuristic only and are not a reliable lie detector.

## Model Versions

See [training/CHANGELOG.md](training/CHANGELOG.md) for detailed version history. Key milestones:

| Version | Key Features |
|---------|--------------|
| V13 | Dueling DQN + Prioritized Experience Replay |
| V14 | Risk penalties + Value extraction |
| V15 | 6-max table + Position-aware rewards |
| V16 | Session-based rewards + All-in penalties |
| V17 | Self-play training + Stronger penalties |
| V18 | Hybrid training + Massive bust penalty |

## Running Tests

You can run the full test suite locally if you have installed dependencies inside a virtual environment:

```bash
pytest backend/tests/
```

Alternatively, you can run the entire testing suite seamlessly through Docker (ensuring a completely sterile environment):

```bash
docker compose run --rm backend pytest backend/tests/
```
