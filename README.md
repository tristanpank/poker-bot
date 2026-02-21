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
| `POST` | `/cv/analyze-raw` | Analyze a raw RGBA byte-stream frame (used by the frontend) |
| `DELETE` | `/cv/session` | Clear per-session baseline state |

Metrics returned include: brightness, motion, edge density, activity zone, pulse BPM (POS algorithm), pulse confidence, skin coverage, stress index, emotion state (`calm` / `focused` / `tense` / `agitated`), bluff risk score, bluff level (`low` / `watch` / `elevated`), bluff delta vs. baseline, and analysis/stream FPS.

## Frontend

The Next.js app (`next-poker-app/`) provides a real-time deception-proxy dashboard:

- **WebRTC loopback stream** – captures your camera at the highest negotiated resolution and frame rate, routes it through a local WebRTC peer connection for frame decoding, and samples frames for analysis
- **Live CV metrics panel** – displays all backend CV metrics (pulse, stress, emotion, bluff risk, signal quality, etc.) updated in real time
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

```bash
pytest backend/tests/
```
