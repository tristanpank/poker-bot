# Poker Bot

A reinforcement learning project for training a No-Limit Texas Hold'em poker agent using Deep Q-Networks (DQN) with various architectural improvements.

## Project Structure

```
├── src/
│   ├── models/          # Neural network definitions (v13-v18)
│   └── workers/         # Multiprocessing episode workers
├── notebooks/
│   ├── training/        # Training notebooks by version
│   └── play_against_bot.ipynb
├── checkpoints/         # Saved model weights (.pt/.pth)
├── results/             # Training curves and evaluation plots
├── archive/             # Older versions and experiments
└── requirements.txt
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the latest training notebook
jupyter notebook notebooks/training/poker_agent_v18.ipynb
```

## Model Versions

See [CHANGELOG.md](CHANGELOG.md) for detailed version history. Key milestones:

| Version | Key Features |
|---------|--------------|
| V13 | Dueling DQN + Prioritized Experience Replay |
| V14 | Risk penalties + Value extraction |
| V15 | 6-max table + Position-aware rewards |
| V16 | Session-based rewards + All-in penalties |
| V17 | Self-play training + Stronger penalties |
| V18 | Hybrid training + Massive bust penalty |

## Dependencies

- PyTorch
- PokerKit
- Gymnasium
- NumPy, Matplotlib, Seaborn

## Playing Against the Bot

```bash
jupyter notebook notebooks/play_against_bot.ipynb
```
