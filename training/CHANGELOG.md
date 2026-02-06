# Changelog

All notable changes to the poker bot model architecture and training methodology.

## [V18] - Hybrid Training with Massive Bust Penalty

**Key Changes:**
- **Massive bust penalty**: -200 BB (4x harsher than V17's -50 BB)
- **Stronger all-in loss penalty**: -15 BB base (was -10 BB)
- **Higher frequency penalty**: -2.0 per excess all-in (was -1.0)
- **Hybrid training mode**: Combines scripted opponents with self-play
  - Phase 1 (sessions 1-1000): 60% scripted, 10% self-play, 30% mixed
  - Phase 2 (sessions 1000+): 33% scripted, 33% self-play, 34% mixed

**Rationale:** Busting ends the game entirely, making all prior profits meaningless. The massive penalty teaches the agent to never risk elimination.

---

## [V17] - Self-Play Training with Stronger Penalties

**Key Changes:**
- **Self-play training**: `OpponentPool` manages snapshots of past models as opponents
- **Stronger all-in loss penalties**: -10 BB base (was -3 BB in V16)
- **Harsher bust penalty**: -50 BB (was -20 BB)
- **Lower all-in frequency threshold**: Target <8% (was 15%)

**New Components:**
- `OpponentPool` class for managing past model snapshots

---

## [V16] - Session-Based Rewards

**Key Changes:**
- **Session-based training**: Rewards calculated over full sessions, not single hands
- **All-in outcome penalties**: Scaled by decision quality
  - Bad: -3.0 BB (equity < 40%)
  - Marginal: -1.5 BB (equity 40-65%)
  - Okay: -1.5 BB (equity 65-80%)
- **All-in frequency penalty**: First 2 free, then -0.5 per extra
- **Bust penalty**: -20 BB for falling below 5 BB stack
- **Survival bonus**: +3 BB for completing session without busting

---

## [V15] - 6-Max Table with Position-Aware Rewards

**Key Changes:**
- **Expanded state dimension**: 520 features (up from 385) for 6-player table
- **Position-aware rewards**:
  - UTG/MP bonus: 1.5x multiplier for profitable plays
  - CO/BTN bonus: 1.2x multiplier
  - Blinds: 1.0x (baseline)
- **Multi-way pot risk penalties**: Higher penalties with more opponents
- **Position consistency bonus**: Rewards tighter play from early position

**Position Constants:**
- UTG (0), MP (1), CO (2), BTN (3), SB (4), BB (5)

---

## [V14] - Advanced Reward Shaping

**Key Changes:**
- **Risk penalty**: Penalizes all-in/large bets with weak hands
  - Based on: equity, risk ratio (amount risked / stack)
- **Value extraction bonus**: Rewards maximizing winnings with strong hands
- **Action consistency rewards**: Matches action aggressiveness to hand strength
- **Pot building bonus**: Rewards gradual pot building with monster hands

**Reward Functions:**
- `compute_risk_penalty(action, equity, risk_ratio)`
- `compute_value_extraction_bonus(won, pot_won, equity, pot_before)`
- `compute_action_consistency(action, equity)`
- `compute_pot_building_bonus(actions_history, equities_history)`

---

## [V13] - Dueling DQN with Prioritized Experience Replay

**Key Changes:**
- **Dueling DQN architecture**: Separates value V(s) and advantage A(s,a) streams
  - Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
- **Prioritized Experience Replay (PER)**:
  - `SumTree` data structure for efficient sampling
  - Samples proportional to TD-error
  - Importance sampling weights to correct bias
  - Beta annealing from 0.4 to 1.0

**Action Space (6 actions):**
- FOLD (0), CALL (1)
- RAISE_SMALL (2): 2x min raise
- RAISE_MEDIUM (3): 0.5x pot
- RAISE_LARGE (4): 1x pot
- ALL_IN (5)

**Hand Strength Categories:**
- 0: Trash (<30%), 1: Marginal (30-45%), 2: Decent (45-60%)
- 3: Strong (60-75%), 4: Monster (>75%)

---

## Earlier Versions (Archive)

### V12
- Basic multiprocessing training with `StatisticsPokerNet`
- Separate model and worker modules for Jupyter compatibility

### V6
- Dual-branch DRQN architecture
- MLP branch for static game state
- LSTM branch for action history (opponent modeling)

### V5
- Simple DQN with episode-based replay buffer
- Basic opponent agents: Maniac, Nit, Random
