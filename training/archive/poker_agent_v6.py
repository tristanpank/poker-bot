#!/usr/bin/env python3
"""
Poker Agent V6 - Adaptive Opponent Modeling (Script Version)

KEY FEATURES:
1. Dual-Branch Architecture:
   - MLP Branch: Processes static game state (cards, pot, stacks)
   - LSTM Branch: Processes action history sequence to detect betting patterns
2. Explicit History Tracking: Environment returns sequence of recent actions
3. Adaptive Training: Trains against mixed opponent pool to force context-aware strategy
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
import random
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Dict, Any

from pokerkit import Automation, NoLimitTexasHoldem, Card

# Constants
SEED = 42
MAX_HISTORY_LEN = 20  # Length of action history sequence
ACTION_EMBED_DIM = 8  # Dimension for embedding actions
HIDDEN_DIM_LSTM = 64
HIDDEN_DIM_MLP = 128

# Action Constants
ENV_FOLD = 0
ENV_CHECK_CALL = 1
ENV_BET_RAISE = 2
NUM_ACTIONS = 3

# Action Markers for History (Vocabulary for LSTM)
# 0: Pad, 1: Fold, 2: Check/Call, 3: Bet/Raise (Agent)
# 4: Fold, 5: Check/Call, 6: Bet/Raise (Opponent)
ACT_PAD = 0
ACT_V_FOLD = 1
ACT_V_CHECK_CALL = 2
ACT_V_BET_RAISE = 3
OPP_FOLD = 4
OPP_CHECK_CALL = 5
OPP_BET_RAISE = 6
HISTORY_VOCAB_SIZE = 7

# Set random seeds
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class PokerKitGymEnvV6(gym.Env):
    """
    Gymnasium wrapper for PokerKit's No-Limit Texas Hold'em.
    Enhanced to return 'action_history' in observation.
    """
    
    def __init__(self, num_players: int = 2, starting_stack: int = 1000, 
                 small_blind: int = 5, big_blind: int = 10):
        super().__init__()
        
        self.num_players = num_players
        self.starting_stack = starting_stack
        self.small_blind = small_blind
        self.big_blind = big_blind
        
        self.game_state_dim = 52*2 + 52*5 + num_players + 1 + 1 + 4 + 1 # Cards + stacks + pot + actor + street + active_player
        
        # Observation is a dict: 'game_state' (vector) and 'history' (integer sequence)
        self.observation_space = spaces.Dict({
            'game_state': spaces.Box(low=0, high=1, shape=(self.game_state_dim,), dtype=np.float32),
            'history': spaces.Box(low=0, high=HISTORY_VOCAB_SIZE-1, shape=(MAX_HISTORY_LEN,), dtype=np.int64)
        })
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        
        self.state = None
        self.agent_player_index = 0
        self.action_history = deque(maxlen=MAX_HISTORY_LEN)
        
    def _card_to_index(self, card: Card) -> int:
        ranks = '23456789TJQKA'
        suits = 'cdhs'
        rank_idx = ranks.index(card.rank)
        suit_idx = suits.index(card.suit)
        return rank_idx * 4 + suit_idx
    
    def _encode_card(self, card: Optional[Card]) -> np.ndarray:
        encoding = np.zeros(52, dtype=np.float32)
        if card is not None:
            encoding[self._card_to_index(card)] = 1.0
        return encoding
    
    def _flatten_cards(self, cards) -> List:
        flat = []
        for item in cards:
            if hasattr(item, 'rank'):
                flat.append(item)
            else:
                flat.extend(self._flatten_cards(item))
        return flat
    
    def _get_observation(self) -> Dict[str, Any]:
        """Construct full observation dict with state vector and history sequence."""
        
        # 1. Build Game State Vector
        state_vector = []
        
        # Hole Cards (Agent only)
        hole_cards = self._flatten_cards(self.state.hole_cards[self.agent_player_index])
        for i in range(2):
            if i < len(hole_cards):
                state_vector.extend(self._encode_card(hole_cards[i]))
            else:
                state_vector.extend(np.zeros(52, dtype=np.float32))
        
        # Board Cards
        board_cards = self._flatten_cards(self.state.board_cards)
        for i in range(5):
            if i < len(board_cards):
                state_vector.extend(self._encode_card(board_cards[i]))
            else:
                state_vector.extend(np.zeros(52, dtype=np.float32))
        
        # Stacks (Normalized)
        for i in range(self.num_players):
            stack = self.state.stacks[i] / self.starting_stack
            state_vector.append(min(stack, 2.0))
        
        # Pot
        total_pot = sum(self.state.bets)
        state_vector.append(total_pot / (self.starting_stack * self.num_players))
        
        # Active Player Index (who is to act)
        if self.state.actor_index is not None:
            state_vector.append(self.state.actor_index / max(1, self.num_players - 1))
        else:
            state_vector.append(0.0)
        
        # Street (Preflop, Flop, Turn, River)
        street = [0.0, 0.0, 0.0, 0.0]
        num_board = len(board_cards)
        if num_board == 0:
            street[0] = 1.0
        elif num_board == 3:
            street[1] = 1.0
        elif num_board == 4:
            street[2] = 1.0
        else:
            street[3] = 1.0
        state_vector.extend(street)
        
        # Agent Index (constant, but usefull for Transformer if generalized)
        state_vector.append(float(self.agent_player_index))

        # 2. Build History Sequence (Padding to Fixed Length)
        history_seq = list(self.action_history)
        pad_len = MAX_HISTORY_LEN - len(history_seq)
        history_padded = [ACT_PAD] * pad_len + history_seq # Pad at beginning or end? Usually beginning for RNNs
        
        return {
            'game_state': np.array(state_vector, dtype=np.float32),
            'history': np.array(history_padded, dtype=np.int64)
        }
    
    def _update_history(self, player_idx: int, action: int):
        """Append action to history deque with player context."""
        # Convert raw action to player-specific action token
        if player_idx == self.agent_player_index:
            if action == ENV_FOLD: token = ACT_V_FOLD
            elif action == ENV_CHECK_CALL: token = ACT_V_CHECK_CALL
            else: token = ACT_V_BET_RAISE
        else:
            if action == ENV_FOLD: token = OPP_FOLD
            elif action == ENV_CHECK_CALL: token = OPP_CHECK_CALL
            else: token = OPP_BET_RAISE
            
        self.action_history.append(token)

    def _get_legal_actions(self) -> List[int]:
        legal = []
        if self.state.can_fold():
            legal.append(ENV_FOLD)
        if self.state.can_check_or_call():
            legal.append(ENV_CHECK_CALL)
        if self.state.can_complete_bet_or_raise_to():
            legal.append(ENV_BET_RAISE)
        return legal if legal else [ENV_CHECK_CALL] # Should never be empty if game active
    
    def _execute_action(self, action: int) -> None:
        if action == ENV_FOLD:
            if self.state.can_fold():
                self.state.fold()
            elif self.state.can_check_or_call():
                self.state.check_or_call()
        elif action == ENV_CHECK_CALL:
            if self.state.can_check_or_call():
                self.state.check_or_call()
            elif self.state.can_fold(): # Fallback
                self.state.fold()
        elif action == ENV_BET_RAISE:
            if self.state.can_complete_bet_or_raise_to():
                min_raise = self.state.min_completion_betting_or_raising_to_amount
                max_raise = self.state.max_completion_betting_or_raising_to_amount
                # Simple AI raise logic: min raise * 2, clamped
                raise_amount = min(min_raise * 2, max_raise)
                self.state.complete_bet_or_raise_to(raise_amount)
            elif self.state.can_check_or_call():
                self.state.check_or_call()
    
    def _run_automations(self) -> None:
        while self.state.can_burn_card():
            self.state.burn_card('??')
        while self.state.can_deal_board():
            self.state.deal_board()
        while self.state.can_push_chips():
            self.state.push_chips()
        while self.state.can_pull_chips():
            self.state.pull_chips()
    
    def reset(self, seed=None, options=None) -> Tuple[Dict, Dict]:
        super().reset(seed=seed)
        self.action_history.clear()
        
        self.state = NoLimitTexasHoldem.create_state(
            automations=(
                Automation.ANTE_POSTING,
                Automation.BET_COLLECTION,
                Automation.BLIND_OR_STRADDLE_POSTING,
                Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
                Automation.HAND_KILLING,
                Automation.CHIPS_PUSHING,
                Automation.CHIPS_PULLING,
            ),
            ante_trimming_status=True,
            raw_antes={-1: 0},
            raw_blinds_or_straddles=(self.small_blind, self.big_blind),
            min_bet=self.big_blind,
            raw_starting_stacks=[self.starting_stack] * self.num_players,
            player_count=self.num_players,
        )
        
        while self.state.can_deal_hole():
            self.state.deal_hole()
        
        self._run_automations()
        
        # Track initial active player if any (rare right after deal but possible)
        # Note: automation handles blinds, so action starts with UTG/SB
        
        return self._get_observation(), {'legal_actions': self._get_legal_actions()}
    
    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        current_actor = self.state.actor_index
        
        # Logic: If it's step(), it means it's Agent's turn (or being acted on behalf)
        # We record the action taken
        if current_actor is not None:
             self._update_history(current_actor, action)

        self._execute_action(action)
        self._run_automations()
        
        done = self.state.status is False
        
        reward = 0.0
        if done:
            final_stack = self.state.stacks[self.agent_player_index]
            reward = (final_stack - self.starting_stack) / self.big_blind
        
        obs = self._get_observation()
        info = {
            'legal_actions': self._get_legal_actions() if not done else [],
            'current_player': self.state.actor_index if not done else None
        }
        
        return obs, reward, done, False, info
    
    def get_final_reward(self) -> float:
        return (self.state.stacks[self.agent_player_index] - self.starting_stack) / self.big_blind
    
    def update_opponent_history(self, action: int):
        """Manually update history when opponent acts (called from training loop)."""
        opp_idx = 1 - self.agent_player_index # 2-player assumption
        self._update_history(opp_idx, action)


# --- ADAPTIVE AGENT ARCHITECTURE ---

class DualBranchDRQN(nn.Module):
    """
    V6 Architecture:
    1. Card Branch (MLP): Processes board/hand state.
    2. History Branch (LSTM): Processes sequence of actions (The 'Context' Memory).
    """
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        
        # --- Branch 1: Static State Processing ---
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # --- Branch 2: Action History Processing ---
        self.action_embedding = nn.Embedding(HISTORY_VOCAB_SIZE, ACTION_EMBED_DIM)
        self.lstm = nn.LSTM(
            input_size=ACTION_EMBED_DIM,
            hidden_size=HIDDEN_DIM_LSTM,
            batch_first=True
        )
        
        # --- Merger & Value Head ---
        # Input: 128 (state) + 64 (history context) = 192
        self.value_head = nn.Sequential(
            nn.Linear(128 + HIDDEN_DIM_LSTM, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
    def forward(self, state, history):
        # state: (batch, state_dim)
        # history: (batch, seq_len)
        
        # 1. State Features
        s_feat = self.state_net(state)
        
        # 2. History Features
        # Embed integer actions -> vectors
        h_embed = self.action_embedding(history) # (batch, seq, embed_dim)
        # LSTM processing
        lstm_out, (hn, cn) = self.lstm(h_embed) 
        # Take last hidden state as context summary
        h_context = hn[-1] # (batch, hidden_dim)
        
        # 3. Combine
        combined = torch.cat([s_feat, h_context], dim=1)
        
        # 4. Q-Values
        q_values = self.value_head(combined)
        return q_values

class ReplayBufferV6:
    """Stores transitions including history sequences."""
    def __init__(self, capacity=20000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, transition):
        # (state, history, action, reward, next_state, next_history, done, legal_actions)
        self.buffer.append(transition)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, min(len(self.buffer), batch_size))
    
    def __len__(self):
        return len(self.buffer)

class AdaptiveAgent:
    def __init__(self, state_dim, action_dim=NUM_ACTIONS, lr=1e-4):
        self.model = DualBranchDRQN(state_dim, action_dim).to(device)
        self.target_model = DualBranchDRQN(state_dim, action_dim).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.99995 # Slow decay
        
    def select_action(self, obs, legal_actions, eval_mode=False):
        if not eval_mode and random.random() < self.epsilon:
            return random.choice(legal_actions)
        
        state_t = torch.FloatTensor(obs['game_state']).unsqueeze(0).to(device)
        h_t = torch.LongTensor(obs['history']).unsqueeze(0).to(device)
        
        with torch.no_grad():
            q_values = self.model(state_t, h_t)
            
        q_numpy = q_values.cpu().numpy().flatten()
        
        # Mask illegal actions
        masked_q = np.full(NUM_ACTIONS, -np.inf)
        for a in legal_actions:
            masked_q[a] = q_numpy[a]
            
        return int(np.argmax(masked_q))

    def train(self, buffer, batch_size=64):
        if len(buffer) < batch_size:
            return None
        
        batch = buffer.sample(batch_size)
        
        # Unpack batch
        # t: (s, h, a, r, ns, nh, d, legal)
        states = torch.FloatTensor(np.array([t[0] for t in batch])).to(device)
        histories = torch.LongTensor(np.array([t[1] for t in batch])).to(device)
        actions = torch.LongTensor(np.array([t[2] for t in batch])).to(device)
        rewards = torch.FloatTensor(np.array([t[3] for t in batch])).to(device)
        next_states = torch.FloatTensor(np.array([t[4] for t in batch])).to(device)
        next_histories = torch.LongTensor(np.array([t[5] for t in batch])).to(device)
        dones = torch.FloatTensor(np.array([t[6] for t in batch])).to(device)
        
        # Current Q
        current_q = self.model(states, histories).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q (Double DQN)
        with torch.no_grad():
            # Select best action with online model
            next_actions = self.model(next_states, next_histories).argmax(1).unsqueeze(1)
            # Eval with target model
            target_q_next = self.target_model(next_states, next_histories).gather(1, next_actions).squeeze(1)
            target = rewards + (1 - dones) * self.gamma * target_q_next
            
        loss = F.mse_loss(current_q, target)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

# --- OPPONENTS ---
class ManiacAgent:
    """Always Raises if possible."""
    def select_action(self, legal_actions):
        if ENV_BET_RAISE in legal_actions: return ENV_BET_RAISE
        if ENV_CHECK_CALL in legal_actions: return ENV_CHECK_CALL
        return ENV_FOLD # Should verify this logic, fold if can't raise?

class NitAgent:
    """Folds to aggression, Calls if checked."""
    def select_action(self, legal_actions):
        if ENV_FOLD in legal_actions and ENV_CHECK_CALL in legal_actions:
            # If facing bet, fold 90%
            if random.random() < 0.9: return ENV_FOLD
        if ENV_CHECK_CALL in legal_actions: return ENV_CHECK_CALL
        return ENV_FOLD

class RandomAgent:
    def select_action(self, legal_actions):
        return random.choice(legal_actions)

# --- TRAINING LOOP ---

def train_v6(num_hands=25000):
    env = PokerKitGymEnvV6()
    agent = AdaptiveAgent(env.game_state_dim)
    buffer = ReplayBufferV6(capacity=50000)
    
    opponents = [ManiacAgent(), NitAgent(), RandomAgent()]
    opp_names = ['Maniac', 'Nit', 'Random']
    
    print(f"Starting V6 Training: {num_hands} hands")
    
    rewards_history = []
    loss_history = []
    
    for hand in range(num_hands):
        # 1. Pick Opponent (Randomly switch to force adaptation)
        opp_idx = random.randint(0, 2)
        opponent = opponents[opp_idx]
        
        obs, info = env.reset()
        done = False
        episode_transitions = []
        
        # Pending transition for agent
        # We need to store: (obs, action) -> wait for reward -> store next_obs
        pending_agent_obs = None
        pending_agent_action = None
        
        while not done:
            current_player = env.state.actor_index
            
            if current_player == env.agent_player_index:
                # -- Agent Turn --
                
                # Close previous transition if existed (opponent acted in between)
                if pending_agent_obs is not None:
                    # Previous action led to this state (after opponent response)
                    # Use 0 reward for intermediate steps
                    episode_transitions.append((
                        pending_agent_obs['game_state'],
                        pending_agent_obs['history'],
                        pending_agent_action,
                        0.0,
                        obs['game_state'],
                        obs['history'],
                        False,
                        info['legal_actions']
                    ))
                
                # Decide
                legal = info['legal_actions']
                action = agent.select_action(obs, legal)
                
                # Execute
                pending_agent_obs = obs # Snapshot current state/history
                pending_agent_action = action
                obs, reward, done, _, info = env.step(action)
                
                if done:
                    # Hand ended on Agent's move (e.g. fold or opponent fold immediately? No, if agent folds hand ends)
                    episode_transitions.append((
                        pending_agent_obs['game_state'],
                        pending_agent_obs['history'],
                        pending_agent_action,
                        0.0, # Will fill final reward later
                        obs['game_state'], # Terminal state
                        obs['history'],
                        True,
                        []
                    ))
                    
            else:
                # -- Opponent Turn --
                legal = info['legal_actions']
                action = opponent.select_action(legal)
                
                # Update environment history manually since step() isn't called by external agent logic directly in this loop structure? 
                # Wait, env.step() handles logic. But here we are calling opponent logic.
                # We need to tell Env that opponent acted so it updates history.
                env.update_opponent_history(action)
                
                # Execute in Env (as if agent did nothing, but we need to advance state)
                # PokerKitEnv usually expects step() to be called. 
                # The env wrapper I wrote has _execute_action. We should use env.step() but maybe modify it?
                # Actually, standard Gym duality: step() advances. But poker is turn based.
                # Let's use internal env methods to advance state for opponent.
                env._execute_action(action)
                env._run_automations()
                
                # Check status after opponent move
                done = env.state.status is False
                if done:
                    # Hand ended on Opponent's move
                    # We must record the transition from the Agent's last move to this terminal state
                    if pending_agent_obs is not None:
                        # Agent acted -> Opponent acted -> Done
                        # We need proper next_obs (the terminal observation)
                        term_obs = env._get_observation()
                        episode_transitions.append((
                            pending_agent_obs['game_state'],
                            pending_agent_obs['history'],
                            pending_agent_action,
                            0.0, # Filled later
                            term_obs['game_state'],
                            term_obs['history'],
                            True,
                            []
                        ))
                else: 
                    # Game continues, update 'obs' for next loop (Agent's perspective)
                    obs = env._get_observation()
                    info['legal_actions'] = env._get_legal_actions()
        
        # End of Hand: Propagate Reward
        final_reward = env.get_final_reward()
        rewards_history.append(final_reward)
        
        # Monte Carlo Return: Assign final reward to ALL actions in hand?
        # Or standard TD: Assign to last, 0 to others? 
        # V5 failed partially due to scarcity. But assigning to ALL correlates all actions to result.
        # Let's try assigning to ALL for strong signal, but maybe discounted? 
        # For poker, just assigning final outcome to all decisions is a robust baseline (MC).
        for i in range(len(episode_transitions)):
            s, h, a, r, ns, nh, d, l = episode_transitions[i]
            # Replace 0.0 with final_reward
            buffer.push((s, h, a, final_reward, ns, nh, d, l))
            
        # Optimization
        if len(buffer) > 1000:
            loss = agent.train(buffer)
            if loss: loss_history.append(loss)
            
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
        
        if hand % 500 == 0:
            agent.update_target()
            
        if hand % 1000 == 0:
            avg_r = np.mean(rewards_history[-100:])
            print(f"Hand {hand} | Avg Reward: {avg_r:.2f} | Epsilon: {agent.epsilon:.2f} | Opponent: {opp_names[opp_idx]}")

    return agent, rewards_history

def evaluate(agent, num_hands=500):
    env = PokerKitGymEnvV6()
    opps = {'Maniac': ManiacAgent(), 'Nit': NitAgent(), 'Random': RandomAgent()}
    
    print("\n--- EVALUATION ---")
    for name, opponent in opps.items():
        total_reward = 0
        wins = 0
        for _ in range(num_hands):
            obs, info = env.reset()
            done = False
            while not done:
                if env.state.actor_index == env.agent_player_index:
                    action = agent.select_action(obs, info['legal_actions'], eval_mode=True)
                    obs, _, done, _, info = env.step(action)
                else:
                    action = opponent.select_action(info['legal_actions'])
                    env.update_opponent_history(action)
                    env._execute_action(action)
                    env._run_automations()
                    done = env.state.status is False
                    if not done:
                        obs = env._get_observation()
                        info['legal_actions'] = env._get_legal_actions()
            
            rew = env.get_final_reward()
            total_reward += rew
            if rew > 0: wins += 1
            
        print(f"Vs {name}: Avg {total_reward/num_hands:.2f} BB/hand | Win Rate: {wins/num_hands:.2%}")

if __name__ == "__main__":
    agent, hist = train_v6(num_hands=20000)
    evaluate(agent)
