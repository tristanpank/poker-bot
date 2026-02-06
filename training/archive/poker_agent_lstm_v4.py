"""
LSTM Poker Agent V4 - Fixed Reward Collection + Extended Training

Fixes from V3:
1. Reward collection now properly captures reward when opponent folds
2. State is updated after opponent acts
3. Training extended to 150 sessions for better maniac performance
4. Evaluation reward capture fixed
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

# Set random seeds
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Action Space
ENV_FOLD = 0
ENV_CHECK_CALL = 1
ENV_BET_RAISE = 2
NUM_ACTIONS = 3


class PokerKitGymEnv(gym.Env):
    """Gymnasium wrapper for PokerKit's No-Limit Texas Hold'em."""
    
    def __init__(self, num_players: int = 2, starting_stack: int = 1000, 
                 small_blind: int = 5, big_blind: int = 10):
        super().__init__()
        
        self.num_players = num_players
        self.starting_stack = starting_stack
        self.small_blind = small_blind
        self.big_blind = big_blind
        
        # State: 2 hole cards (52*2) + 5 board (52*5) + stacks + pot + position + street
        self.game_state_dim = 52*2 + 52*5 + num_players + 1 + 1 + 4  # 372
        
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.game_state_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        
        self.state = None
        self.agent_player_index = 0
        
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
    
    def _get_game_state(self) -> np.ndarray:
        """Get the full game state as a flat vector."""
        state_vector = []
        
        # Hole cards (104 dims)
        hole_cards = self._flatten_cards(self.state.hole_cards[self.agent_player_index])
        for i in range(2):
            if i < len(hole_cards):
                state_vector.extend(self._encode_card(hole_cards[i]))
            else:
                state_vector.extend(np.zeros(52, dtype=np.float32))
        
        # Board cards (260 dims)
        board_cards = self._flatten_cards(self.state.board_cards)
        for i in range(5):
            if i < len(board_cards):
                state_vector.extend(self._encode_card(board_cards[i]))
            else:
                state_vector.extend(np.zeros(52, dtype=np.float32))
        
        # Stack sizes (normalized)
        for i in range(self.num_players):
            stack = self.state.stacks[i] / self.starting_stack
            state_vector.append(min(stack, 2.0))
        
        # Pot size (normalized)
        total_pot = sum(self.state.bets)
        state_vector.append(total_pot / (self.starting_stack * self.num_players))
        
        # Position indicator
        if self.state.actor_index is not None:
            state_vector.append(self.state.actor_index / max(1, self.num_players - 1))
        else:
            state_vector.append(0.0)
        
        # Street indicator (one-hot)
        street = [0.0, 0.0, 0.0, 0.0]
        num_board = len(board_cards)
        if num_board == 0:
            street[0] = 1.0  # Preflop
        elif num_board == 3:
            street[1] = 1.0  # Flop
        elif num_board == 4:
            street[2] = 1.0  # Turn
        else:
            street[3] = 1.0  # River
        state_vector.extend(street)
        
        return np.array(state_vector, dtype=np.float32)
    
    def _get_legal_actions(self) -> List[int]:
        legal = []
        if self.state.can_fold():
            legal.append(ENV_FOLD)
        if self.state.can_check_or_call():
            legal.append(ENV_CHECK_CALL)
        if self.state.can_complete_bet_or_raise_to():
            legal.append(ENV_BET_RAISE)
        return legal if legal else [ENV_CHECK_CALL]
    
    def _execute_action(self, action: int) -> None:
        if action == ENV_FOLD:
            if self.state.can_fold():
                self.state.fold()
            elif self.state.can_check_or_call():
                self.state.check_or_call()
        elif action == ENV_CHECK_CALL:
            if self.state.can_check_or_call():
                self.state.check_or_call()
            elif self.state.can_fold():
                self.state.fold()
        elif action == ENV_BET_RAISE:
            if self.state.can_complete_bet_or_raise_to():
                min_raise = self.state.min_completion_betting_or_raising_to_amount
                max_raise = self.state.max_completion_betting_or_raising_to_amount
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
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        
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
        
        return self._get_game_state(), {'legal_actions': self._get_legal_actions()}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self._execute_action(action)
        self._run_automations()
        
        done = self.state.status is False
        
        reward = 0.0
        if done:
            final_stack = self.state.stacks[self.agent_player_index]
            reward = (final_stack - self.starting_stack) / self.big_blind
        
        obs = self._get_game_state()
        info = {
            'legal_actions': self._get_legal_actions() if not done else [],
            'current_player': self.state.actor_index if not done else None
        }
        
        return obs, reward, done, False, info
    
    def get_current_player(self) -> Optional[int]:
        if self.state.status is False:
            return None
        return self.state.actor_index


class SequenceReplayBuffer:
    """Replay buffer that stores full episodes with initial hidden state for DRQN training."""
    
    def __init__(self, capacity: int = 5000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, episode: List[Tuple], initial_hidden: Tuple[torch.Tensor, torch.Tensor]):
        """Store a complete episode with its initial hidden state."""
        if len(episode) > 0:
            self.buffer.append((episode, initial_hidden))
    
    def sample(self, batch_size: int):
        """Sample a batch of (episode, initial_hidden) tuples."""
        batch = random.sample(list(self.buffer), min(batch_size, len(self.buffer)))
        return batch
    
    def __len__(self):
        return len(self.buffer)


class DRQN(nn.Module):
    """Deep Recurrent Q-Network - LSTM processes full state sequence."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, 
                 output_dim: int = NUM_ACTIONS, num_layers: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Feature extractor
        self.fc_feat = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=128, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True
        )
        
        # Q-value head
        self.fc_out = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor, hidden: Optional[Tuple] = None):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        batch_size, seq_len, _ = x.size()
        
        x_flat = x.view(-1, x.size(2))
        features = self.fc_feat(x_flat)
        features = features.view(batch_size, seq_len, -1)
        
        if hidden is None:
            lstm_out, new_hidden = self.lstm(features)
        else:
            lstm_out, new_hidden = self.lstm(features, hidden)
        
        if seq_len == 1:
            q_values = self.fc_out(lstm_out[:, -1, :])
        else:
            q_values = self.fc_out(lstm_out)
        
        return q_values, new_hidden
    
    def init_hidden(self, batch_size: int):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h, c)


class ManiacAgent:
    """Always raises/bets when possible, otherwise calls."""
    
    def select_action(self, legal_actions: List[int]) -> int:
        if ENV_BET_RAISE in legal_actions:
            return ENV_BET_RAISE
        if ENV_CHECK_CALL in legal_actions:
            return ENV_CHECK_CALL
        return ENV_FOLD


class NitAgent:
    """Always folds when facing aggression, checks otherwise."""
    
    def select_action(self, legal_actions: List[int]) -> int:
        if ENV_FOLD in legal_actions and ENV_CHECK_CALL in legal_actions:
            if random.random() < 0.8:
                return ENV_FOLD
        if ENV_CHECK_CALL in legal_actions:
            return ENV_CHECK_CALL
        return ENV_FOLD


class RandomAgent:
    """Randomly selects from legal actions."""
    
    def select_action(self, legal_actions: List[int]) -> int:
        return random.choice(legal_actions)


class DRQNAgent:
    """DRQN Agent with session-based LSTM state management."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 lr: float = 1e-3, gamma: float = 0.99,
                 epsilon_start: float = 1.0, epsilon_min: float = 0.05,
                 epsilon_decay: float = 0.9995):
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.model = DRQN(input_dim, hidden_dim).to(device)
        self.target_model = DRQN(input_dim, hidden_dim).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        self.current_hidden = None
    
    def start_new_session(self):
        """Reset LSTM hidden state at start of a new session."""
        self.current_hidden = None
        print("[Session Reset] LSTM hidden state cleared.")
    
    def get_hidden_for_episode(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get current hidden state to store for episode."""
        if self.current_hidden is None:
            return self.model.init_hidden(1)
        return (self.current_hidden[0].clone(), self.current_hidden[1].clone())
    
    def select_action(self, state: np.ndarray, legal_actions: List[int], 
                      eval_mode: bool = False) -> int:
        if not eval_mode and random.random() < self.epsilon:
            return random.choice(legal_actions)
        
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        with torch.no_grad():
            q_values, self.current_hidden = self.model(state_t, self.current_hidden)
        
        q_values = q_values.cpu().numpy().flatten()
        masked_q = np.full(NUM_ACTIONS, -np.inf)
        for a in legal_actions:
            masked_q[a] = q_values[a]
        
        return int(np.argmax(masked_q))
    
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def train_step(self, buffer: SequenceReplayBuffer, batch_size: int = 16) -> Optional[float]:
        if len(buffer) < batch_size:
            return None
        
        batch = buffer.sample(batch_size)
        total_loss = 0.0
        num_transitions = 0
        
        for episode, initial_hidden in batch:
            if len(episode) == 0:
                continue
            
            hidden = (initial_hidden[0].to(device), initial_hidden[1].to(device))
            
            for obs, action, reward, next_obs, done, legal_actions in episode:
                state_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
                q_values, hidden = self.model(state_t, hidden)
                q_value = q_values[0, action]
                
                if done:
                    target = reward
                else:
                    next_state_t = torch.FloatTensor(next_obs).unsqueeze(0).to(device)
                    with torch.no_grad():
                        next_q_values, _ = self.target_model(next_state_t, hidden)
                        target = reward + self.gamma * next_q_values.max().item()
                
                loss = F.mse_loss(q_value, torch.tensor(target, device=device))
                total_loss += loss
                num_transitions += 1
        
        if num_transitions > 0:
            avg_loss = total_loss / num_transitions
            
            self.optimizer.zero_grad()
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            return avg_loss.item()
        
        return None


def train_agent(num_sessions: int = 150, hands_per_session: int = 100,
                batch_size: int = 16, target_update_freq: int = 10):
    """
    Train the DRQN agent with session-based LSTM state management.
    
    FIXED: Reward is now correctly captured when opponent folds.
    Extended training: 150 sessions (up from 100) for better maniac performance.
    """
    env = PokerKitGymEnv(num_players=2)
    agent = DRQNAgent(input_dim=env.game_state_dim)
    buffer = SequenceReplayBuffer(capacity=5000)
    
    opponents = {
        'maniac': ManiacAgent(),
        'nit': NitAgent(),
        'random': RandomAgent()
    }
    
    rewards_history = []
    session_rewards = []
    loss_history = []
    
    print("=" * 60)
    print("LSTM Poker Agent V4 - Fixed Reward + Extended Training")
    print("=" * 60)
    print("\nKey fixes:")
    print("  - Reward correctly captured when opponent folds")
    print("  - State updated after opponent acts")
    print("  - Extended training: 150 sessions (15,000 hands)")
    print("=" * 60)
    print("\nStarting training...")
    print(f"Sessions: {num_sessions}, Hands per session: {hands_per_session}")
    print(f"Total hands: {num_sessions * hands_per_session}")
    print()
    
    for session in range(num_sessions):
        # Select opponent - first 75 sessions maniac, last 75 nit
        if session < num_sessions // 2:
            opponent_name = 'maniac'
        else:
            opponent_name = 'nit'
        
        opponent = opponents[opponent_name]
        
        # Start new session - RESET LSTM hidden state
        agent.start_new_session()
        
        session_reward = 0.0
        hands_played = 0
        
        for hand in range(hands_per_session):
            # Get initial hidden state for this episode
            initial_hidden = agent.get_hidden_for_episode()
            
            # Reset environment
            state, info = env.reset()
            episode_transitions = []
            done = False
            
            while not done:
                current_player = env.get_current_player()
                legal_actions = info['legal_actions']
                
                if current_player is None:
                    break
                
                if current_player == env.agent_player_index:
                    # Agent's turn
                    action = agent.select_action(state, legal_actions)
                    next_state, reward, done, truncated, info = env.step(action)
                    
                    # Store transition (reward will be updated at end)
                    episode_transitions.append((
                        state, action, 0.0, next_state, done,
                        info.get('legal_actions', [])
                    ))
                    
                    state = next_state
                else:
                    # Opponent's turn
                    opp_action = opponent.select_action(legal_actions)
                    next_state, reward, done, truncated, info = env.step(opp_action)
                    state = next_state  # FIX: Update state after opponent acts
            
            # FIX: Calculate final reward from stack change (works regardless of who acted last)
            final_reward = (env.state.stacks[env.agent_player_index] - env.starting_stack) / env.big_blind
            
            # Update last transition with actual reward
            if episode_transitions:
                last = episode_transitions[-1]
                episode_transitions[-1] = (last[0], last[1], final_reward, last[3], True, last[5])
                buffer.push(episode_transitions, initial_hidden)
                rewards_history.append(final_reward)
                session_reward += final_reward
                hands_played += 1
            
            # Train
            if len(buffer) >= batch_size:
                loss = agent.train_step(buffer, batch_size)
                if loss is not None:
                    loss_history.append(loss)
            
            # Update epsilon per hand
            agent.update_epsilon()
        
        # Update target network
        if session % target_update_freq == 0:
            agent.update_target_network()
        
        avg_session_reward = session_reward / max(1, hands_played)
        session_rewards.append(avg_session_reward)
        
        print(f"Session {session+1}/{num_sessions} | Opponent: {opponent_name} | "
              f"Avg Reward: {avg_session_reward:.2f} | Epsilon: {agent.epsilon:.3f}")
    
    print("\nTraining complete!")
    
    return agent, rewards_history, session_rewards, loss_history


def evaluate_agent(agent: DRQNAgent, num_hands: int = 100, 
                   opponent_type: str = 'random') -> Dict:
    """Evaluate agent against a specific opponent type."""
    env = PokerKitGymEnv(num_players=2)
    
    opponents = {
        'maniac': ManiacAgent(),
        'nit': NitAgent(),
        'random': RandomAgent()
    }
    opponent = opponents[opponent_type]
    
    # Reset hidden state for evaluation
    agent.start_new_session()
    
    wins = 0
    losses = 0
    ties = 0
    total_reward = 0.0
    
    for hand in range(num_hands):
        state, info = env.reset()
        done = False
        
        while not done:
            current_player = env.get_current_player()
            legal_actions = info['legal_actions']
            
            if current_player is None:
                break
            
            if current_player == env.agent_player_index:
                action = agent.select_action(state, legal_actions, eval_mode=True)
            else:
                action = opponent.select_action(legal_actions)
            
            next_state, reward, done, truncated, info = env.step(action)
            state = next_state  # FIX: Always update state
        
        # FIX: Calculate reward from stack change (works regardless of who acted last)
        hand_reward = (env.state.stacks[env.agent_player_index] - env.starting_stack) / env.big_blind
        
        total_reward += hand_reward
        if hand_reward > 0:
            wins += 1
        elif hand_reward < 0:
            losses += 1
        else:
            ties += 1
    
    results = {
        'opponent': opponent_type,
        'hands_played': num_hands,
        'wins': wins,
        'losses': losses,
        'ties': ties,
        'win_rate': wins / num_hands,
        'total_reward': total_reward,
        'avg_reward': total_reward / num_hands
    }
    
    print(f"\n=== Evaluation vs {opponent_type.upper()} ===")
    print(f"Hands: {num_hands}")
    print(f"W/L/T: {wins}/{losses}/{ties}")
    print(f"Win Rate: {results['win_rate']:.1%}")
    print(f"Total Profit: {total_reward:.1f} BBs")
    print(f"Avg Profit/Hand: {results['avg_reward']:.2f} BBs")
    
    return results


if __name__ == "__main__":
    # Train the agent
    agent, rewards_history, session_rewards, loss_history = train_agent()
    
    # Visualize training
    print("\n" + "=" * 60)
    print("TRAINING VISUALIZATION")
    print("=" * 60)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Session rewards
    axes[0].plot(session_rewards, alpha=0.7)
    axes[0].axhline(y=0, color='r', linestyle='--', label='Break-even')
    axes[0].axvline(x=75, color='g', linestyle='--', alpha=0.5, label='Switch to Nit')
    axes[0].set_xlabel('Session')
    axes[0].set_ylabel('Average Reward (BBs)')
    axes[0].set_title('Training Progress - Session Rewards')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss history
    if loss_history:
        window = 100
        smoothed_loss = np.convolve(loss_history, np.ones(window)/window, mode='valid')
        axes[1].plot(smoothed_loss, alpha=0.7)
        axes[1].set_xlabel('Training Step')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Training Loss (Smoothed)')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_results_v4.png', dpi=150)
    plt.show()
    print("\nTraining plot saved to training_results_v4.png")
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    
    results = {}
    for opp in ['random', 'maniac', 'nit']:
        results[opp] = evaluate_agent(agent, num_hands=100, opponent_type=opp)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for opp, r in results.items():
        status = "✓ PASS" if r['avg_reward'] > 0 else "✗ FAIL"
        print(f"{opp.capitalize():10} | Avg: {r['avg_reward']:+.2f} BB/hand | {status}")
