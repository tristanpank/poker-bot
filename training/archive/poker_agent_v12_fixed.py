"""
Poker Agent V12 Training Script - Fixed Version

This script fixes the multiprocessing issue where workers were using random
actions instead of the trained model. 

Run with: python poker_agent_v12_fixed.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt
from typing import List
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Import worker function from separate module
from poker_worker import run_single_episode
from poker_model import StatisticsPokerNet

# Constants
SEED = 42
NUM_ACTIONS = 3

# ============================================
# CONFIGURABLE SETTINGS
# ============================================
NUM_WORKERS = 4           # Number of parallel processes
EPISODES_PER_BATCH = 64   # Episodes to collect before training
EQUITY_ITERATIONS = 20    # Monte Carlo iterations for equity
BATCH_SIZE = 512          # Training batch size
BUFFER_CAPACITY = 300000  # Replay buffer size

# LOGGING SETTINGS
LOG_PERCENT = 2           # Log every X% of training
CHECKPOINT_PERCENT = 10   # Detailed checkpoint every X%

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================
# Reward Shaping
# ============================================

def compute_shaped_reward(final_bb_profit, contexts, actions):
    if not contexts:
        return final_bb_profit
    
    reward = final_bb_profit
    ev_adjustment = 0.0
    
    for ctx, action in zip(contexts, actions):
        if not ctx: continue
        equity = ctx.get('equity', 0.5)
        pot_bb = ctx.get('pot_bb', 0)
        to_call_bb = ctx.get('to_call_bb', 0)
        excess_equity = ctx.get('excess_equity', 0)
        
        if action == 0:  # Fold
            if excess_equity < -0.05:
                ev_adjustment += 0.5 * abs(excess_equity)
            elif excess_equity > 0.1:
                ev_adjustment -= 0.5 * excess_equity
        elif action == 1:  # Call
            if to_call_bb > 0:
                if excess_equity > 0.05:
                    ev_adjustment += 0.2 * excess_equity
                elif excess_equity < -0.1:
                    ev_adjustment -= 0.5 * abs(excess_equity) * (pot_bb / 10.0 + 1)
        elif action == 2:  # Raise
            if equity > 0.6:
                ev_adjustment += 0.3 * (equity - 0.5)
            elif equity < 0.35 and final_bb_profit <= 0:
                ev_adjustment -= 0.3
    
    last_pot = contexts[-1].get('pot_bb', 10) if contexts else 10
    pot_importance = min(last_pot / 15.0, 3.0)
    
    if final_bb_profit > 0:
        profit_component = final_bb_profit * (1 + 0.05 * pot_importance)
    else:
        profit_component = final_bb_profit * (1 + 0.1 * pot_importance)
    
    return profit_component + ev_adjustment


# ============================================
# Replay Buffer & Agent
# ============================================

class ReplayBuffer:
    def __init__(self, capacity=BUFFER_CAPACITY):
        self.buffer = deque(maxlen=capacity)
    
    def push_batch(self, transitions):
        self.buffer.extend(transitions)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, min(len(self.buffer), batch_size))
    
    def __len__(self):
        return len(self.buffer)


class StatisticsAgent:
    def __init__(self, state_dim=380, lr=5e-5):
        self.model = StatisticsPokerNet(state_dim).to(device)
        self.target_model = StatisticsPokerNet(state_dim).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.99990
    
    def train(self, buffer, batch_size=BATCH_SIZE):
        if len(buffer) < batch_size:
            return None
        
        batch = buffer.sample(batch_size)
        states = torch.FloatTensor(np.array([t[0] for t in batch])).to(device)
        actions = torch.LongTensor(np.array([t[1] for t in batch])).to(device)
        rewards = torch.FloatTensor(np.array([t[2] for t in batch])).to(device)
        next_states = torch.FloatTensor(np.array([t[3] for t in batch])).to(device)
        dones = torch.FloatTensor(np.array([t[4] for t in batch])).to(device)
        
        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_actions = self.model(next_states).argmax(1).unsqueeze(1)
            target_q_next = self.target_model(next_states).gather(1, next_actions).squeeze(1)
            target = rewards + (1 - dones) * self.gamma * target_q_next
        
        loss = F.mse_loss(current_q, target)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def get_model_state_dict_cpu(self):
        """Get model state dict on CPU for multiprocessing."""
        return {k: v.cpu() for k, v in self.model.state_dict().items()}


# ============================================
# Parallel Training Loop
# ============================================

def train_v12_parallel(num_hands=200000):
    """Training with process-based parallelism."""
    
    agent = StatisticsAgent()
    buffer = ReplayBuffer()
    
    stats = {'ValueBot': {'rewards': [], 'wins': 0, 'hands': 0},
             'BluffBot': {'rewards': [], 'wins': 0, 'hands': 0},
             'Balanced': {'rewards': [], 'wins': 0, 'hands': 0}}
    opp_types = ['value', 'bluff', 'balanced']
    opp_name_map = {'value': 'ValueBot', 'bluff': 'BluffBot', 'balanced': 'Balanced'}
    
    all_rewards = []
    all_shaped = []
    loss_history = []
    action_counts = {0: 0, 1: 0, 2: 0}
    
    start_time = time.time()
    total_hands = 0
    batch_num = 0
    
    # Calculate logging intervals
    log_interval = max(100, int(num_hands * LOG_PERCENT / 100))
    checkpoint_interval = max(500, int(num_hands * CHECKPOINT_PERCENT / 100))
    
    print("=" * 70)
    print(f"TRAINING V12 FIXED (Multiprocessing, {NUM_WORKERS} workers)")
    print("=" * 70)
    print(f"Target hands: {num_hands:,}")
    print(f"Episodes per batch: {EPISODES_PER_BATCH}")
    print(f"Equity iterations: {EQUITY_ITERATIONS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Log interval: every {log_interval:,} hands ({LOG_PERCENT}%)")
    print(f"Checkpoint interval: every {checkpoint_interval:,} hands ({CHECKPOINT_PERCENT}%)")
    print()
    
    # Epsilon decay
    decay_steps = int(num_hands * 0.8)
    agent.epsilon_decay = (agent.epsilon_min / agent.epsilon) ** (1 / decay_steps)
    print(f"Epsilon: {agent.epsilon:.2f} -> {agent.epsilon_min:.2f} at hand {decay_steps:,}")
    print("=" * 70)
    print()
    
    last_log_hand = 0
    last_checkpoint_hand = 0
    
    # Use fork context on Linux/WSL for better performance
    try:
        ctx = mp.get_context('fork')
        print("Using 'fork' context for multiprocessing")
    except ValueError:
        ctx = mp.get_context('spawn')
        print("Using 'spawn' context for multiprocessing")
    print()
    
    with ProcessPoolExecutor(max_workers=NUM_WORKERS, mp_context=ctx) as executor:
        while total_hands < num_hands:
            batch_num += 1
            
            # Generate batch of episode arguments - NOW INCLUDES MODEL WEIGHTS
            episode_args = []
            model_state_dict = agent.get_model_state_dict_cpu()
            for i in range(EPISODES_PER_BATCH):
                seed = random.randint(0, 2**31)
                opp = random.choice(opp_types)
                episode_args.append((seed, agent.epsilon, opp, EQUITY_ITERATIONS, model_state_dict))
            
            # Run episodes in parallel
            futures = [executor.submit(run_single_episode, args) for args in episode_args]
            
            # Collect results
            batch_transitions = []
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=120)
                    
                    # Compute shaped reward
                    shaped = compute_shaped_reward(
                        result['final_reward'],
                        result['contexts'],
                        result['actions']
                    )
                    
                    # Add transitions with shaped reward
                    for s, a, _, ns, d in result['transitions']:
                        batch_transitions.append((s, a, shaped, ns, d))
                    
                    # Track stats
                    opp_name = opp_name_map[result['opp_type']]
                    stats[opp_name]['rewards'].append(result['final_reward'])
                    stats[opp_name]['hands'] += 1
                    if result['won']:
                        stats[opp_name]['wins'] += 1
                    
                    all_rewards.append(result['final_reward'])
                    all_shaped.append(shaped)
                    
                    for a, c in result['action_counts'].items():
                        action_counts[a] += c
                    
                    total_hands += 1
                    
                except Exception as e:
                    print(f"Episode failed: {e}")
                    continue
            
            # Add to buffer
            buffer.push_batch(batch_transitions)
            
            # Update epsilon
            for _ in range(len(batch_transitions) // 2):
                agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
            
            # Training
            if len(buffer) > BATCH_SIZE * 2:
                for _ in range(4):
                    loss = agent.train(buffer)
                    if loss is not None:
                        loss_history.append(loss)
            
            if batch_num % 10 == 0:
                agent.update_target()
            
            # Logging
            if total_hands - last_log_hand >= log_interval:
                last_log_hand = total_hands
                elapsed = time.time() - start_time
                hands_per_sec = total_hands / elapsed if elapsed > 0 else 0
                eta_min = (num_hands - total_hands) / hands_per_sec / 60 if hands_per_sec > 0 else 0
                
                recent_len = min(log_interval, len(all_rewards))
                recent_reward = np.mean(all_rewards[-recent_len:]) if all_rewards else 0
                recent_shaped = np.mean(all_shaped[-recent_len:]) if all_shaped else 0
                recent_loss = np.mean(loss_history[-100:]) if loss_history else 0
                
                total_actions = sum(action_counts.values())
                fold_pct = action_counts[0] / total_actions * 100 if total_actions > 0 else 0
                call_pct = action_counts[1] / total_actions * 100 if total_actions > 0 else 0
                raise_pct = action_counts[2] / total_actions * 100 if total_actions > 0 else 0
                
                print(f"Hand {total_hands:,}/{num_hands:,} ({total_hands/num_hands*100:.1f}%) | "
                      f"Eps={agent.epsilon:.3f} | "
                      f"BB: {recent_reward:+.2f} | "
                      f"Loss: {recent_loss:.2f} | "
                      f"Speed: {hands_per_sec:.1f} h/s | "
                      f"ETA: {eta_min:.1f}m")
                print(f"        Fold {fold_pct:.1f}% | Call {call_pct:.1f}% | Raise {raise_pct:.1f}% | Buf: {len(buffer):,}")
            
            # Checkpoint
            if total_hands - last_checkpoint_hand >= checkpoint_interval:
                last_checkpoint_hand = total_hands
                print(f"\n{'='*70}")
                print(f"CHECKPOINT: {total_hands:,} / {num_hands:,} ({total_hands/num_hands*100:.1f}%)")
                print(f"{'='*70}")
                total_wins = sum(s['wins'] for s in stats.values())
                total_h = sum(s['hands'] for s in stats.values())
                print(f"Cumulative: {sum(all_rewards):+.1f} BB | Win Rate: {total_wins/max(1,total_h):.1%}")
                for name in ['ValueBot', 'BluffBot', 'Balanced']:
                    if stats[name]['hands'] > 0:
                        recent = stats[name]['rewards'][-500:] if stats[name]['rewards'] else [0]
                        wr = stats[name]['wins'] / stats[name]['hands']
                        print(f"  {name:12s}: Avg {np.mean(recent):+.2f} BB | Win {wr:.1%} | Hands: {stats[name]['hands']:,}")
                print()
    
    # Complete
    total_time = time.time() - start_time
    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Total hands: {total_hands:,}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Average speed: {total_hands/total_time:.1f} hands/second")
    print()
    
    return agent, stats, all_rewards, all_shaped


if __name__ == "__main__":
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CPU cores: {mp.cpu_count()}, using {NUM_WORKERS} workers")
    print(f"Logging every {LOG_PERCENT}%, checkpoints every {CHECKPOINT_PERCENT}%")
    print()
    
    # Train for 50,000 hands first to verify fix
    agent, stats, rewards, shaped = train_v12_parallel(num_hands=50000)
    
    # Save the model
    torch.save(agent.model.state_dict(), "poker_agent_v12_fixed.pth")
    print("\nModel saved to poker_agent_v12_fixed.pth")
