import sys
import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from collections import deque
import logging

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src', 'models'))
sys.path.insert(0, os.path.join(project_root, 'src', 'workers'))

from poker_worker_v20 import run_training_batch_v20, POSITION_NAMES
from poker_model_v20 import (
    DuelingPokerNet, PrioritizedReplayBuffer, NUM_ACTIONS_V20 as NUM_ACTIONS, OpponentPool
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Configurable Training Parameters ──
NUM_WORKERS = 8
BATCHES_PER_CYCLE = 8
STATE_DIM = 520
EQUITY_ITERATIONS = 512
HANDS_PER_BATCH = 32
EPSILON_RATIO = 0.05

OPPONENT_POOL_SIZE = 20
SNAPSHOT_INTERVAL_EARLY = 20  # Reduced from 100
SNAPSHOT_INTERVAL_LATE = 50    # Reduced from 500
SNAPSHOT_PHASE_THRESHOLD = 500 # Reduced from 2000


# ── Chart styling ──
DARK_BG = '#1e1e2e'
DARK_FG = '#cdd6f4'
DARK_GRID = '#45475a'
ACCENT_COLORS = ['#f38ba8', '#a6e3a1', '#89b4fa', '#fab387', '#cba6f7', '#94e2d5', '#f9e2af']
ACTION_NAMES = ['Fold', 'Check', 'Call', '33%', '66%', '100%', 'All-In']

# ── Shared state between training thread and GUI ──
gui_state = {
    'batch': 0,
    'total_batches': 10000,
    'epsilon': 1.0,
    'loss': 0.0,
    'bb_100': 0.0,
    'status': 'Idle',
    'speed': 0.0,
    'eta': 0.0,
    'pool_size': 0,
    'total_hands': 0,
    'buffer_size': 0,
    'buffer_cap': 300000,
    'action_pcts': {i: 0.0 for i in range(NUM_ACTIONS)},
    'position_stats': {pos: {'hands': 0, 'avg': 0.0, 'win': 0.0} for pos in range(6)},
    'vpip': 0.0,
    'pfr': 0.0,
    'three_bet': 0.0,
    'perf_stats': {
        'total_time': 0.0,
        'sim_time': 0.0,
        'nn_time': 0.0,
        'mc_equity_time': 0.0,
        'overhead_time': 0.0,
    },
    # Time-series histories for charts
    'loss_history': [],
    'bb100_history': [],
    'epsilon_history': [],
    'action_history': {i: [] for i in range(NUM_ACTIONS)},
    'position_bb100_history': {pos: [] for pos in range(6)},
    'hands_history': [],
}

# ── Training control flags (thread-safe via GIL for simple bools/ints) ──
training_control = {
    'paused': False,
    'stop': False,
    'running': False,
    'hands_per_batch': HANDS_PER_BATCH,
    'target_batches': 10000,
    'agent': None,  # Will hold the DuelingAgent reference
}


class DuelingAgent:
    def __init__(self, state_dim=520, hidden_dim=512):
        self.device = device
        self.policy_net = DuelingPokerNet(state_dim, hidden_dim).to(device)
        self.target_net = DuelingPokerNet(state_dim, hidden_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=3e-4)
        self.buffer = PrioritizedReplayBuffer(capacity=300000)
        self.opponent_pool = OpponentPool(max_size=OPPONENT_POOL_SIZE, device='cpu')
        self.gamma = 0.99
        self.batch_size = 256
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.999

    def get_model_state_dict_cpu(self):
        return {k: v.cpu() for k, v in self.policy_net.state_dict().items()}

    def save_snapshot_to_pool(self, training_step: int):
        self.opponent_pool.add_snapshot(self.policy_net, training_step)

    def get_opponent_pool_state_dicts(self):
        return [m.state_dict() for m in self.opponent_pool.models]

    def add_transitions(self, transitions):
        self.buffer.push_batch(transitions)

    def train_step(self) -> float:
        if len(self.buffer) < self.batch_size:
            return 0.0
        samples, indices, weights = self.buffer.sample(self.batch_size)
        if not samples:
            return 0.0

        states = torch.FloatTensor(np.array([s[0] for s in samples])).to(self.device)
        actions = torch.LongTensor(np.array([s[1] for s in samples])).to(self.device)
        rewards = torch.FloatTensor(np.array([s[2] for s in samples])).to(self.device)
        next_states = torch.FloatTensor(np.array([s[3] for s in samples])).to(self.device)
        dones = torch.FloatTensor(np.array([float(s[4]) for s in samples])).to(self.device)
        weights = torch.FloatTensor(np.array(weights)).to(self.device)

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1)
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            targets = rewards + self.gamma * next_q * (1 - dones)

        td_errors = q_values - targets
        loss = (weights * td_errors.pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        self.buffer.update_priorities(indices, td_errors.detach().cpu().numpy())
        return loss.item()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# ═══════════════════════════════════════════════════════════════
#  Training Thread
# ═══════════════════════════════════════════════════════════════

def training_thread_func():
    global gui_state, training_control
    ctrl = training_control
    num_batches = ctrl['target_batches']

    gui_state['total_batches'] = num_batches
    gui_state['status'] = 'Training...'
    ctrl['running'] = True

    agent = DuelingAgent(state_dim=STATE_DIM)
    ctrl['agent'] = agent

    decay_cycles = int(num_batches * EPSILON_RATIO)
    if decay_cycles > 0:
        agent.epsilon_decay = (agent.epsilon_min / agent.epsilon) ** (1.0 / decay_cycles)

    all_hand_profits = deque(maxlen=20000)
    total_batches = 0
    total_hands = 0
    total_vpip = 0
    total_pfr = 0
    total_3bet = 0
    
    agg_perf_stats = {
        'total_time': 0.0,
        'sim_time': 0.0,
        'nn_time': 0.0,
        'mc_equity_time': 0.0,
        'overhead_time': 0.0,
    }
    
    next_snapshot = SNAPSHOT_INTERVAL_EARLY

    action_window = deque(maxlen=40)
    position_stats = {pos: {'profits': [], 'wins': 0, 'hands': 0} for pos in range(6)}

    start_time = time.time()
    agent.save_snapshot_to_pool(training_step=0)

    try:
        ctx = mp.get_context('fork')
    except Exception:
        ctx = mp.get_context('spawn')

    with ProcessPoolExecutor(max_workers=NUM_WORKERS, mp_context=ctx) as executor:
        while total_batches < ctrl['target_batches']:
            # Check pause
            while ctrl['paused'] and not ctrl['stop']:
                gui_state['status'] = 'Paused'
                time.sleep(0.2)
            if ctrl['stop']:
                break

            gui_state['status'] = 'Training...'

            # Read live-adjustable params
            hands_per_batch = ctrl['hands_per_batch']

            model_state_dict = agent.get_model_state_dict_cpu()
            opponent_pool = agent.get_opponent_pool_state_dicts()

            batch_args = []
            for i in range(BATCHES_PER_CYCLE):
                seed = random.randint(0, 2**31 - 1)
                sn = total_batches + i + 1
                batch_args.append((
                    seed, agent.epsilon, EQUITY_ITERATIONS, model_state_dict,
                    hands_per_batch, opponent_pool, sn
                ))

            futures = [executor.submit(run_training_batch_v20, args) for args in batch_args]
            all_cycle_transitions = []
            cycle_action_counts = {i: 0 for i in range(NUM_ACTIONS)}

            for future in as_completed(futures):
                try:
                    res = future.result()
                    all_cycle_transitions.extend(res['transitions'])

                    completed_h = res.get('hands_completed', 0)
                    total_hands += completed_h

                    for a, c in res.get('action_counts', {}).items():
                        cycle_action_counts[a] += c

                    hand_profits = res.get('hand_profits', [])
                    for i, pos in enumerate(res.get('position_history', [])):
                        if i < len(hand_profits):
                            profit = hand_profits[i]
                            all_hand_profits.append(profit)
                            position_stats[pos]['profits'].append(profit)
                            position_stats[pos]['hands'] += 1
                            if profit > 0:
                                position_stats[pos]['wins'] += 1

                    total_vpip += res.get('vpip_count', 0)
                    total_pfr += res.get('pfr_count', 0)
                    total_3bet += res.get('three_bet_count', 0)
                    
                    batch_perf = res.get('perf_stats', {})
                    for k in agg_perf_stats:
                        agg_perf_stats[k] += batch_perf.get(k, 0.0)

                    total_batches += 1
                except Exception as e:
                    logging.error(f"Worker error: {e}")

            action_window.append(cycle_action_counts)
            agent.add_transitions(all_cycle_transitions)
            loss = sum(agent.train_step() for _ in range(8)) / 8.0
            agent.decay_epsilon()

            if total_batches % 100 < BATCHES_PER_CYCLE:
                agent.update_target()

            if total_batches >= next_snapshot:
                agent.save_snapshot_to_pool(training_step=total_batches)
                if total_batches >= SNAPSHOT_PHASE_THRESHOLD:
                    next_snapshot += SNAPSHOT_INTERVAL_LATE
                else:
                    next_snapshot += SNAPSHOT_INTERVAL_EARLY

            # ── Compute stats for GUI ──
            recent_p = float(np.mean(all_hand_profits)) if all_hand_profits else 0.0
            bb_100 = recent_p * 100

            elapsed = time.time() - start_time
            speed = total_batches / elapsed if elapsed > 0 else 0
            remaining = ((ctrl['target_batches'] - total_batches) / speed / 60) if speed > 0 else 0

            recent_actions_agg = {i: sum(w.get(i, 0) for w in action_window) for i in range(NUM_ACTIONS)}
            tot_acts = sum(recent_actions_agg.values())
            action_pcts = {a: recent_actions_agg[a] / tot_acts * 100 if tot_acts > 0 else 0
                           for a in range(NUM_ACTIONS)}

            p_stats_gui = {}
            for pos in range(6):
                h = position_stats[pos]['hands']
                profits_list = list(position_stats[pos]['profits'])
                p_stats_gui[pos] = {
                    'hands': h,
                    'avg': (float(np.mean(profits_list)) * 100) if profits_list else 0.0,
                    'win': (position_stats[pos]['wins'] / h * 100) if h > 0 else 0.0,
                }

            # Push to gui_state
            gui_state['batch'] = total_batches
            gui_state['total_batches'] = ctrl['target_batches']
            gui_state['loss'] = loss
            gui_state['epsilon'] = agent.epsilon
            gui_state['bb_100'] = bb_100
            gui_state['speed'] = speed
            gui_state['eta'] = remaining
            gui_state['pool_size'] = len(agent.opponent_pool)
            gui_state['total_hands'] = total_hands
            gui_state['buffer_size'] = len(agent.buffer)
            gui_state['action_pcts'] = action_pcts
            gui_state['position_stats'] = p_stats_gui
            gui_state['vpip'] = (total_vpip / total_hands * 100) if total_hands > 0 else 0.0
            gui_state['pfr'] = (total_pfr / total_hands * 100) if total_hands > 0 else 0.0
            gui_state['three_bet'] = (total_3bet / total_hands * 100) if total_hands > 0 else 0.0
            
            gui_state['perf_stats'] = {
                k: (v / total_hands) * 1000 if total_hands > 0 else 0.0 # Save as ms per hand
                for k, v in agg_perf_stats.items()
            }

            # Time-series
            gui_state['loss_history'].append(loss)
            gui_state['bb100_history'].append(bb_100)
            gui_state['epsilon_history'].append(agent.epsilon)
            gui_state['hands_history'].append(total_hands)
            for a in range(NUM_ACTIONS):
                gui_state['action_history'][a].append(action_pcts.get(a, 0.0))
            for pos in range(6):
                gui_state['position_bb100_history'][pos].append(p_stats_gui[pos]['avg'])

    gui_state['status'] = 'Done'
    ctrl['running'] = False


# ═══════════════════════════════════════════════════════════════
#  Evaluation
# ═══════════════════════════════════════════════════════════════

def run_evaluation(agent, num_hands=200, opponent_type='random'):
    """Evaluate the trained model against scripted opponents."""
    if agent is None:
        return None

    model_state_dict = agent.get_model_state_dict_cpu()
    results = {
        'hands': 0, 'total_profit_bb': 0.0, 'wins': 0,
        'action_counts': {i: 0 for i in range(NUM_ACTIONS)},
    }

    try:
        ctx = mp.get_context('fork')
    except Exception:
        ctx = mp.get_context('spawn')

    # Run evaluation hands in batches
    batch_size = 50
    batches = max(1, num_hands // batch_size)
    
    total_vpip = 0
    total_pfr = 0
    total_3bet = 0

    with ProcessPoolExecutor(max_workers=NUM_WORKERS, mp_context=ctx) as executor:
        futures = []
        for b in range(batches):
            seed = random.randint(0, 2**31 - 1)
            args = (seed, 0.0, EQUITY_ITERATIONS, model_state_dict,
                    batch_size, [], b + 1)  # epsilon=0 for greedy
            futures.append(executor.submit(run_training_batch_v20, args))

        for future in as_completed(futures):
            try:
                res = future.result()
                results['hands'] += res.get('hands_completed', 0)
                for p in res.get('hand_profits', []):
                    results['total_profit_bb'] += p
                    if p > 0:
                        results['wins'] += 1
                for a, c in res.get('action_counts', {}).items():
                    results['action_counts'][a] += c
                total_vpip += res.get('vpip_count', 0)
                total_pfr += res.get('pfr_count', 0)
                total_3bet += res.get('three_bet_count', 0)
            except Exception as e:
                logging.error(f"Eval error: {e}")

    if results['hands'] > 0:
        results['bb_100'] = (results['total_profit_bb'] / results['hands']) * 100
        results['win_rate'] = results['wins'] / results['hands'] * 100
        results['vpip'] = (total_vpip / results['hands']) * 100
        results['pfr'] = (total_pfr / results['hands']) * 100
        results['three_bet'] = (total_3bet / results['hands']) * 100
    else:
        results['bb_100'] = 0.0
        results['win_rate'] = 0.0
        results['vpip'] = 0.0
        results['pfr'] = 0.0
        results['three_bet'] = 0.0

    return results


# ═══════════════════════════════════════════════════════════════
#  GUI
# ═══════════════════════════════════════════════════════════════

class TrainingGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Poker Agent V20 — Training Dashboard")
        self.geometry("1400x900")
        self.configure(bg=DARK_BG)
        self.train_thread = None

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('.', background=DARK_BG, foreground=DARK_FG, fieldbackground='#313244')
        style.configure('TLabel', background=DARK_BG, foreground=DARK_FG)
        style.configure('TFrame', background=DARK_BG)
        style.configure('TLabelframe', background=DARK_BG, foreground='#89b4fa')
        style.configure('TLabelframe.Label', background=DARK_BG, foreground='#89b4fa')
        style.configure('TNotebook', background=DARK_BG)
        style.configure('TNotebook.Tab', background='#313244', foreground=DARK_FG, padding=[12, 4])
        style.map('TNotebook.Tab', background=[('selected', '#45475a')])
        style.configure('Accent.TButton', background='#89b4fa', foreground='#1e1e2e', padding=[10, 4])
        style.configure('TProgressbar', troughcolor='#313244', background='#a6e3a1')
        style.configure('TCheckbutton', background=DARK_BG, foreground=DARK_FG)
        style.configure('TSpinbox', fieldbackground='#313244', foreground=DARK_FG)

        # ── Notebook (tabs) ──
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.training_tab = ttk.Frame(self.notebook)
        self.perf_tab = ttk.Frame(self.notebook)
        self.eval_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.training_tab, text='  Training  ')
        self.notebook.add(self.perf_tab, text='  Performance  ')
        self.notebook.add(self.eval_tab, text='  Evaluation  ')

        self._build_training_tab()
        self._build_perf_tab()
        self._build_eval_tab()
        self.update_gui()

    # ──────────────────────────────────────────────────
    #  Training Tab
    # ──────────────────────────────────────────────────
    def _build_training_tab(self):
        tab = self.training_tab

        # ── Top status bar ──
        top = ttk.Frame(tab)
        top.pack(fill=tk.X, padx=10, pady=(10, 5))

        self.lbl_status = ttk.Label(top, text="Status: Idle", font=("Segoe UI", 13, "bold"))
        self.lbl_status.pack(side=tk.LEFT)

        self.lbl_speed = ttk.Label(top, text="Speed: — | ETA: —", font=("Segoe UI", 11))
        self.lbl_speed.pack(side=tk.RIGHT)

        # Progress bar
        prog_frame = ttk.Frame(tab)
        prog_frame.pack(fill=tk.X, padx=10, pady=(0, 5))
        self.prog_bar = ttk.Progressbar(prog_frame, orient=tk.HORIZONTAL, mode='determinate')
        self.prog_bar.pack(fill=tk.X)
        self.lbl_progress = ttk.Label(prog_frame, text="0 / 10,000 batches", font=("Segoe UI", 10))
        self.lbl_progress.pack(anchor=tk.W)

        # ── Middle: Charts (2x2 grid) ──
        chart_frame = ttk.Frame(tab)
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.fig = Figure(figsize=(12, 5), dpi=100, facecolor=DARK_BG)
        self.fig.subplots_adjust(hspace=0.45, wspace=0.3)

        self.ax_bb100 = self.fig.add_subplot(2, 2, 1)
        self.ax_loss = self.fig.add_subplot(2, 2, 2)
        self.ax_actions = self.fig.add_subplot(2, 2, 3)
        self.ax_positions = self.fig.add_subplot(2, 2, 4)

        for ax in [self.ax_bb100, self.ax_loss, self.ax_actions, self.ax_positions]:
            ax.set_facecolor('#181825')
            ax.tick_params(colors=DARK_FG, labelsize=8)
            for spine in ax.spines.values():
                spine.set_color(DARK_GRID)
            ax.grid(True, color=DARK_GRID, alpha=0.3, linewidth=0.5)

        self.ax_bb100.set_title('BB/100 Over Time', color=DARK_FG, fontsize=10)
        self.ax_loss.set_title('Training Loss', color=DARK_FG, fontsize=10)
        self.ax_actions.set_title('Action Distribution', color=DARK_FG, fontsize=10)
        self.ax_positions.set_title('Position Performance (BB/100)', color=DARK_FG, fontsize=10)

        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # ── Bottom: Stats + Controls ──
        bottom = ttk.Frame(tab)
        bottom.pack(fill=tk.X, padx=10, pady=(5, 10))

        # Stats row 1
        stats1 = ttk.Frame(bottom)
        stats1.pack(fill=tk.X, pady=2)

        self.lbl_eps = ttk.Label(stats1, text="ε: 1.000", font=("Consolas", 11))
        self.lbl_eps.pack(side=tk.LEFT, padx=(0, 20))
        self.lbl_loss_val = ttk.Label(stats1, text="Loss: 0.000", font=("Consolas", 11))
        self.lbl_loss_val.pack(side=tk.LEFT, padx=(0, 20))
        self.lbl_bb100_val = ttk.Label(stats1, text="BB/100: +0.0", font=("Consolas", 11))
        self.lbl_bb100_val.pack(side=tk.LEFT, padx=(0, 20))
        self.lbl_buffer = ttk.Label(stats1, text="Buffer: 0/300k", font=("Consolas", 11))
        self.lbl_buffer.pack(side=tk.LEFT, padx=(0, 20))
        self.lbl_hands = ttk.Label(stats1, text="Hands: 0", font=("Consolas", 11))
        self.lbl_hands.pack(side=tk.LEFT, padx=(0, 20))
        self.lbl_pool = ttk.Label(stats1, text="Pool: 0", font=("Consolas", 11))
        self.lbl_pool.pack(side=tk.LEFT)

        # Stats row 2: action distribution text
        stats2 = ttk.Frame(bottom)
        stats2.pack(fill=tk.X, pady=2)

        self.action_labels = {}
        for i, name in enumerate(ACTION_NAMES):
            lbl = ttk.Label(stats2, text=f"{name}: 0.0%", font=("Consolas", 10),
                            foreground=ACCENT_COLORS[i % len(ACCENT_COLORS)])
            lbl.pack(side=tk.LEFT, padx=(0, 12))
            self.action_labels[i] = lbl
            
        # VPIP, PFR, 3-Bet Labels
        # Separate frame packed to the right side of stats2
        self.lbl_vpip = ttk.Label(stats2, text="VPIP: 0.0%", font=("Consolas", 10, "bold"), foreground='#f9e2af')
        self.lbl_vpip.pack(side=tk.RIGHT, padx=(12, 0))
        self.lbl_pfr = ttk.Label(stats2, text="PFR: 0.0%", font=("Consolas", 10, "bold"), foreground='#f9e2af')
        self.lbl_pfr.pack(side=tk.RIGHT, padx=(12, 0))
        self.lbl_3bet = ttk.Label(stats2, text="3-Bet: 0.0%", font=("Consolas", 10, "bold"), foreground='#f9e2af')
        self.lbl_3bet.pack(side=tk.RIGHT, padx=(12, 0))

        # Stats row 3: position stats
        stats3 = ttk.Frame(bottom)
        stats3.pack(fill=tk.X, pady=2)

        self.pos_labels = {}
        for i, name in enumerate(POSITION_NAMES):
            lbl = ttk.Label(stats3, text=f"{name}: — BB/100", font=("Consolas", 10))
            lbl.pack(side=tk.LEFT, padx=(0, 12))
            self.pos_labels[i] = lbl

        # ── Controls row ──
        ctrl_frame = ttk.LabelFrame(bottom, text="Controls", padding=8)
        ctrl_frame.pack(fill=tk.X, pady=(8, 0))

        # Start / Stop
        self.btn_start = ttk.Button(ctrl_frame, text="▶ Start Training", command=self._start_training)
        self.btn_start.pack(side=tk.LEFT, padx=4)

        self.btn_pause = ttk.Button(ctrl_frame, text="⏸ Pause", command=self._toggle_pause, state=tk.DISABLED)
        self.btn_pause.pack(side=tk.LEFT, padx=4)

        self.btn_stop = ttk.Button(ctrl_frame, text="⏹ Stop", command=self._stop_training, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=4)

        # Target batches
        ttk.Label(ctrl_frame, text="  Target Batches:").pack(side=tk.LEFT, padx=(12, 2))
        self.spin_batches = tk.Spinbox(ctrl_frame, from_=100, to=999999, width=8,
                                       bg='#313244', fg=DARK_FG, buttonbackground='#45475a',
                                       insertbackground=DARK_FG)
        self.spin_batches.delete(0, tk.END)
        self.spin_batches.insert(0, '10000')
        self.spin_batches.pack(side=tk.LEFT, padx=2)

        self.btn_extend = ttk.Button(ctrl_frame, text="+ Extend", command=self._extend_training)
        self.btn_extend.pack(side=tk.LEFT, padx=4)

        # Hands per batch
        ttk.Label(ctrl_frame, text="  Hands/Batch:").pack(side=tk.LEFT, padx=(12, 2))
        self.spin_hands = tk.Spinbox(ctrl_frame, from_=1, to=512, width=5,
                                      bg='#313244', fg=DARK_FG, buttonbackground='#45475a',
                                      insertbackground=DARK_FG)
        self.spin_hands.delete(0, tk.END)
        self.spin_hands.insert(0, str(HANDS_PER_BATCH))
        self.spin_hands.pack(side=tk.LEFT, padx=2)
        self.spin_hands.bind('<Return>', self._update_hands_per_batch)
        self.spin_hands.bind('<FocusOut>', self._update_hands_per_batch)


        # Save
        self.btn_save = ttk.Button(ctrl_frame, text="💾 Save Model", command=self._save_model)
        self.btn_save.pack(side=tk.RIGHT, padx=4)

    # ──────────────────────────────────────────────────
    #  Performance Tab
    # ──────────────────────────────────────────────────
    def _build_perf_tab(self):
        tab = self.perf_tab
        
        top = ttk.Frame(tab)
        top.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(top, text="Training Overhead Profiler", font=("Segoe UI", 14, "bold"), foreground='#a6e3a1').pack(anchor=tk.W)
        ttk.Label(top, text="Average computation time per hand simulated (in milliseconds).", font=("Segoe UI", 10)).pack(anchor=tk.W, pady=(2, 10))
        
        # Grid layout for timing metrics
        grid_frame = ttk.Frame(tab)
        grid_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.perf_labels = {}
        
        metrics = [
            ('total_time', 'Total Time/Hand', '#cdd6f4'),
            ('sim_time', 'Game Simulation', '#f9e2af'),
            ('nn_time', 'NN Inference', '#89b4fa'),
            ('mc_equity_time', 'MC Equity Simulation', '#f38ba8'),
            ('overhead_time', 'Worker Overhead/Misc', '#a6adc8'),
        ]
        
        for i, (key, title, color) in enumerate(metrics):
            row = ttk.Frame(grid_frame)
            row.pack(fill=tk.X, pady=6)
            
            ttk.Label(row, text=f"{title}:", font=("Consolas", 12), width=25).pack(side=tk.LEFT)
            lbl = ttk.Label(row, text="0.00 ms", font=("Consolas", 12, "bold"), foreground=color)
            lbl.pack(side=tk.LEFT, padx=10)
            
            self.perf_labels[key] = lbl

    # ──────────────────────────────────────────────────
    #  Evaluation Tab
    # ──────────────────────────────────────────────────
    def _build_eval_tab(self):
        tab = self.eval_tab

        ctrl = ttk.LabelFrame(tab, text="Evaluation Settings", padding=10)
        ctrl.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(ctrl, text="Evaluation Hands:").pack(side=tk.LEFT, padx=(0, 4))
        self.spin_eval_hands = tk.Spinbox(ctrl, from_=50, to=10000, width=7,
                                           bg='#313244', fg=DARK_FG, buttonbackground='#45475a',
                                           insertbackground=DARK_FG)
        self.spin_eval_hands.delete(0, tk.END)
        self.spin_eval_hands.insert(0, '500')
        self.spin_eval_hands.pack(side=tk.LEFT, padx=4)

        self.btn_eval = ttk.Button(ctrl, text="▶ Run Evaluation", command=self._run_eval)
        self.btn_eval.pack(side=tk.LEFT, padx=12)

        self.lbl_eval_status = ttk.Label(ctrl, text="", font=("Segoe UI", 10, "italic"))
        self.lbl_eval_status.pack(side=tk.LEFT, padx=12)

        # Results frame
        results_frame = ttk.Frame(tab)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # Left: text results
        text_frame = ttk.LabelFrame(results_frame, text="Results", padding=10)
        text_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.eval_text = tk.Text(text_frame, bg='#181825', fg=DARK_FG, font=("Consolas", 11),
                                  wrap=tk.WORD, state=tk.DISABLED, insertbackground=DARK_FG,
                                  relief=tk.FLAT, padx=10, pady=10)
        self.eval_text.pack(fill=tk.BOTH, expand=True)

        # Right: chart
        chart_frame = ttk.LabelFrame(results_frame, text="Action Distribution", padding=10)
        chart_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        self.eval_fig = Figure(figsize=(5, 4), dpi=100, facecolor=DARK_BG)
        self.eval_ax = self.eval_fig.add_subplot(1, 1, 1)
        self.eval_ax.set_facecolor('#181825')
        self.eval_ax.tick_params(colors=DARK_FG, labelsize=9)
        for spine in self.eval_ax.spines.values():
            spine.set_color(DARK_GRID)

        self.eval_canvas = FigureCanvasTkAgg(self.eval_fig, master=chart_frame)
        self.eval_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ──────────────────────────────────────────────────
    #  Control Callbacks
    # ──────────────────────────────────────────────────
    def _start_training(self):
        if training_control['running']:
            return

        # Reset histories
        for key in ['loss_history', 'bb100_history', 'epsilon_history', 'hands_history']:
            gui_state[key] = []
        for i in range(NUM_ACTIONS):
            gui_state['action_history'][i] = []
        for pos in range(6):
            gui_state['position_bb100_history'][pos] = []

        try:
            target = int(self.spin_batches.get())
        except ValueError:
            target = 10000

        training_control['target_batches'] = target
        training_control['stop'] = False
        training_control['paused'] = False

        self.btn_start.config(state=tk.DISABLED)
        self.btn_pause.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.NORMAL)

        self.train_thread = threading.Thread(target=training_thread_func, daemon=True)
        self.train_thread.start()

    def _toggle_pause(self):
        training_control['paused'] = not training_control['paused']
        if training_control['paused']:
            self.btn_pause.config(text="▶ Resume")
        else:
            self.btn_pause.config(text="⏸ Pause")

    def _stop_training(self):
        training_control['stop'] = True
        self.btn_start.config(state=tk.NORMAL)
        self.btn_pause.config(state=tk.DISABLED, text="⏸ Pause")
        self.btn_stop.config(state=tk.DISABLED)

    def _extend_training(self):
        try:
            extra = int(self.spin_batches.get())
        except ValueError:
            extra = 1000
        training_control['target_batches'] = gui_state['batch'] + extra
        gui_state['total_batches'] = training_control['target_batches']

        # If training finished, restart it
        if not training_control['running']:
            training_control['stop'] = False
            training_control['paused'] = False
            self.btn_start.config(state=tk.DISABLED)
            self.btn_pause.config(state=tk.NORMAL)
            self.btn_stop.config(state=tk.NORMAL)
            self.train_thread = threading.Thread(target=training_thread_func, daemon=True)
            self.train_thread.start()

    def _update_hands_per_batch(self, event=None):
        try:
            val = int(self.spin_hands.get())
            training_control['hands_per_batch'] = max(1, min(512, val))
        except ValueError:
            pass

    def _save_model(self):
        agent = training_control.get('agent')
        if agent is None:
            messagebox.showwarning("Save", "No trained model to save.")
            return
        save_path = os.path.join(project_root, 'models', 'poker_agent_v20.pt')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            'policy_net': agent.policy_net.state_dict(),
            'target_net': agent.target_net.state_dict(),
            'optimizer': agent.optimizer.state_dict(),
            'epsilon': agent.epsilon,
        }, save_path)
        messagebox.showinfo("Save", f"Model saved to:\n{save_path}")

    def _run_eval(self):
        agent = training_control.get('agent')
        if agent is None:
            messagebox.showwarning("Evaluate", "No trained model available. Train first.")
            return

        try:
            num_hands = int(self.spin_eval_hands.get())
        except ValueError:
            num_hands = 500

        self.lbl_eval_status.config(text="Running evaluation...")
        self.btn_eval.config(state=tk.DISABLED)
        self.update_idletasks()

        def _eval_thread():
            results = run_evaluation(agent, num_hands=num_hands)
            self.after(0, lambda: self._show_eval_results(results))

        threading.Thread(target=_eval_thread, daemon=True).start()

    def _show_eval_results(self, results):
        self.btn_eval.config(state=tk.NORMAL)
        self.lbl_eval_status.config(text="Done!")

        if results is None:
            return

        # Text results
        self.eval_text.config(state=tk.NORMAL)
        self.eval_text.delete('1.0', tk.END)

        lines = [
            f"{'═' * 40}",
            f"  EVALUATION RESULTS",
            f"{'═' * 40}",
            f"",
            f"  Hands Played:    {results['hands']:,}",
            f"  Total Profit:    {results['total_profit_bb']:+.1f} BB",
            f"  BB/100:          {results['bb_100']:+.1f}",
            f"  Win Rate:        {results['win_rate']:.1f}%",
            f"",
            f"  VPIP:            {results.get('vpip', 0):.1f}%",
            f"  PFR:             {results.get('pfr', 0):.1f}%",
            f"  3-Bet:           {results.get('three_bet', 0):.1f}%",
            f"",
            f"{'─' * 40}",
            f"  ACTION BREAKDOWN",
            f"{'─' * 40}",
        ]

        tot = sum(results['action_counts'].values())
        for i, name in enumerate(ACTION_NAMES):
            count = results['action_counts'].get(i, 0)
            pct = count / tot * 100 if tot > 0 else 0
            lines.append(f"  {name:8s}  {count:5d}  ({pct:5.1f}%)")

        self.eval_text.insert(tk.END, '\n'.join(lines))
        self.eval_text.config(state=tk.DISABLED)

        # Bar chart
        self.eval_ax.clear()
        self.eval_ax.set_facecolor('#181825')
        counts = [results['action_counts'].get(i, 0) for i in range(NUM_ACTIONS)]
        tot = sum(counts)
        pcts = [c / tot * 100 if tot > 0 else 0 for c in counts]
        bars = self.eval_ax.bar(ACTION_NAMES, pcts, color=ACCENT_COLORS[:NUM_ACTIONS], edgecolor='none')
        self.eval_ax.set_ylabel('% of Actions', color=DARK_FG, fontsize=9)
        self.eval_ax.set_title(f'Eval Actions ({results["hands"]:,} hands)', color=DARK_FG, fontsize=10)
        self.eval_ax.tick_params(colors=DARK_FG, labelsize=8)
        for spine in self.eval_ax.spines.values():
            spine.set_color(DARK_GRID)

        for bar, pct in zip(bars, pcts):
            if pct > 2:
                self.eval_ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                                  f'{pct:.1f}%', ha='center', va='bottom', color=DARK_FG, fontsize=8)

        self.eval_fig.tight_layout()
        self.eval_canvas.draw()

    # ──────────────────────────────────────────────────
    #  GUI Update Loop
    # ──────────────────────────────────────────────────
    def update_gui(self):
        gs = gui_state

        # Status bar
        self.lbl_status.config(text=f"Status: {gs['status']}")
        pct = gs['batch'] / max(1, gs['total_batches']) * 100
        self.prog_bar['value'] = pct
        self.lbl_progress.config(text=f"{gs['batch']:,} / {gs['total_batches']:,} batches ({pct:.0f}%)")
        self.lbl_speed.config(text=f"Speed: {gs['speed']:.1f} b/s | ETA: {gs['eta']:.1f}m | Hands: {gs['total_hands']:,}")

        # Stats
        self.lbl_eps.config(text=f"ε: {gs['epsilon']:.4f}")
        self.lbl_loss_val.config(text=f"Loss: {gs['loss']:.4f}")
        self.lbl_bb100_val.config(text=f"BB/100: {gs['bb_100']:+.1f}")
        buf_pct = gs['buffer_size'] / gs['buffer_cap'] * 100 if gs['buffer_cap'] > 0 else 0
        self.lbl_buffer.config(text=f"Buffer: {gs['buffer_size']:,}/{gs['buffer_cap']//1000}k ({buf_pct:.0f}%)")
        self.lbl_hands.config(text=f"Hands: {gs['total_hands']:,}")
        self.lbl_pool.config(text=f"Pool: {gs['pool_size']}")

        # Action labels
        for i, name in enumerate(ACTION_NAMES):
            if i in self.action_labels:
                self.action_labels[i].config(text=f"{name}: {gs['action_pcts'].get(i, 0):.1f}%")
                
        # VPIP, PFR, 3Bet
        self.lbl_vpip.config(text=f"VPIP: {gs.get('vpip', 0.0):.1f}%")
        self.lbl_pfr.config(text=f"PFR: {gs.get('pfr', 0.0):.1f}%")
        self.lbl_3bet.config(text=f"3-Bet: {gs.get('three_bet', 0.0):.1f}%")

        # Position labels
        for i, pname in enumerate(POSITION_NAMES):
            ps = gs['position_stats'].get(i, {})
            h = ps.get('hands', 0)
            avg = ps.get('avg', 0)
            win = ps.get('win', 0)
            self.pos_labels[i].config(text=f"{pname}: {h:,}h {avg:+.1f}BB/100 {win:.0f}%W")

        # Re-enable start button when done
        if gs['status'] == 'Done' and not training_control['running']:
            self.btn_start.config(state=tk.NORMAL)
            self.btn_pause.config(state=tk.DISABLED, text="⏸ Pause")
            self.btn_stop.config(state=tk.DISABLED)
            
        # Update Performance Tab
        for key, lbl in self.perf_labels.items():
            val = gs['perf_stats'].get(key, 0.0)
            lbl.config(text=f"{val:.2f} ms")

        # ── Update Charts ──
        self._update_charts(gs)

        self.after(1000, self.update_gui)

    def _update_charts(self, gs):
        # ── BB/100 ──
        self.ax_bb100.clear()
        self.ax_bb100.set_facecolor('#181825')
        self.ax_bb100.grid(True, color=DARK_GRID, alpha=0.3, linewidth=0.5)
        self.ax_bb100.set_title('BB/100 Over Time', color=DARK_FG, fontsize=10)
        if gs['bb100_history']:
            data = gs['bb100_history']
            self.ax_bb100.plot(data, color='#a6e3a1', linewidth=1, alpha=0.4)
            # Smoothed line
            if len(data) > 10:
                window = min(50, len(data) // 3)
                smoothed = np.convolve(data, np.ones(window) / window, mode='valid')
                self.ax_bb100.plot(range(window - 1, len(data)), smoothed, color='#a6e3a1', linewidth=2)
            self.ax_bb100.axhline(y=0, color='#585b70', linewidth=1, linestyle='--')
        self.ax_bb100.tick_params(colors=DARK_FG, labelsize=8)
        for spine in self.ax_bb100.spines.values():
            spine.set_color(DARK_GRID)

        # ── Loss ──
        self.ax_loss.clear()
        self.ax_loss.set_facecolor('#181825')
        self.ax_loss.grid(True, color=DARK_GRID, alpha=0.3, linewidth=0.5)
        self.ax_loss.set_title('Training Loss', color=DARK_FG, fontsize=10)
        if gs['loss_history']:
            data = gs['loss_history']
            self.ax_loss.plot(data, color='#f38ba8', linewidth=1, alpha=0.4)
            if len(data) > 10:
                window = min(50, len(data) // 3)
                smoothed = np.convolve(data, np.ones(window) / window, mode='valid')
                self.ax_loss.plot(range(window - 1, len(data)), smoothed, color='#f38ba8', linewidth=2)
        self.ax_loss.tick_params(colors=DARK_FG, labelsize=8)
        for spine in self.ax_loss.spines.values():
            spine.set_color(DARK_GRID)

        # ── Action Distribution (stacked area) ──
        self.ax_actions.clear()
        self.ax_actions.set_facecolor('#181825')
        self.ax_actions.grid(True, color=DARK_GRID, alpha=0.3, linewidth=0.5)
        self.ax_actions.set_title('Action Distribution', color=DARK_FG, fontsize=10)
        hist = gs['action_history']
        if hist[0]:
            n = len(hist[0])
            x = range(n)
            stacks = [hist[i] for i in range(NUM_ACTIONS)]
            self.ax_actions.stackplot(x, *stacks, colors=ACCENT_COLORS[:NUM_ACTIONS],
                                      labels=ACTION_NAMES, alpha=0.8)
            if n < 100:  # Only show legend when not too crowded
                self.ax_actions.legend(loc='upper right', fontsize=6, framealpha=0.5,
                                       facecolor=DARK_BG, edgecolor=DARK_GRID, labelcolor=DARK_FG)
        self.ax_actions.tick_params(colors=DARK_FG, labelsize=8)
        for spine in self.ax_actions.spines.values():
            spine.set_color(DARK_GRID)

        # ── Position Performance (bar chart) ──
        self.ax_positions.clear()
        self.ax_positions.set_facecolor('#181825')
        self.ax_positions.grid(True, color=DARK_GRID, alpha=0.3, linewidth=0.5, axis='y')
        self.ax_positions.set_title('Position Performance (BB/100)', color=DARK_FG, fontsize=10)
        pos_vals = [gs['position_stats'].get(i, {}).get('avg', 0) for i in range(6)]
        colors = ['#a6e3a1' if v >= 0 else '#f38ba8' for v in pos_vals]
        bars = self.ax_positions.bar(POSITION_NAMES, pos_vals, color=colors, edgecolor='none', alpha=0.85)
        self.ax_positions.axhline(y=0, color='#585b70', linewidth=1, linestyle='--')
        for bar, val in zip(bars, pos_vals):
            self.ax_positions.text(bar.get_x() + bar.get_width() / 2,
                                    bar.get_height() + (0.5 if val >= 0 else -1.5),
                                    f'{val:+.1f}', ha='center', va='bottom' if val >= 0 else 'top',
                                    color=DARK_FG, fontsize=8)
        self.ax_positions.tick_params(colors=DARK_FG, labelsize=8)
        for spine in self.ax_positions.spines.values():
            spine.set_color(DARK_GRID)

        self.fig.tight_layout(pad=1.5)
        self.canvas.draw_idle()


if __name__ == '__main__':
    mp.freeze_support()
    app = TrainingGUI()
    app.mainloop()
