import os
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Optional

import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.dirname(CURRENT_DIR)
TRAINERS_DIR = os.path.join(SRC_ROOT, "trainers")
if TRAINERS_DIR not in sys.path:
    sys.path.insert(0, TRAINERS_DIR)

from poker_trainer_v22 import DeepCFRTrainerV21

MATPLOTLIB_AVAILABLE = False
MATPLOTLIB_ERROR = None
Figure = None
FigureCanvasTkAgg = None
try:
    import matplotlib

    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg as _FigureCanvasTkAgg
    from matplotlib.figure import Figure as _Figure

    Figure = _Figure
    FigureCanvasTkAgg = _FigureCanvasTkAgg
    MATPLOTLIB_AVAILABLE = True
except Exception as exc:  # pragma: no cover
    MATPLOTLIB_ERROR = exc

IMAGE_CAPTURE_AVAILABLE = False
ImageGrab = None
try:
    from PIL import ImageGrab as _ImageGrab

    ImageGrab = _ImageGrab
    IMAGE_CAPTURE_AVAILABLE = True
except Exception:
    IMAGE_CAPTURE_AVAILABLE = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_VERSION = "v22"
MODEL_LABEL = MODEL_VERSION.upper()
DEFAULT_CHECKPOINT_FILENAME = f"poker_agent_{MODEL_VERSION}_deepcfr.pt"
DEFAULT_CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "models", DEFAULT_CHECKPOINT_FILENAME)
DEFAULT_RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

DARK_BG = "#1e1e2e"
DARK_FG = "#cdd6f4"
DARK_GRID = "#45475a"
ACCENT_COLORS = ["#f38ba8", "#a6e3a1", "#89b4fa", "#fab387", "#cba6f7", "#94e2d5", "#f9e2af"]
ACTION_NAMES = ["Fold", "Check", "Call", "Raise 1/2", "Raise Pot"]
POSITION_NAMES = ["SB", "BB", "UTG", "MP", "CO", "BTN"]
RANK_LABELS = list("AKQJT98765432")
EVAL_SCOPE_LABELS = ["Overall"] + POSITION_NAMES
EVAL_METRIC_LABELS = ["VPIP", "PFR", "3-Bet"]
EVAL_METRIC_KEY_BY_LABEL = {
    "VPIP": "vpip",
    "PFR": "pfr",
    "3-Bet": "three_bet",
}
EVAL_METRIC_LABEL_BY_KEY = {
    "vpip": "VPIP",
    "pfr": "PFR",
    "three_bet": "3-Bet",
}
EVAL_METRIC_POSITION_ATTR = {
    "vpip": "vpip_by_position",
    "pfr": "pfr_by_position",
    "three_bet": "three_bet_by_position",
}
EVAL_METRIC_GRID_ATTR = {
    "vpip": "vpip_hand_grid",
    "pfr": "pfr_hand_grid",
    "three_bet": "three_bet_hand_grid",
}
EVAL_METRIC_GRID_BY_POSITION_ATTR = {
    "vpip": "vpip_hand_grid_by_position",
    "pfr": "pfr_hand_grid_by_position",
    "three_bet": "three_bet_hand_grid_by_position",
}


class TrainingGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"Poker Agent {MODEL_LABEL} - Training Dashboard")
        self.geometry("1400x900")
        self.configure(bg=DARK_BG)
        self.train_thread: Optional[threading.Thread] = None

        self.trainer = DeepCFRTrainerV21()
        self.training_running = False
        self.stop_requested = False
        self.pause_requested = False
        self.training_start_time: Optional[float] = None
        self._last_eval_report = None

        self.gui_state = {
            "batch": 0,
            "total_batches": 524_288,
            "epsilon": 0.0,
            "loss": 0.0,
            "aux_loss": 0.0,
            "bb_100": 0.0,
            "bb_per_hand": 0.0,
            "bb100_window": 0,
            "style_window": 0,
            "position_window": 0,
            "status": "Idle",
            "speed": 0.0,
            "eta": 0.0,
            "pool_size": 0,
            "learner_steps": 0,
            "total_hands": 0,
            "buffer_size": 0,
            "buffer_cap": 1000000,
            "buffer_size_aux": 0,
            "buffer_cap_aux": 2000000,
            "action_pcts": {i: 0.0 for i in range(len(ACTION_NAMES))},
            "position_stats": {i: {"hands": 0, "avg": 0.0, "win": 0.0} for i in range(6)},
            "vpip": 0.0,
            "pfr": 0.0,
            "three_bet": 0.0,
            "state_err": 0.0,
            "fallbacks": 0.0,
            "perf_stats": {
                "total_time": 0.0,
                "sim_time": 0.0,
                "nn_time": 0.0,
                "mc_equity_time": 0.0,
                "overhead_time": 0.0,
            },
            "loss_history": [],
            "aux_loss_history": [],
            "bb100_history": [],
            "epsilon_history": [],
            "action_history": {i: [] for i in range(len(ACTION_NAMES))},
            "position_bb100_history": {i: [] for i in range(6)},
            "hands_history": [],
        }

        style = ttk.Style()
        style.theme_use("clam")
        style.configure(".", background=DARK_BG, foreground=DARK_FG, fieldbackground="#313244")
        style.configure("TLabel", background=DARK_BG, foreground=DARK_FG)
        style.configure("TFrame", background=DARK_BG)
        style.configure("TLabelframe", background=DARK_BG, foreground="#89b4fa")
        style.configure("TLabelframe.Label", background=DARK_BG, foreground="#89b4fa")
        style.configure("TNotebook", background=DARK_BG)
        style.configure("TNotebook.Tab", background="#313244", foreground=DARK_FG, padding=[12, 4])
        style.map("TNotebook.Tab", background=[("selected", "#45475a")])
        style.configure("Accent.TButton", background="#89b4fa", foreground="#1e1e2e", padding=[10, 4])
        style.configure("TProgressbar", troughcolor="#313244", background="#a6e3a1")
        style.configure("TCheckbutton", background=DARK_BG, foreground=DARK_FG)
        style.configure("TSpinbox", fieldbackground="#313244", foreground=DARK_FG)

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.training_tab = ttk.Frame(self.notebook)
        self.perf_tab = ttk.Frame(self.notebook)
        self.eval_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.training_tab, text="  Training  ")
        self.notebook.add(self.perf_tab, text="  Performance  ")
        self.notebook.add(self.eval_tab, text="  Evaluation  ")

        self._build_training_tab()
        self._build_perf_tab()
        self._build_eval_tab()
        self.update_gui()

    def _build_training_tab(self):
        tab = self.training_tab

        top = ttk.Frame(tab)
        top.pack(fill=tk.X, padx=10, pady=(10, 5))

        self.lbl_status = ttk.Label(top, text="Status: Idle", font=("Segoe UI", 13, "bold"))
        self.lbl_status.pack(side=tk.LEFT)

        self.lbl_phase = ttk.Label(top, text="Learner: Waiting", font=("Segoe UI", 10), foreground="#a6adc8")
        self.lbl_phase.pack(side=tk.LEFT, padx=(16, 0))

        self.lbl_speed = ttk.Label(top, text="Speed: - | ETA: -", font=("Segoe UI", 11))
        self.lbl_speed.pack(side=tk.RIGHT)

        prog_frame = ttk.Frame(tab)
        prog_frame.pack(fill=tk.X, padx=10, pady=(0, 5))
        self.prog_bar = ttk.Progressbar(prog_frame, orient=tk.HORIZONTAL, mode="determinate")
        self.prog_bar.pack(fill=tk.X)
        self.lbl_progress = ttk.Label(prog_frame, text="0 / 524,288 traversals", font=("Segoe UI", 10))
        self.lbl_progress.pack(anchor=tk.W)

        chart_frame = ttk.Frame(tab)
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        if MATPLOTLIB_AVAILABLE:
            self.fig = Figure(figsize=(12, 5), dpi=100, facecolor=DARK_BG)
            self.fig.subplots_adjust(hspace=0.45, wspace=0.3)

            self.ax_bb100 = self.fig.add_subplot(2, 2, 1)
            self.ax_loss = self.fig.add_subplot(2, 2, 2)
            self.ax_actions = self.fig.add_subplot(2, 2, 3)
            self.ax_positions = self.fig.add_subplot(2, 2, 4)

            for ax in [self.ax_bb100, self.ax_loss, self.ax_actions, self.ax_positions]:
                ax.set_facecolor("#181825")
                ax.tick_params(colors=DARK_FG, labelsize=8)
                for spine in ax.spines.values():
                    spine.set_color(DARK_GRID)
                ax.grid(True, color=DARK_GRID, alpha=0.3, linewidth=0.5)

            self.ax_bb100.set_title("BB/100 Over Time", color=DARK_FG, fontsize=10)
            self.ax_loss.set_title("Training Loss", color=DARK_FG, fontsize=10)
            self.ax_actions.set_title("Action Distribution", color=DARK_FG, fontsize=10)
            self.ax_positions.set_title("Position Performance (BB/100)", color=DARK_FG, fontsize=10)

            self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        else:
            self.fig = None
            self.ax_bb100 = None
            self.ax_loss = None
            self.ax_actions = None
            self.ax_positions = None
            self.canvas = None
            self.tk_chart_canvases = {}
            grid = ttk.Frame(chart_frame)
            grid.pack(fill=tk.BOTH, expand=True)
            specs = [
                ("bb100", "BB/100 Over Time"),
                ("loss", "Training Loss"),
                ("actions", "Action Distribution"),
                ("positions", "Position Performance (BB/100)"),
            ]
            for idx, (key, title) in enumerate(specs):
                cell = ttk.Frame(grid)
                cell.grid(row=idx // 2, column=idx % 2, sticky="nsew", padx=4, pady=4)
                grid.grid_rowconfigure(idx // 2, weight=1)
                grid.grid_columnconfigure(idx % 2, weight=1)
                ttk.Label(cell, text=title, font=("Segoe UI", 10)).pack(anchor=tk.W)
                canvas = tk.Canvas(cell, bg="#181825", highlightthickness=1, highlightbackground=DARK_GRID, height=220)
                canvas.pack(fill=tk.BOTH, expand=True)
                self.tk_chart_canvases[key] = canvas

        bottom = ttk.Frame(tab)
        bottom.pack(fill=tk.X, padx=10, pady=(5, 10))

        stats1 = ttk.Frame(bottom)
        stats1.pack(fill=tk.X, pady=2)

        self.lbl_eps = ttk.Label(stats1, text="Decisions: 0", font=("Consolas", 11))
        self.lbl_eps.pack(side=tk.LEFT, padx=(0, 20))
        self.lbl_loss_val = ttk.Label(stats1, text="Regret: 0.0000 | Strategy: 0.0000", font=("Consolas", 11))
        self.lbl_loss_val.pack(side=tk.LEFT, padx=(0, 20))
        self.lbl_bb100_val = ttk.Label(stats1, text="BB/hand: +0.000 | BB/100: +0.0", font=("Consolas", 11))
        self.lbl_bb100_val.pack(side=tk.LEFT, padx=(0, 20))
        self.lbl_buffer = ttk.Label(stats1, text="Adv: 0/1000k | Strat: 0/2000k", font=("Consolas", 11))
        self.lbl_buffer.pack(side=tk.LEFT, padx=(0, 20))
        self.lbl_hands = ttk.Label(stats1, text="Traversals: 0", font=("Consolas", 11))
        self.lbl_hands.pack(side=tk.LEFT, padx=(0, 20))
        self.lbl_pool = ttk.Label(stats1, text="Pool: 0", font=("Consolas", 11))
        self.lbl_pool.pack(side=tk.LEFT)

        stats2 = ttk.Frame(bottom)
        stats2.pack(fill=tk.X, pady=2)

        self.action_labels = {}
        for i, name in enumerate(ACTION_NAMES):
            lbl = ttk.Label(
                stats2,
                text=f"{name}: 0.0%",
                font=("Consolas", 10),
                foreground=ACCENT_COLORS[i % len(ACCENT_COLORS)],
            )
            lbl.pack(side=tk.LEFT, padx=(0, 12))
            self.action_labels[i] = lbl

        self.lbl_vpip = ttk.Label(stats2, text="VPIP: 0.0%", font=("Consolas", 10, "bold"), foreground="#f9e2af")
        self.lbl_vpip.pack(side=tk.RIGHT, padx=(12, 0))
        self.lbl_pfr = ttk.Label(stats2, text="PFR: 0.0%", font=("Consolas", 10, "bold"), foreground="#f9e2af")
        self.lbl_pfr.pack(side=tk.RIGHT, padx=(12, 0))
        self.lbl_3bet = ttk.Label(stats2, text="3-Bet: 0.0%", font=("Consolas", 10, "bold"), foreground="#f9e2af")
        self.lbl_3bet.pack(side=tk.RIGHT, padx=(12, 0))

        stats3 = ttk.Frame(bottom)
        stats3.pack(fill=tk.X, pady=2)

        self.pos_labels = {}
        for i, name in enumerate(POSITION_NAMES):
            lbl = ttk.Label(stats3, text=f"{name}: - BB", font=("Consolas", 10))
            lbl.pack(side=tk.LEFT, padx=(0, 12))
            self.pos_labels[i] = lbl

        self.lbl_health = ttk.Label(
            bottom,
            text=(
                "Healthy: BB/hand is the traverser's rolling raw stack delta, BB/100 is that value times 100, "
                "VPIP/PFR/3-Bet are rolling preflop rates, "
                "and State Err/Fallbacks stay at 0."
            ),
            font=("Segoe UI", 9),
            foreground="#a6adc8",
        )
        self.lbl_health.pack(anchor=tk.W, pady=(6, 0))

        ctrl_frame = ttk.LabelFrame(bottom, text="Controls", padding=8)
        ctrl_frame.pack(fill=tk.X, pady=(8, 0))

        self.btn_start = ttk.Button(ctrl_frame, text="Start Training", command=self._start_training)
        self.btn_start.pack(side=tk.LEFT, padx=4)

        self.btn_pause = ttk.Button(ctrl_frame, text="Pause", command=self._toggle_pause, state=tk.DISABLED)
        self.btn_pause.pack(side=tk.LEFT, padx=4)

        self.btn_stop = ttk.Button(ctrl_frame, text="Stop", command=self._stop_training, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=4)

        ttk.Label(ctrl_frame, text="  Target Traversals:").pack(side=tk.LEFT, padx=(12, 2))
        self.spin_batches = tk.Spinbox(
            ctrl_frame,
            from_=100,
            to=99999999,
            width=8,
            bg="#313244",
            fg=DARK_FG,
            buttonbackground="#45475a",
            insertbackground=DARK_FG,
        )
        self.spin_batches.delete(0, tk.END)
        self.spin_batches.insert(0, "524288")
        self.spin_batches.pack(side=tk.LEFT, padx=2)

        self.btn_extend = ttk.Button(ctrl_frame, text="+ Extend", command=self._extend_training)
        self.btn_extend.pack(side=tk.LEFT, padx=4)

        ttk.Label(ctrl_frame, text="  Traversals/Chunk:").pack(side=tk.LEFT, padx=(12, 2))
        self.spin_hands = tk.Spinbox(
            ctrl_frame,
            from_=1,
            to=4096,
            width=5,
            bg="#313244",
            fg=DARK_FG,
            buttonbackground="#45475a",
            insertbackground=DARK_FG,
        )
        self.spin_hands.delete(0, tk.END)
        self.spin_hands.insert(0, str(self.trainer.config.traversals_per_chunk))
        self.spin_hands.pack(side=tk.LEFT, padx=2)
        self.spin_hands.bind("<Return>", self._update_hands_per_batch)
        self.spin_hands.bind("<FocusOut>", self._update_hands_per_batch)

        self.btn_save = ttk.Button(ctrl_frame, text="Save Model", command=self._save_model)
        self.btn_save.pack(side=tk.RIGHT, padx=4)

    def _build_perf_tab(self):
        tab = self.perf_tab

        top = ttk.Frame(tab)
        top.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(top, text="Training Overhead Profiler", font=("Segoe UI", 14, "bold"), foreground="#a6e3a1").pack(anchor=tk.W)
        ttk.Label(
            top,
            text="Average wall-clock time per traversal, split by rollout and optimization (milliseconds).",
            font=("Segoe UI", 10),
        ).pack(anchor=tk.W, pady=(2, 10))

        grid_frame = ttk.Frame(tab)
        grid_frame.pack(fill=tk.X, padx=20, pady=10)

        self.perf_labels = {}

        metrics = [
            ("total_time", "Total Time/Traversal", "#cdd6f4"),
            ("rollout_total_time", "Rollout / Game Sim", "#f9e2af"),
            ("state_init_time", "State Init / Deal", "#94e2d5"),
            ("chance_time", "Chance / Board Deal", "#f5c2e7"),
            ("traverser_state_time", "Traverser State Encode", "#89b4fa"),
            ("opponent_state_time", "Opponent State Encode", "#74c7ec"),
            ("regret_infer_time", "Regret Inference", "#b4befe"),
            ("strategy_infer_time", "Strategy Inference", "#cba6f7"),
            ("branch_clone_time", "Branch Clone", "#fab387"),
            ("apply_time", "Apply Actions", "#f38ba8"),
            ("regret_train_time", "Regret Training", "#a6e3a1"),
            ("strategy_train_time", "Strategy Training", "#a6adc8"),
            ("snapshot_time", "Snapshot Publish", "#f9e2af"),
            ("other_time", "Untracked / Other", "#bac2de"),
        ]

        for key, title, color in metrics:
            row = ttk.Frame(grid_frame)
            row.pack(fill=tk.X, pady=6)

            ttk.Label(row, text=f"{title}:", font=("Consolas", 12), width=25).pack(side=tk.LEFT)
            lbl = ttk.Label(row, text="0.00", font=("Consolas", 12, "bold"), foreground=color)
            lbl.pack(side=tk.LEFT, padx=10)

            self.perf_labels[key] = lbl

    def _build_eval_tab(self):
        tab = self.eval_tab

        ctrl = ttk.LabelFrame(tab, text="Evaluation Settings", padding=10)
        ctrl.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(ctrl, text="Evaluation Hands:").pack(side=tk.LEFT, padx=(0, 4))
        self.spin_eval_hands = tk.Spinbox(
            ctrl,
            from_=1,
            to=10000,
            width=7,
            bg="#313244",
            fg=DARK_FG,
            buttonbackground="#45475a",
            insertbackground=DARK_FG,
        )
        self.spin_eval_hands.delete(0, tk.END)
        self.spin_eval_hands.insert(0, "500")
        self.spin_eval_hands.pack(side=tk.LEFT, padx=4)

        ttk.Label(ctrl, text="Mode:").pack(side=tk.LEFT, padx=(12, 4))
        self.eval_mode = ttk.Combobox(ctrl, values=["heuristics", "checkpoints", "v21_table"], width=12, state="readonly")
        self.eval_mode.set("heuristics")
        self.eval_mode.pack(side=tk.LEFT, padx=4)

        ttk.Label(ctrl, text="Chart Scope:").pack(side=tk.LEFT, padx=(12, 4))
        self.eval_chart_scope = ttk.Combobox(ctrl, values=EVAL_SCOPE_LABELS, width=10, state="readonly")
        self.eval_chart_scope.set("Overall")
        self.eval_chart_scope.pack(side=tk.LEFT, padx=4)
        self.eval_chart_scope.bind("<<ComboboxSelected>>", self._on_eval_chart_selection_changed)

        ttk.Label(ctrl, text="Metric:").pack(side=tk.LEFT, padx=(12, 4))
        self.eval_chart_metric = ttk.Combobox(ctrl, values=EVAL_METRIC_LABELS, width=8, state="readonly")
        self.eval_chart_metric.set("VPIP")
        self.eval_chart_metric.pack(side=tk.LEFT, padx=4)
        self.eval_chart_metric.bind("<<ComboboxSelected>>", self._on_eval_chart_selection_changed)

        self.btn_eval = ttk.Button(ctrl, text="Run Evaluation", command=self._run_eval)
        self.btn_eval.pack(side=tk.LEFT, padx=12)

        self.lbl_eval_status = ttk.Label(ctrl, text="", font=("Segoe UI", 10, "italic"))
        self.lbl_eval_status.pack(side=tk.LEFT, padx=12)

        results_frame = ttk.Frame(tab)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        text_frame = ttk.LabelFrame(results_frame, text="Results", padding=10)
        text_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.eval_text = tk.Text(
            text_frame,
            bg="#181825",
            fg=DARK_FG,
            font=("Consolas", 11),
            wrap=tk.WORD,
            state=tk.DISABLED,
            insertbackground=DARK_FG,
            relief=tk.FLAT,
            padx=10,
            pady=10,
        )
        self.eval_text.pack(fill=tk.BOTH, expand=True)

        chart_frame = ttk.LabelFrame(results_frame, text="Evaluation Charts", padding=10)
        chart_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        if MATPLOTLIB_AVAILABLE:
            self.eval_fig = Figure(figsize=(7, 9), dpi=100, facecolor=DARK_BG)
            self.eval_fig.subplots_adjust(hspace=0.55)
            self.eval_ax = self.eval_fig.add_subplot(3, 1, 1)
            self.eval_vpip_ax = self.eval_fig.add_subplot(3, 1, 2)
            self.eval_hand_ax = self.eval_fig.add_subplot(3, 1, 3)
            for ax in [self.eval_ax, self.eval_vpip_ax, self.eval_hand_ax]:
                ax.set_facecolor("#181825")
                ax.tick_params(colors=DARK_FG, labelsize=9)
                for spine in ax.spines.values():
                    spine.set_color(DARK_GRID)

            self.eval_canvas = FigureCanvasTkAgg(self.eval_fig, master=chart_frame)
            self.eval_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        else:
            self.eval_fig = None
            self.eval_ax = None
            self.eval_vpip_ax = None
            self.eval_hand_ax = None
            self.eval_canvas = None
            self.eval_fallback_action_canvas = tk.Canvas(
                chart_frame,
                bg="#181825",
                highlightthickness=1,
                highlightbackground=DARK_GRID,
                height=220,
            )
            self.eval_fallback_action_canvas.pack(fill=tk.BOTH, expand=True, pady=(0, 4))
            self.eval_fallback_vpip_canvas = tk.Canvas(
                chart_frame,
                bg="#181825",
                highlightthickness=1,
                highlightbackground=DARK_GRID,
                height=220,
            )
            self.eval_fallback_vpip_canvas.pack(fill=tk.BOTH, expand=True, pady=4)
            self.eval_fallback_hand_canvas = tk.Canvas(
                chart_frame,
                bg="#181825",
                highlightthickness=1,
                highlightbackground=DARK_GRID,
                height=300,
            )
            self.eval_fallback_hand_canvas.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

    def _read_int(self, widget: tk.Spinbox, fallback: int) -> int:
        try:
            return max(1, int(widget.get()))
        except ValueError:
            return fallback

    def _reset_histories(self) -> None:
        for key in ["loss_history", "aux_loss_history", "bb100_history", "epsilon_history", "hands_history"]:
            self.gui_state[key] = []
        for i in range(len(ACTION_NAMES)):
            self.gui_state["action_history"][i] = []
        for pos in range(6):
            self.gui_state["position_bb100_history"][pos] = []

    def _training_thread_func(self) -> None:
        self.training_running = True
        self.stop_requested = False
        self.gui_state["status"] = "Training..."
        if self.training_start_time is None:
            self.training_start_time = time.time()

        while self.gui_state["batch"] < self.gui_state["total_batches"]:
            while self.pause_requested and not self.stop_requested:
                self.gui_state["status"] = "Paused"
                time.sleep(0.2)

            if self.stop_requested:
                break

            self.gui_state["status"] = "Training..."
            chunk = self._read_int(self.spin_hands, self.trainer.config.traversals_per_chunk)
            self.trainer.config.traversals_per_chunk = chunk
            remaining = self.gui_state["total_batches"] - self.gui_state["batch"]
            run_chunk = min(chunk, max(1, remaining))
            chunk_start = time.perf_counter()
            snapshot = self.trainer.train_for_traversals(run_chunk)
            chunk_elapsed = max(1e-6, time.perf_counter() - chunk_start)
            self._ingest_snapshot(snapshot, run_chunk, chunk_elapsed)

        self.gui_state["status"] = "Stopped" if self.stop_requested else "Done"
        self.training_running = False

    def _ingest_snapshot(self, snapshot, chunk_size: int, chunk_elapsed_seconds: float) -> None:
        total_actions = max(1, sum(snapshot.action_histogram))
        action_pcts = {
            i: (snapshot.action_histogram[i] / total_actions * 100.0)
            for i in range(len(ACTION_NAMES))
        }

        pos_stats = {}
        approx_hands = max(0, snapshot.traversals_completed // 6)
        for i, name in enumerate(POSITION_NAMES):
            avg = float(snapshot.position_avg_utility_bb.get(name, 0.0)) * 100.0
            pos_stats[i] = {"hands": approx_hands, "avg": avg, "win": 0.0}

        chunk_size = max(1, int(chunk_size))
        chunk_elapsed_seconds = max(1e-6, float(chunk_elapsed_seconds))
        chunk_speed = float(chunk_size / chunk_elapsed_seconds)
        prev_speed = float(self.gui_state.get("speed", 0.0))
        speed = chunk_speed if prev_speed <= 0.0 else (0.35 * chunk_speed + 0.65 * prev_speed)
        remaining = (
            (self.gui_state["total_batches"] - snapshot.traversals_completed) / speed / 60.0
            if speed > 0
            else 0.0
        )

        self.gui_state["batch"] = snapshot.traversals_completed
        self.gui_state["loss"] = snapshot.regret_loss
        self.gui_state["aux_loss"] = snapshot.strategy_loss
        self.gui_state["epsilon"] = snapshot.traverser_decisions
        self.gui_state["bb_per_hand"] = snapshot.avg_utility_bb
        self.gui_state["bb_100"] = snapshot.avg_utility_bb * 100.0
        self.gui_state["bb100_window"] = snapshot.utility_window_count
        self.gui_state["style_window"] = snapshot.style_window_count
        self.gui_state["position_window"] = snapshot.position_window_size
        self.gui_state["speed"] = speed
        self.gui_state["eta"] = remaining
        self.gui_state["pool_size"] = snapshot.checkpoint_pool_size
        self.gui_state["learner_steps"] = snapshot.learner_steps
        self.gui_state["total_hands"] = snapshot.traversals_completed
        self.gui_state["buffer_size"] = snapshot.advantage_buffer_size
        self.gui_state["buffer_size_aux"] = snapshot.strategy_buffer_size
        self.gui_state["action_pcts"] = action_pcts
        self.gui_state["position_stats"] = pos_stats
        self.gui_state["vpip"] = snapshot.vpip * 100.0
        self.gui_state["pfr"] = snapshot.pfr * 100.0
        self.gui_state["three_bet"] = snapshot.three_bet * 100.0
        self.gui_state["state_err"] = float(snapshot.invalid_state_count)
        self.gui_state["fallbacks"] = float(snapshot.invalid_action_count)
        self.gui_state["perf_stats"] = dict(snapshot.perf_breakdown_ms)

        self.gui_state["loss_history"].append(snapshot.regret_loss)
        self.gui_state["aux_loss_history"].append(snapshot.strategy_loss)
        self.gui_state["bb100_history"].append(snapshot.avg_utility_bb * 100.0)
        self.gui_state["epsilon_history"].append(float(snapshot.traverser_decisions))
        self.gui_state["hands_history"].append(snapshot.traversals_completed)
        for i in range(len(ACTION_NAMES)):
            self.gui_state["action_history"][i].append(action_pcts.get(i, 0.0))
        for pos in range(6):
            self.gui_state["position_bb100_history"][pos].append(pos_stats[pos]["avg"])

    def _start_training(self) -> None:
        if self.training_running:
            return

        self._reset_histories()

        try:
            target = int(self.spin_batches.get())
        except ValueError:
            target = 524288

        current = self.gui_state["batch"]
        if current > 0:
            self.gui_state["total_batches"] = max(target, current + target if target <= current else target)
        else:
            self.gui_state["total_batches"] = target

        self.stop_requested = False
        self.pause_requested = False

        self.btn_start.config(state=tk.DISABLED)
        self.btn_pause.config(state=tk.NORMAL, text="Pause")
        self.btn_stop.config(state=tk.NORMAL)

        self.train_thread = threading.Thread(target=self._training_thread_func, daemon=True)
        self.train_thread.start()

    def _toggle_pause(self) -> None:
        self.pause_requested = not self.pause_requested
        if self.pause_requested:
            self.btn_pause.config(text="Resume")
        else:
            self.btn_pause.config(text="Pause")

    def _stop_training(self) -> None:
        self.stop_requested = True
        self.pause_requested = False
        self.btn_start.config(state=tk.NORMAL)
        self.btn_pause.config(state=tk.DISABLED, text="Pause")
        self.btn_stop.config(state=tk.DISABLED)

    def _extend_training(self) -> None:
        try:
            extra = int(self.spin_batches.get())
        except ValueError:
            extra = 1000
        self.gui_state["total_batches"] = self.gui_state["batch"] + extra

        if not self.training_running:
            self.stop_requested = False
            self.pause_requested = False
            self.btn_start.config(state=tk.DISABLED)
            self.btn_pause.config(state=tk.NORMAL, text="Pause")
            self.btn_stop.config(state=tk.NORMAL)
            self.train_thread = threading.Thread(target=self._training_thread_func, daemon=True)
            self.train_thread.start()

    def _update_hands_per_batch(self, event=None) -> None:
        try:
            val = int(self.spin_hands.get())
            self.trainer.config.traversals_per_chunk = max(1, min(4096, val))
        except ValueError:
            pass

    def _save_model(self) -> None:
        self.trainer.save_checkpoint(DEFAULT_CHECKPOINT_PATH)
        os.makedirs(DEFAULT_RESULTS_DIR, exist_ok=True)
        image_filename = f"{os.path.splitext(DEFAULT_CHECKPOINT_FILENAME)[0]}_training_tab.png"
        image_path = os.path.join(DEFAULT_RESULTS_DIR, image_filename)
        image_saved, detail = self._save_training_tab_image(image_path)
        if image_saved:
            messagebox.showinfo("Save", f"Model saved to:\n{DEFAULT_CHECKPOINT_PATH}\n\nTraining tab image saved to:\n{detail}")
        else:
            messagebox.showwarning(
                "Save",
                f"Model saved to:\n{DEFAULT_CHECKPOINT_PATH}\n\nTraining tab image export failed:\n{detail}",
            )

    def _save_training_tab_image(self, image_path: str) -> tuple[bool, str]:
        prior_tab = None
        try:
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            prior_tab = self.notebook.select()
            self.notebook.select(self.training_tab)
            self.update()
            self.update_idletasks()

            x = int(self.training_tab.winfo_rootx())
            y = int(self.training_tab.winfo_rooty())
            w = int(self.training_tab.winfo_width())
            h = int(self.training_tab.winfo_height())
            if w <= 1 or h <= 1:
                raise RuntimeError("Training tab size is not ready for capture.")

            if IMAGE_CAPTURE_AVAILABLE and ImageGrab is not None:
                image = ImageGrab.grab(bbox=(x, y, x + w, y + h))
                image.save(image_path)
                return True, image_path

            if MATPLOTLIB_AVAILABLE and self.fig is not None:
                self.fig.savefig(image_path, dpi=160, bbox_inches="tight")
                return True, f"{image_path} (charts-only fallback)"

            raise RuntimeError("No image capture backend is available.")
        except Exception as exc:
            return False, str(exc)
        finally:
            if prior_tab:
                self.notebook.select(prior_tab)
                self.update_idletasks()

    def _run_eval(self) -> None:
        try:
            num_hands = int(self.spin_eval_hands.get())
        except ValueError:
            num_hands = 500

        mode = self.eval_mode.get().strip() or "heuristics"
        self.lbl_eval_status.config(text="Running evaluation...")
        self.btn_eval.config(state=tk.DISABLED)
        self.update_idletasks()

        def _eval_thread():
            try:
                if mode == "checkpoints":
                    results = self.trainer.evaluate_vs_checkpoint_pool(num_hands)
                elif mode == "v21_table":
                    results = self.trainer.evaluate_vs_v21_table(num_hands)
                else:
                    results = self.trainer.evaluate_vs_heuristics(num_hands)
                self.after(0, lambda: self._show_eval_results(results))
            except Exception as exc:
                error_message = str(exc)
                self.after(0, lambda msg=error_message: self._show_eval_failure(msg))

        threading.Thread(target=_eval_thread, daemon=True).start()

    def _show_eval_failure(self, error_message: str) -> None:
        self.btn_eval.config(state=tk.NORMAL)
        self.lbl_eval_status.config(text="Evaluation failed")
        messagebox.showerror("Evaluation Error", error_message)

    def _selected_eval_scope_key(self) -> str:
        label = (self.eval_chart_scope.get() if hasattr(self, "eval_chart_scope") else "Overall").strip()
        return label if label in POSITION_NAMES else "overall"

    def _selected_eval_metric_key(self) -> str:
        label = (self.eval_chart_metric.get() if hasattr(self, "eval_chart_metric") else "VPIP").strip()
        return EVAL_METRIC_KEY_BY_LABEL.get(label, "vpip")

    def _eval_metric_label(self, metric_key: str) -> str:
        return EVAL_METRIC_LABEL_BY_KEY.get(metric_key, "VPIP")

    def _extract_eval_metric_data(self, results, metric_key: str, scope_key: str):
        metric_key = metric_key if metric_key in EVAL_METRIC_LABEL_BY_KEY else "vpip"
        overall_pct = float(getattr(results, metric_key, 0.0)) * 100.0
        by_position_attr = EVAL_METRIC_POSITION_ATTR.get(metric_key, "vpip_by_position")
        by_position = getattr(results, by_position_attr, {}) or {}

        if scope_key in POSITION_NAMES:
            scope_label = scope_key
            selected_pct = float(by_position.get(scope_key, 0.0)) * 100.0
        else:
            scope_label = "Overall"
            selected_pct = overall_pct

        hand_grid_attr = EVAL_METRIC_GRID_ATTR.get(metric_key, "vpip_hand_grid")
        hand_grid = getattr(results, hand_grid_attr, [])
        if scope_key in POSITION_NAMES:
            hand_grid_by_position_attr = EVAL_METRIC_GRID_BY_POSITION_ATTR.get(metric_key, "vpip_hand_grid_by_position")
            hand_grid_by_position = getattr(results, hand_grid_by_position_attr, {}) or {}
            hand_grid = hand_grid_by_position.get(scope_key, hand_grid)
        return overall_pct, selected_pct, hand_grid, scope_label

    def _on_eval_chart_selection_changed(self, _event=None) -> None:
        if self._last_eval_report is not None:
            self._render_eval_charts(self._last_eval_report)

    def _render_eval_charts(self, results) -> None:
        scope_key = self._selected_eval_scope_key()
        metric_key = self._selected_eval_metric_key()
        metric_label = self._eval_metric_label(metric_key)
        overall_pct, selected_pct, hand_grid, scope_label = self._extract_eval_metric_data(
            results,
            metric_key,
            scope_key,
        )

        if self.eval_ax is not None:
            self.eval_ax.clear()
            self.eval_ax.set_facecolor("#181825")
            counts = list(results.action_histogram)
            total = sum(counts)
            pcts = [count / total * 100.0 if total > 0 else 0.0 for count in counts]
            bars = self.eval_ax.bar(ACTION_NAMES, pcts, color=ACCENT_COLORS[: len(ACTION_NAMES)], edgecolor="none")
            self.eval_ax.set_ylabel("% of Actions", color=DARK_FG, fontsize=9)
            self.eval_ax.set_title(f"Eval Actions ({results.hands:,} hands)", color=DARK_FG, fontsize=10)
            self.eval_ax.tick_params(colors=DARK_FG, labelsize=8)
            for spine in self.eval_ax.spines.values():
                spine.set_color(DARK_GRID)

            for bar, pct in zip(bars, pcts):
                if pct > 2.0:
                    self.eval_ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5,
                        f"{pct:.1f}%",
                        ha="center",
                        va="bottom",
                        color=DARK_FG,
                        fontsize=8,
                    )

            self.eval_vpip_ax.clear()
            self.eval_vpip_ax.set_facecolor("#181825")
            if scope_key in POSITION_NAMES:
                labels = [scope_key]
                values = [selected_pct]
                colors = [ACCENT_COLORS[POSITION_NAMES.index(scope_key) % len(ACCENT_COLORS)]]
            else:
                labels = [metric_label]
                values = [overall_pct]
                colors = [ACCENT_COLORS[2]]
            metric_bars = self.eval_vpip_ax.bar(labels, values, color=colors, edgecolor="none")
            self.eval_vpip_ax.set_ylabel(f"{metric_label} %", color=DARK_FG, fontsize=9)
            subtitle = scope_label
            self.eval_vpip_ax.set_title(f"{metric_label} {subtitle}", color=DARK_FG, fontsize=10)
            self.eval_vpip_ax.tick_params(colors=DARK_FG, labelsize=8)
            for spine in self.eval_vpip_ax.spines.values():
                spine.set_color(DARK_GRID)
            top = max(5.0, max(values) * 1.2 if values else 5.0)
            self.eval_vpip_ax.set_ylim(0.0, top)
            for bar, pct in zip(metric_bars, values):
                self.eval_vpip_ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(0.3, top * 0.02),
                    f"{pct:.1f}%",
                    ha="center",
                    va="bottom",
                    color=DARK_FG,
                    fontsize=8,
                )

            self.eval_hand_ax.clear()
            self.eval_hand_ax.set_facecolor("#181825")
            grid = np.asarray(hand_grid, dtype=np.float32) * 100.0
            if grid.shape == (13, 13):
                self.eval_hand_ax.imshow(grid, cmap="RdYlGn", vmin=0.0, vmax=100.0, aspect="auto")
                self.eval_hand_ax.set_xticks(range(len(RANK_LABELS)))
                self.eval_hand_ax.set_xticklabels(RANK_LABELS, color=DARK_FG, fontsize=7)
                self.eval_hand_ax.set_yticks(range(len(RANK_LABELS)))
                self.eval_hand_ax.set_yticklabels(RANK_LABELS, color=DARK_FG, fontsize=7)
                self.eval_hand_ax.set_title(f"{scope_label} {metric_label} by Starting Hand", color=DARK_FG, fontsize=10)
                self.eval_hand_ax.text(
                    0.5,
                    -0.17,
                    "Upper-right = suited | Lower-left = offsuit | Diagonal = pairs",
                    transform=self.eval_hand_ax.transAxes,
                    ha="center",
                    va="top",
                    color=DARK_FG,
                    fontsize=8,
                    clip_on=False,
                )
            else:
                self.eval_hand_ax.set_title(f"{scope_label} {metric_label} by Starting Hand (No Data)", color=DARK_FG, fontsize=10)
                self.eval_hand_ax.set_xticks([])
                self.eval_hand_ax.set_yticks([])
            self.eval_canvas.draw()
        else:
            self._draw_eval_fallback(
                results.action_histogram,
                metric_label,
                scope_key,
                selected_pct,
                hand_grid,
                scope_label,
            )

    def _show_eval_results(self, results) -> None:
        self.btn_eval.config(state=tk.NORMAL)
        self.lbl_eval_status.config(text="Done!")

        if results is None:
            return
        self._last_eval_report = results

        self.eval_text.config(state=tk.NORMAL)
        self.eval_text.delete("1.0", tk.END)

        lines = [
            f"{'=' * 40}",
            "  EVALUATION RESULTS",
            f"{'=' * 40}",
            "",
            f"  Mode:            {results.mode}",
            f"  Hands Played:    {results.hands:,}",
            f"  Avg Profit:      {results.avg_profit_bb:+.3f} BB",
            f"  Win Rate:        {results.win_rate:.1%}",
            f"  VPIP:            {results.vpip:.1%}",
            f"  PFR:             {results.pfr:.1%}",
            f"  3-Bet:           {results.three_bet:.1%}",
            f"  Illegal Actions: {results.illegal_action_count}",
            f"  Runtime:         {results.runtime_seconds:.2f}s",
            "",
            f"{'-' * 40}",
            "  POSITION BREAKDOWN",
            f"{'-' * 40}",
        ]

        for pos_name in POSITION_NAMES:
            lines.append(f"  {pos_name:4s} {results.position_avg_profit_bb.get(pos_name, 0.0):+8.3f} BB")

        lines.append("")
        lines.append(f"{'-' * 40}")
        lines.append("  PREFLOP STYLE BY POSITION")
        lines.append(f"{'-' * 40}")
        for pos_name in POSITION_NAMES:
            lines.append(
                f"  {pos_name:4s} VPIP {results.vpip_by_position.get(pos_name, 0.0):6.1%} | "
                f"PFR {results.pfr_by_position.get(pos_name, 0.0):6.1%} | "
                f"3-Bet {results.three_bet_by_position.get(pos_name, 0.0):6.1%}"
            )

        lines.append("")
        lines.append(f"{'-' * 40}")
        lines.append("  ACTION BREAKDOWN")
        lines.append(f"{'-' * 40}")

        total = sum(results.action_histogram)
        for i, name in enumerate(ACTION_NAMES):
            count = results.action_histogram[i]
            pct = count / total * 100.0 if total > 0 else 0.0
            lines.append(f"  {name:8s}  {count:5d}  ({pct:5.1f}%)")

        self.eval_text.insert(tk.END, "\n".join(lines))
        self.eval_text.config(state=tk.DISABLED)
        self._render_eval_charts(results)

    def update_gui(self) -> None:
        gs = self.gui_state

        self.lbl_status.config(text=f"Status: {gs['status']}")
        warmup_target = self.trainer.config.warmup_advantage_samples
        if gs["buffer_size"] < warmup_target:
            phase_text = f"Learner: Warmup {gs['buffer_size']:,}/{warmup_target:,}"
        else:
            phase_text = f"Learner: Updating | Steps: {gs.get('learner_steps', 0):,} | Pool: {gs['pool_size']}"
        self.lbl_phase.config(text=phase_text)
        pct = gs["batch"] / max(1, gs["total_batches"]) * 100.0
        self.prog_bar["value"] = pct
        self.lbl_progress.config(text=f"{gs['batch']:,} / {gs['total_batches']:,} traversals ({pct:.0f}%)")
        self.lbl_speed.config(text=f"Speed: {gs['speed']:.1f} t/s | ETA: {gs['eta']:.1f}m | Traversals: {gs['total_hands']:,}")

        self.lbl_eps.config(text=f"Decisions: {int(gs['epsilon']):,}")
        self.lbl_loss_val.config(text=f"Regret: {gs['loss']:.4f} | Strategy: {gs['aux_loss']:.4f}")
        bb100_window = int(gs.get("bb100_window", 0))
        self.lbl_bb100_val.config(
            text=f"BB/hand: {gs.get('bb_per_hand', 0.0):+.3f} | BB/100: {gs['bb_100']:+.1f} ({bb100_window}h)"
        )
        self.lbl_buffer.config(
            text=(
                f"Adv: {gs['buffer_size']:,}/{gs['buffer_cap']//1000}k "
                f"| Strat: {gs['buffer_size_aux']:,}/{gs['buffer_cap_aux']//1000}k"
            )
        )
        self.lbl_hands.config(text=f"Traversals: {gs['total_hands']:,}")
        self.lbl_pool.config(text=f"Pool: {gs['pool_size']}")

        for i, name in enumerate(ACTION_NAMES):
            if i in self.action_labels:
                self.action_labels[i].config(text=f"{name}: {gs['action_pcts'].get(i, 0.0):.1f}%")

        self.lbl_vpip.config(text=f"VPIP: {gs.get('vpip', 0.0):.1f}%")
        self.lbl_pfr.config(text=f"PFR: {gs.get('pfr', 0.0):.1f}%")
        self.lbl_3bet.config(text=f"3-Bet: {gs.get('three_bet', 0.0):.1f}%")

        for i, pname in enumerate(POSITION_NAMES):
            ps = gs["position_stats"].get(i, {})
            avg = ps.get("avg", 0.0)
            self.pos_labels[i].config(text=f"{pname}: {avg:+.1f} BB/100")

        if gs["status"] in {"Done", "Stopped"} and not self.training_running:
            self.btn_start.config(state=tk.NORMAL)
            self.btn_pause.config(state=tk.DISABLED, text="Pause")
            self.btn_stop.config(state=tk.DISABLED)

        for key, lbl in self.perf_labels.items():
            val = gs["perf_stats"].get(key, 0.0)
            lbl.config(text=f"{val:.3f} ms")

        total_action_pct = sum(gs["action_pcts"].values())
        fold_pct = gs["action_pcts"].get(0, 0.0)
        entropy = self.trainer.get_snapshot().action_entropy
        if gs["buffer_size"] < warmup_target:
            health_text = (
                f"Warming up: updates start after {warmup_target:,} advantage samples. "
                f"BB/hand uses the last {int(gs.get('bb100_window', 0)):,} traversals; BB/100 is BB/hand x 100."
            )
        elif int(gs.get("state_err", 0.0)) > 0 or int(gs.get("fallbacks", 0.0)) > 0:
            health_text = "Warning: state/action errors should stay at 0. Stop and inspect encoder or action mapping."
        elif total_action_pct > 0.0 and fold_pct >= 80.0:
            health_text = (
                "Warning: policy is folding too often. Healthy training should show some calls and raises, "
                "not mostly Fold."
            )
        elif total_action_pct > 0.0 and entropy < 0.35:
            health_text = "Warning: entropy is very low. Healthy values are roughly 0.8-2.0 with a mixed action profile."
        else:
            health_text = (
                f"Healthy: BB/hand is over the last {int(gs.get('bb100_window', 0)):,} traversals, "
                f"BB/100 is the same rolling value x 100, "
                f"VPIP/PFR/3-Bet use the last {int(gs.get('style_window', 0)):,}, "
                f"and position BB/100 uses up to {int(gs.get('position_window', 0))} seat samples."
            )
        self.lbl_health.config(text=health_text)

        self._update_charts(gs)
        self.after(1000, self.update_gui)

    def _update_charts(self, gs) -> None:
        if self.fig is None:
            self._update_tk_charts(gs)
            return

        self.ax_bb100.clear()
        self.ax_bb100.set_facecolor("#181825")
        self.ax_bb100.grid(True, color=DARK_GRID, alpha=0.3, linewidth=0.5)
        self.ax_bb100.set_title("BB/100 Over Time", color=DARK_FG, fontsize=10)
        if gs["bb100_history"]:
            data = gs["bb100_history"]
            self.ax_bb100.plot(data, color="#a6e3a1", linewidth=1, alpha=0.4)
            if len(data) > 10:
                window = min(50, max(2, len(data) // 3))
                smoothed = np.convolve(data, np.ones(window) / window, mode="valid")
                self.ax_bb100.plot(range(window - 1, len(data)), smoothed, color="#a6e3a1", linewidth=2)
            self.ax_bb100.axhline(y=0, color="#585b70", linewidth=1, linestyle="--")
        self.ax_bb100.tick_params(colors=DARK_FG, labelsize=8)
        for spine in self.ax_bb100.spines.values():
            spine.set_color(DARK_GRID)

        self.ax_loss.clear()
        self.ax_loss.set_facecolor("#181825")
        self.ax_loss.grid(True, color=DARK_GRID, alpha=0.3, linewidth=0.5)
        self.ax_loss.set_title("Training Loss", color=DARK_FG, fontsize=10)

        if gs["loss_history"]:
            data_regret = gs["loss_history"]
            self.ax_loss.plot(data_regret, color="#f38ba8", linewidth=1, alpha=0.4, label="Regret")
            if len(data_regret) > 10:
                window = min(50, max(2, len(data_regret) // 3))
                smoothed = np.convolve(data_regret, np.ones(window) / window, mode="valid")
                self.ax_loss.plot(range(window - 1, len(data_regret)), smoothed, color="#f38ba8", linewidth=2)

        if gs["aux_loss_history"]:
            data_strategy = gs["aux_loss_history"]
            self.ax_loss.plot(data_strategy, color="#89b4fa", linewidth=1, alpha=0.35, label="Strategy")
            if len(data_strategy) > 10:
                window = min(50, max(2, len(data_strategy) // 3))
                smoothed = np.convolve(data_strategy, np.ones(window) / window, mode="valid")
                self.ax_loss.plot(range(window - 1, len(data_strategy)), smoothed, color="#89b4fa", linewidth=2)
        if gs["loss_history"] or gs["aux_loss_history"]:
            self.ax_loss.legend(
                loc="upper right",
                fontsize=6,
                framealpha=0.5,
                facecolor=DARK_BG,
                edgecolor=DARK_GRID,
                labelcolor=DARK_FG,
            )
        self.ax_loss.tick_params(colors=DARK_FG, labelsize=8)
        for spine in self.ax_loss.spines.values():
            spine.set_color(DARK_GRID)

        self.ax_actions.clear()
        self.ax_actions.set_facecolor("#181825")
        self.ax_actions.grid(True, color=DARK_GRID, alpha=0.3, linewidth=0.5)
        self.ax_actions.set_title("Action Distribution", color=DARK_FG, fontsize=10)
        hist = gs["action_history"]
        if hist[0]:
            n = len(hist[0])
            x = range(n)
            stacks = [hist[i] for i in range(len(ACTION_NAMES))]
            self.ax_actions.stackplot(x, *stacks, colors=ACCENT_COLORS[: len(ACTION_NAMES)], labels=ACTION_NAMES, alpha=0.8)
            if n < 100:
                self.ax_actions.legend(
                    loc="upper right",
                    fontsize=6,
                    framealpha=0.5,
                    facecolor=DARK_BG,
                    edgecolor=DARK_GRID,
                    labelcolor=DARK_FG,
                )
        self.ax_actions.tick_params(colors=DARK_FG, labelsize=8)
        for spine in self.ax_actions.spines.values():
            spine.set_color(DARK_GRID)

        self.ax_positions.clear()
        self.ax_positions.set_facecolor("#181825")
        self.ax_positions.grid(True, color=DARK_GRID, alpha=0.3, linewidth=0.5, axis="y")
        self.ax_positions.set_title("Position Performance (BB/100)", color=DARK_FG, fontsize=10)
        pos_vals = [gs["position_stats"].get(i, {}).get("avg", 0.0) for i in range(6)]
        colors = ["#a6e3a1" if v >= 0 else "#f38ba8" for v in pos_vals]
        bars = self.ax_positions.bar(POSITION_NAMES, pos_vals, color=colors, edgecolor="none", alpha=0.85)
        self.ax_positions.axhline(y=0, color="#585b70", linewidth=1, linestyle="--")
        for bar, val in zip(bars, pos_vals):
            y = bar.get_height() + (0.5 if val >= 0 else -1.5)
            self.ax_positions.text(
                bar.get_x() + bar.get_width() / 2,
                y,
                f"{val:+.1f}",
                ha="center",
                va="bottom" if val >= 0 else "top",
                color=DARK_FG,
                fontsize=8,
            )
        self.ax_positions.tick_params(colors=DARK_FG, labelsize=8)
        for spine in self.ax_positions.spines.values():
            spine.set_color(DARK_GRID)

        self.canvas.draw_idle()

    def _update_tk_charts(self, gs) -> None:
        canvases = getattr(self, "tk_chart_canvases", {})
        if not canvases:
            return

        self._draw_line_chart(
            canvases.get("bb100"),
            gs["bb100_history"],
            "#a6e3a1",
            y_zero=True,
        )
        self._draw_line_chart(
            canvases.get("loss"),
            gs["loss_history"],
            "#f38ba8",
            y_zero=False,
            secondary=gs["aux_loss_history"],
            secondary_color="#89b4fa",
        )
        self._draw_stack_chart(
            canvases.get("actions"),
            [gs["action_history"][i] for i in range(len(ACTION_NAMES))],
            ACCENT_COLORS[: len(ACTION_NAMES)],
        )
        self._draw_bar_chart(
            canvases.get("positions"),
            POSITION_NAMES,
            [gs["position_stats"].get(i, {}).get("avg", 0.0) for i in range(len(POSITION_NAMES))],
        )

    def _draw_chart_frame(self, canvas: tk.Canvas):
        if canvas is None:
            return None
        canvas.delete("all")
        canvas.update_idletasks()
        width = max(200, canvas.winfo_width())
        height = max(140, canvas.winfo_height())
        canvas.create_rectangle(1, 1, width - 1, height - 1, outline=DARK_GRID, width=1)
        left = 36
        top = 12
        right = width - 12
        bottom = height - 24
        canvas.create_line(left, top, left, bottom, fill=DARK_GRID, width=1)
        canvas.create_line(left, bottom, right, bottom, fill=DARK_GRID, width=1)
        return width, height, left, top, right, bottom

    def _draw_line_chart(
        self,
        canvas: tk.Canvas,
        data,
        color: str,
        y_zero: bool,
        secondary=None,
        secondary_color: str = "#89b4fa",
    ) -> None:
        frame = self._draw_chart_frame(canvas)
        if frame is None:
            return
        width, height, left, top, right, bottom = frame

        series = []
        if data:
            series.extend(float(v) for v in data)
        if secondary:
            series.extend(float(v) for v in secondary)
        if not series:
            canvas.create_text(
                width / 2,
                height / 2,
                text="Waiting for data",
                fill="#6c7086",
                font=("Consolas", 11),
            )
            return

        y_min = min(series)
        y_max = max(series)
        if y_zero:
            y_min = min(y_min, 0.0)
            y_max = max(y_max, 0.0)
        if abs(y_max - y_min) < 1e-6:
            pad = max(1.0, abs(y_max) * 0.1 + 0.5)
            y_min -= pad
            y_max += pad
        else:
            pad = (y_max - y_min) * 0.1
            y_min -= pad
            y_max += pad

        def y_to_px(value: float) -> float:
            ratio = (value - y_min) / max(1e-9, y_max - y_min)
            return bottom - ratio * (bottom - top)

        if y_zero and y_min <= 0.0 <= y_max:
            zero_y = y_to_px(0.0)
            canvas.create_line(left, zero_y, right, zero_y, fill="#585b70", dash=(4, 3))

        self._draw_line_series(canvas, data, color, left, top, right, bottom, y_min, y_max)
        if secondary:
            self._draw_line_series(
                canvas,
                secondary,
                secondary_color,
                left,
                top,
                right,
                bottom,
                y_min,
                y_max,
            )

        canvas.create_text(left - 6, top, text=f"{y_max:.1f}", fill="#7f849c", anchor="e", font=("Consolas", 8))
        canvas.create_text(
            left - 6,
            bottom,
            text=f"{y_min:.1f}",
            fill="#7f849c",
            anchor="e",
            font=("Consolas", 8),
        )

    def _draw_line_series(
        self,
        canvas: tk.Canvas,
        data,
        color: str,
        left: int,
        top: int,
        right: int,
        bottom: int,
        y_min: float,
        y_max: float,
    ) -> None:
        if not data:
            return

        count = len(data)
        x_span = max(1, right - left)
        y_span = max(1e-9, y_max - y_min)
        points = []
        for idx, value in enumerate(data):
            x = left if count == 1 else left + (idx / max(1, count - 1)) * x_span
            y = bottom - ((float(value) - y_min) / y_span) * (bottom - top)
            points.extend((x, y))

        if len(points) >= 4:
            canvas.create_line(*points, fill=color, width=1)

        if count > 10:
            window = min(50, max(2, count // 3))
            kernel = np.ones(window, dtype=np.float32) / float(window)
            smoothed = np.convolve(np.asarray(data, dtype=np.float32), kernel, mode="valid")
            smooth_points = []
            smooth_count = len(smoothed)
            for idx, value in enumerate(smoothed):
                src_idx = idx + window - 1
                x = left if count == 1 else left + (src_idx / max(1, count - 1)) * x_span
                y = bottom - ((float(value) - y_min) / y_span) * (bottom - top)
                smooth_points.extend((x, y))
            if len(smooth_points) >= 4:
                canvas.create_line(*smooth_points, fill=color, width=2)

    def _draw_stack_chart(self, canvas: tk.Canvas, histories, colors) -> None:
        frame = self._draw_chart_frame(canvas)
        if frame is None:
            return
        width, height, left, top, right, bottom = frame

        if not histories or not histories[0]:
            canvas.create_text(
                width / 2,
                height / 2,
                text="Waiting for data",
                fill="#6c7086",
                font=("Consolas", 11),
            )
            return

        count = len(histories[0])
        x_span = max(1, right - left)
        cumulative = np.zeros(count, dtype=np.float32)

        for idx, history in enumerate(histories):
            arr = np.asarray(history, dtype=np.float32)
            top_line = cumulative + arr
            polygon = []
            for point_idx in range(count):
                x = left if count == 1 else left + (point_idx / max(1, count - 1)) * x_span
                y = bottom - (top_line[point_idx] / 100.0) * (bottom - top)
                polygon.extend((x, y))
            for point_idx in range(count - 1, -1, -1):
                x = left if count == 1 else left + (point_idx / max(1, count - 1)) * x_span
                y = bottom - (cumulative[point_idx] / 100.0) * (bottom - top)
                polygon.extend((x, y))
            if len(polygon) >= 6:
                canvas.create_polygon(
                    *polygon,
                    fill=colors[idx % len(colors)],
                    outline="",
                    stipple="gray50",
                )
            cumulative = top_line

        for pct in (25, 50, 75):
            y = bottom - (pct / 100.0) * (bottom - top)
            canvas.create_line(left, y, right, y, fill=DARK_GRID, dash=(2, 4))
            canvas.create_text(left - 6, y, text=str(pct), fill="#7f849c", anchor="e", font=("Consolas", 8))

    def _draw_bar_chart(self, canvas: tk.Canvas, labels, values) -> None:
        frame = self._draw_chart_frame(canvas)
        if frame is None:
            return
        width, height, left, top, right, bottom = frame

        if not values:
            canvas.create_text(
                width / 2,
                height / 2,
                text="Waiting for data",
                fill="#6c7086",
                font=("Consolas", 11),
            )
            return

        y_min = min(0.0, min(values))
        y_max = max(0.0, max(values))
        if abs(y_max - y_min) < 1e-6:
            y_min -= 1.0
            y_max += 1.0
        else:
            pad = (y_max - y_min) * 0.1
            y_min -= pad
            y_max += pad

        def y_to_px(value: float) -> float:
            ratio = (value - y_min) / max(1e-9, y_max - y_min)
            return bottom - ratio * (bottom - top)

        zero_y = y_to_px(0.0)
        canvas.create_line(left, zero_y, right, zero_y, fill="#585b70", dash=(4, 3))

        bar_area = max(1, right - left)
        slot = bar_area / max(1, len(values))
        bar_width = max(10, slot * 0.65)

        for idx, value in enumerate(values):
            center_x = left + slot * (idx + 0.5)
            x0 = center_x - bar_width / 2
            x1 = center_x + bar_width / 2
            y1 = y_to_px(float(value))
            fill = "#a6e3a1" if value >= 0 else "#f38ba8"
            canvas.create_rectangle(x0, zero_y, x1, y1, fill=fill, outline="")
            canvas.create_text(center_x, bottom + 10, text=labels[idx], fill=DARK_FG, font=("Consolas", 8))
            text_y = y1 - 8 if value < 0 else y1 + 8
            anchor = "s" if value < 0 else "n"
            canvas.create_text(center_x, text_y, text=f"{value:+.1f}", fill=DARK_FG, font=("Consolas", 8), anchor=anchor)

    def _draw_percentage_bar_chart(self, canvas: tk.Canvas, labels, values, title: str) -> None:
        frame = self._draw_chart_frame(canvas)
        if frame is None:
            return
        width, _, left, top, right, bottom = frame
        canvas.create_text((left + right) / 2, top + 2, text=title, fill=DARK_FG, font=("Consolas", 9, "bold"), anchor="n")
        chart_top = top + 18
        if not values:
            canvas.create_text(width / 2, (chart_top + bottom) / 2, text="Waiting for data", fill="#6c7086", font=("Consolas", 11))
            return

        max_pct = max(1.0, max(float(v) for v in values))
        slot = max(1.0, (right - left) / max(1, len(values)))
        bar_width = max(10.0, slot * 0.65)
        for idx, value in enumerate(values):
            center_x = left + slot * (idx + 0.5)
            x0 = center_x - bar_width / 2
            x1 = center_x + bar_width / 2
            y = bottom - (float(value) / max_pct) * max(1.0, (bottom - chart_top))
            canvas.create_rectangle(x0, bottom, x1, y, fill=ACCENT_COLORS[idx % len(ACCENT_COLORS)], outline="")
            canvas.create_text(center_x, bottom + 10, text=labels[idx], fill=DARK_FG, font=("Consolas", 8))
            canvas.create_text(center_x, y - 8, text=f"{float(value):.1f}%", fill=DARK_FG, font=("Consolas", 8))

    def _draw_heatmap_chart(self, canvas: tk.Canvas, grid_values, metric_label: str = "VPIP", scope_label: str = "Overall") -> None:
        frame = self._draw_chart_frame(canvas)
        if frame is None:
            return
        width, height, left, top, right, bottom = frame
        canvas.create_text(
            (left + right) / 2,
            top + 2,
            text=f"{scope_label} {metric_label} by Starting Hand",
            fill=DARK_FG,
            font=("Consolas", 9, "bold"),
            anchor="n",
        )
        chart_top = top + 18
        grid = np.asarray(grid_values, dtype=np.float32)
        if grid.shape != (13, 13):
            canvas.create_text(width / 2, height / 2, text="Waiting for data", fill="#6c7086", font=("Consolas", 11))
            return
        cell_w = (right - left) / 13.0
        cell_h = (bottom - chart_top) / 13.0
        for row in range(13):
            for col in range(13):
                value = float(grid[row, col])
                x0 = left + col * cell_w
                y0 = chart_top + row * cell_h
                x1 = x0 + cell_w
                y1 = y0 + cell_h
                red = int(243 - (77 * value))
                green = int(139 + (88 * value))
                blue = int(168 - (78 * value))
                fill = f"#{red:02x}{green:02x}{blue:02x}"
                canvas.create_rectangle(x0, y0, x1, y1, fill=fill, outline=DARK_GRID)
                label = "--" if value <= 0.0 else f"{value * 100.0:.0f}"
                canvas.create_text((x0 + x1) / 2, (y0 + y1) / 2, text=label, fill=DARK_FG, font=("Consolas", 5))
        for idx, rank in enumerate(RANK_LABELS):
            canvas.create_text(left + idx * cell_w + cell_w / 2, bottom + 10, text=rank, fill=DARK_FG, font=("Consolas", 7))
            canvas.create_text(left - 10, chart_top + idx * cell_h + cell_h / 2, text=rank, fill=DARK_FG, font=("Consolas", 7))
        canvas.create_text(right - 2, chart_top + 2, text="Suited", fill=DARK_FG, font=("Consolas", 7), anchor="ne")
        canvas.create_text(left + 2, bottom - 2, text="Offsuit", fill=DARK_FG, font=("Consolas", 7), anchor="sw")
        canvas.create_text(
            (left + right) / 2,
            height - 4,
            text="Upper-right = suited | Lower-left = offsuit | Diagonal = pairs",
            fill=DARK_FG,
            font=("Consolas", 7),
            anchor="s",
        )

    def _draw_eval_fallback(self, counts, metric_label, scope_key, selected_pct, hand_grid, scope_label) -> None:
        action_canvas = getattr(self, "eval_fallback_action_canvas", None)
        vpip_canvas = getattr(self, "eval_fallback_vpip_canvas", None)
        hand_canvas = getattr(self, "eval_fallback_hand_canvas", None)
        if action_canvas is None or vpip_canvas is None or hand_canvas is None:
            return

        frame = self._draw_chart_frame(action_canvas)
        if frame is None:
            return
        width, height, left, top, right, bottom = frame

        total = sum(counts)
        pcts = [(count / total * 100.0) if total > 0 else 0.0 for count in counts]
        max_pct = max([1.0] + pcts)
        bar_area = max(1, right - left)
        slot = bar_area / max(1, len(pcts))
        bar_width = max(12, slot * 0.65)

        for idx, pct in enumerate(pcts):
            center_x = left + slot * (idx + 0.5)
            x0 = center_x - bar_width / 2
            x1 = center_x + bar_width / 2
            y = bottom - (pct / max_pct) * (bottom - top)
            action_canvas.create_rectangle(
                x0,
                bottom,
                x1,
                y,
                fill=ACCENT_COLORS[idx % len(ACCENT_COLORS)],
                outline="",
            )
            action_canvas.create_text(center_x, bottom + 10, text=ACTION_NAMES[idx], fill=DARK_FG, font=("Consolas", 8))
            if pct > 0.0:
                action_canvas.create_text(center_x, y - 8, text=f"{pct:.1f}%", fill=DARK_FG, font=("Consolas", 8))

        label = metric_label if scope_key == "overall" else scope_key
        self._draw_percentage_bar_chart(vpip_canvas, [label], [selected_pct], f"{metric_label} {scope_label}")
        self._draw_heatmap_chart(hand_canvas, hand_grid, metric_label=metric_label, scope_label=scope_label)


if __name__ == "__main__":
    app = TrainingGUI()
    app.mainloop()
