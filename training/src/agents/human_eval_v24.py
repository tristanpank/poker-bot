
import argparse
import math
import os
import random
import sys
import time
import tkinter as tk
from types import SimpleNamespace
from tkinter import filedialog, messagebox, ttk
from typing import Dict, Optional

import numpy as np
import torch
from pokerkit import Automation, NoLimitTexasHoldem

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(SRC_ROOT)
FEATURES_DIR = os.path.join(SRC_ROOT, "features")
MODELS_DIR = os.path.join(SRC_ROOT, "models")
WORKERS_DIR = os.path.join(SRC_ROOT, "workers")
for _path in (FEATURES_DIR, MODELS_DIR, WORKERS_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from poker_model_v24 import PokerDeepCFRNet, load_compatible_state_dict
from poker_state_v24 import (
    ACTION_ALL_IN,
    ACTION_CALL,
    ACTION_CHECK,
    ACTION_COUNT_V21,
    ACTION_NAMES_V21,
    ALL_RAISE_ACTIONS,
    POSITION_NAMES_V21,
    STATE_DIM_V21,
    abstract_raise_target,
    build_legal_action_mask,
    encode_info_state,
    flatten_cards_list,
    street_from_board_len,
)
from poker_worker_v24 import (
    HandContext,
    _new_preflop_stats,
    _policy_action_for_snapshot,
    _record_postflop_action_stats,
    _record_preflop_action_stats,
    apply_abstract_action,
)

NUM_PLAYERS = 6
DEFAULT_BIG_BLIND = 10
DEFAULT_STACK_BB = 100.0
DEFAULT_HERO_SEAT = 5
DEFAULT_INTER_HAND_DELAY_MS = 850
DEFAULT_CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "models", "poker_agent_v24_deepcfr.pt")
ACTION_SHORT_NAMES = ["F", "X", "C", "R25", "R50", "R75", "R100", "R125", "R150", "R200", "AI"]
CARD_BACK = "?? ??"

COLOR_BG = "#0f1420"
COLOR_PANEL = "#1b2436"
COLOR_PANEL_ALT = "#151d2c"
COLOR_TEXT = "#dbe7ff"
COLOR_MUTED = "#90a4c4"
COLOR_ACCENT = "#67d4ff"
COLOR_HERO = "#5da9ff"
COLOR_CLONE = "#425b82"
COLOR_ACTIVE = "#ffcb6b"
COLOR_FOLDED = "#29354d"
COLOR_TABLE_OUTER = "#0f5a45"
COLOR_TABLE_INNER = "#187a5f"
COLOR_BOARD_BG = "#0d1220"

STREET_NAMES = {0: "Preflop", 1: "Flop", 2: "Turn", 3: "River"}


def _format_card(card) -> str:
    rank = getattr(card.rank, "value", str(card.rank))
    suit = getattr(card.suit, "value", str(card.suit))
    return f"{rank}{suit}"


def _format_cards(cards) -> str:
    flat = flatten_cards_list(cards)
    if not flat:
        return "-"
    return " ".join(_format_card(card) for card in flat)


def _seat_label(seat: int) -> str:
    return f"{POSITION_NAMES_V21[seat]} ({seat})"


def _sample_action(probs: np.ndarray, rng: random.Random) -> int:
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    total = float(probs.sum())
    if total <= 0.0:
        return int(np.argmax(probs))
    draw = rng.random() * total
    cumulative = 0.0
    for idx, value in enumerate(probs):
        cumulative += float(value)
        if draw <= cumulative:
            return int(idx)
    return int(np.argmax(probs))


def _is_raise_action(action_id: int) -> bool:
    return int(action_id) in ALL_RAISE_ACTIONS


def _state_dict_from_payload(payload: object) -> Dict[str, torch.Tensor]:
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload is not a dictionary.")
    if "model_state_dict" in payload and isinstance(payload["model_state_dict"], dict):
        return payload["model_state_dict"]
    if "state_dict" in payload and isinstance(payload["state_dict"], dict):
        return payload["state_dict"]
    if any(str(key).endswith("input_layer.weight") for key in payload.keys()):
        return payload
    raise ValueError("Could not find model state dict in checkpoint payload.")


def _config_from_payload(payload: object) -> Dict[str, object]:
    if isinstance(payload, dict):
        cfg = payload.get("config", {})
        if isinstance(cfg, dict):
            return cfg
    return {}


def _infer_model_dims(state_dict: Dict[str, torch.Tensor], config: Dict[str, object]) -> tuple[int, int, int]:
    hidden_dim = int(config.get("hidden_dim", 0) or 0)
    state_dim = int(config.get("state_dim", 0) or 0)
    action_dim = int(config.get("action_count", 0) or 0)

    input_weight = None
    regret_weight = None
    for key, tensor in state_dict.items():
        if input_weight is None and str(key).endswith("input_layer.weight"):
            input_weight = tensor
        if regret_weight is None and str(key).endswith("regret_head.2.weight"):
            regret_weight = tensor

    if input_weight is not None:
        hidden_dim = int(input_weight.shape[0])
        state_dim = int(input_weight.shape[1])
    if regret_weight is not None:
        action_dim = int(regret_weight.shape[0])

    hidden_dim = hidden_dim if hidden_dim > 0 else 256
    state_dim = state_dim if state_dim > 0 else STATE_DIM_V21
    action_dim = action_dim if action_dim > 0 else ACTION_COUNT_V21
    return state_dim, hidden_dim, action_dim


def _aligned_state_vector_for_model(model: Optional[PokerDeepCFRNet], state_vec: np.ndarray) -> np.ndarray:
    if model is None:
        return state_vec
    expected_dim = int(getattr(model, "state_dim", 0) or 0)
    if expected_dim <= 0:
        return state_vec
    actual_dim = int(state_vec.shape[0])
    if actual_dim == expected_dim:
        return state_vec
    if actual_dim > expected_dim:
        return state_vec[:expected_dim]
    aligned = np.zeros(expected_dim, dtype=np.float32)
    aligned[:actual_dim] = state_vec
    return aligned


def load_policy(path: str) -> tuple[PokerDeepCFRNet, Dict[str, object]]:
    payload = torch.load(path, map_location="cpu")
    state_dict = _state_dict_from_payload(payload)
    config = _config_from_payload(payload)
    _, hidden_dim, _ = _infer_model_dims(state_dict, config)
    model = PokerDeepCFRNet(
        state_dim=STATE_DIM_V21,
        hidden_dim=hidden_dim,
        action_dim=ACTION_COUNT_V21,
        init_weights=False,
    )
    load_compatible_state_dict(model, state_dict)
    model.eval()
    model.to("cpu")
    for param in model.parameters():
        param.requires_grad_(False)
    return model, config


def _runtime_policy_config(config: Optional[Dict[str, object]]) -> SimpleNamespace:
    payload = dict(config or {})
    defaults = {
        "exploit_prior_enabled": True,
        "exploit_prior_strength": 0.55,
        "exploit_min_confidence": 0.10,
        "exploit_only_preflop_unopened": False,
        "exploit_teacher_mix": 0.65,
        "opponent_profile_short_alpha": 0.25,
        "opponent_profile_long_alpha": 0.05,
        "opponent_profile_confidence_scale": 256.0,
    }
    defaults.update(payload)
    return SimpleNamespace(**defaults)


def _opponent_metric_specs() -> tuple:
    return (
        ("vpip", "vpip_counts", "preflop_opportunities"),
        ("pfr", "pfr_counts", "preflop_opportunities"),
        ("three_bet", "three_bet_counts", "faced_open_opportunities"),
        ("fold_to_open", "fold_vs_open_counts", "faced_open_opportunities"),
        ("fold_to_three_bet", "fold_vs_three_bet_counts", "faced_three_bet_opportunities"),
        ("call_open", "call_vs_open_counts", "faced_open_opportunities"),
        ("squeeze", "squeeze_counts", "squeeze_opportunities"),
        ("fold_to_cbet_flop", "fold_vs_cbet_flop_counts", "faced_cbet_flop_opportunities"),
        ("fold_to_cbet_turn", "fold_vs_cbet_turn_counts", "faced_cbet_turn_opportunities"),
        ("aggression", "aggression_counts", "aggression_opportunities"),
    )


def _new_opponent_profile_record() -> Dict[str, float]:
    record: Dict[str, float] = {"hands_played_total": 0.0}
    for metric_name, _, _ in _opponent_metric_specs():
        record[f"{metric_name}_hits_total"] = 0.0
        record[f"{metric_name}_opp_total"] = 0.0
        record[f"{metric_name}_short"] = 0.0
        record[f"{metric_name}_long"] = 0.0
    return record


def _profile_record(store: Dict[str, Dict[str, float]], player_id: str) -> Dict[str, float]:
    key = str(player_id or "unknown")
    record = store.get(key)
    if record is None:
        record = _new_opponent_profile_record()
        store[key] = record
    else:
        defaults = _new_opponent_profile_record()
        for metric_key, default_value in defaults.items():
            if metric_key not in record:
                record[metric_key] = float(default_value)
    return record


def _profile_tuple_for_player(store: Dict[str, Dict[str, float]], player_id: str, config) -> tuple:
    record = _profile_record(store, player_id)
    confidence_scale = float(max(1.0, getattr(config, "opponent_profile_confidence_scale", 256.0)))
    hands_played_total = float(max(0.0, record.get("hands_played_total", 0.0)))
    confidence = float(max(0.0, min(1.0, hands_played_total / confidence_scale)))
    trend_mix = float(max(0.0, min(1.0, confidence * 1.5)))

    values = []
    for metric_name, _, _ in _opponent_metric_specs():
        short_rate = float(max(0.0, min(1.0, record.get(f"{metric_name}_short", 0.0))))
        long_rate = float(max(0.0, min(1.0, record.get(f"{metric_name}_long", 0.0))))
        values.append(float(max(0.0, min(1.0, (trend_mix * short_rate) + ((1.0 - trend_mix) * long_rate)))))
    values.append(confidence)
    return tuple(values)


def _profiles_by_seat_for_actor(
    store: Dict[str, Dict[str, float]],
    seat_to_player_id: Dict[int, str],
    config,
) -> Dict[int, tuple]:
    return {
        int(seat): _profile_tuple_for_player(store, player_id, config)
        for seat, player_id in seat_to_player_id.items()
    }


def _update_profile_store_from_hand(
    store: Dict[str, Dict[str, float]],
    seat_to_player_id: Dict[int, str],
    hand_profile_stats: Dict[str, list],
    config,
) -> None:
    if not hand_profile_stats:
        return

    hands_played = hand_profile_stats.get("hands_played", [])
    short_alpha = float(max(0.0, min(1.0, getattr(config, "opponent_profile_short_alpha", 0.25))))
    long_alpha = float(max(0.0, min(1.0, getattr(config, "opponent_profile_long_alpha", 0.05))))

    for seat, player_id in seat_to_player_id.items():
        record = _profile_record(store, player_id)
        if seat < len(hands_played):
            record["hands_played_total"] += float(hands_played[seat])

        for metric_name, hit_key, opp_key in _opponent_metric_specs():
            hit_values = hand_profile_stats.get(hit_key, [])
            opp_values = hand_profile_stats.get(opp_key, [])
            hits = float(hit_values[seat]) if seat < len(hit_values) else 0.0
            opportunities = float(opp_values[seat]) if seat < len(opp_values) else 0.0
            record[f"{metric_name}_hits_total"] += hits
            record[f"{metric_name}_opp_total"] += opportunities
            if opportunities > 1e-9:
                observed_rate = float(max(0.0, min(1.0, hits / opportunities)))
                prev_short = float(record.get(f"{metric_name}_short", 0.0))
                prev_long = float(record.get(f"{metric_name}_long", 0.0))
                record[f"{metric_name}_short"] = ((1.0 - short_alpha) * prev_short) + (short_alpha * observed_rate)
                record[f"{metric_name}_long"] = ((1.0 - long_alpha) * prev_long) + (long_alpha * observed_rate)


class HumanEvalGUI(tk.Tk):
    def __init__(self, model_path: str, hero_seat: int, big_blind: int, stack_bb: float, seed: Optional[int]):
        super().__init__()
        self.title("Poker v24 Human Eval - Continuous Table Session")
        self.geometry("1440x940")
        self.configure(bg=COLOR_BG)

        self.rng = random.Random(seed if seed is not None else int(time.time() * 1_000_000))

        self.model_path_var = tk.StringVar(value=model_path)
        self.show_model_cards_var = tk.BooleanVar(value=False)
        self.auto_continue_var = tk.BooleanVar(value=True)
        self.rotate_hero_var = tk.BooleanVar(value=True)
        self.inter_hand_delay_var = tk.IntVar(value=DEFAULT_INTER_HAND_DELAY_MS)
        self.model_rebuy_mode_var = tk.StringVar(value="Below Threshold")
        self.model_rebuy_threshold_bb_var = tk.DoubleVar(value=20.0)
        self.model_rebuy_to_bb_var = tk.DoubleVar(value=max(20.0, float(stack_bb)))
        self.human_rebuy_mode_var = tk.StringVar(value="Below Threshold")
        self.human_rebuy_threshold_bb_var = tk.DoubleVar(value=20.0)
        self.human_rebuy_to_bb_var = tk.DoubleVar(value=max(20.0, float(stack_bb)))
        self.hero_start_seat_var = tk.StringVar(value=_seat_label(int(hero_seat) % NUM_PLAYERS))
        self.big_blind_var = tk.IntVar(value=max(2, int(big_blind)))
        self.stack_bb_var = tk.DoubleVar(value=max(20.0, float(stack_bb)))

        self.hero_start_seat = int(hero_seat) % NUM_PLAYERS
        self.current_hero_seat = self.hero_start_seat

        self.base_model: Optional[PokerDeepCFRNet] = None
        self.model_runtime_config = _runtime_policy_config({})
        self.seat_models: Dict[int, PokerDeepCFRNet] = {}
        self.clone_name_by_seat: Dict[int, str] = {}
        self.clone_id_by_seat: Dict[int, int] = {}
        self.player_id_by_seat: Dict[int, str] = {}
        self.opponent_profile_store: Dict[str, Dict[str, float]] = {}
        default_stack_chips = int(round(float(stack_bb) * float(big_blind)))
        self.hero_bankroll_chips = default_stack_chips
        self.clone_bankroll_chips = [default_stack_chips] * (NUM_PLAYERS - 1)

        self.state = None
        self.hand_ctx: Optional[HandContext] = None
        self.hand_active = False
        self.awaiting_hero_action = False
        self.session_running = False
        self.hand_index = 0
        self.last_action_by_seat = {seat: "-" for seat in range(NUM_PLAYERS)}

        self.hand_preflop_opp = [False] * NUM_PLAYERS
        self.hand_vpip = [False] * NUM_PLAYERS
        self.hand_pfr = [False] * NUM_PLAYERS
        self.hand_three_bet = [False] * NUM_PLAYERS
        self.hand_profile_stats = _new_preflop_stats(NUM_PLAYERS)
        self.hand_profile_stats = _new_preflop_stats(NUM_PLAYERS)

        self.stats = {
            "hands": 0,
            "hero_profit_bb": 0.0,
            "hero_wins": 0,
            "hero_actions": np.zeros(ACTION_COUNT_V21, dtype=np.int64),
            "model_actions": np.zeros(ACTION_COUNT_V21, dtype=np.int64),
            "hero_illegal_actions": 0,
            "model_illegal_actions": 0,
            "hero_preflop_opp": 0,
            "hero_vpip": 0,
            "hero_pfr": 0,
            "hero_three_bet": 0,
            "model_preflop_opp": 0,
            "model_vpip": 0,
            "model_pfr": 0,
            "model_three_bet": 0,
        }

        self.status_var = tk.StringVar(value="Load a model, then start a session.")
        self.session_var = tk.StringVar(value="")
        self.hero_style_var = tk.StringVar(value="")
        self.model_style_var = tk.StringVar(value="")
        self.action_mix_var = tk.StringVar(value="")
        self.illegal_var = tk.StringVar(value="")
        self.exploit_read_var = tk.StringVar(value="")
        self.last_exploit_actor: Optional[int] = None
        self.last_exploit_details: Optional[Dict[str, object]] = None

        self.table_canvas: Optional[tk.Canvas] = None
        self.action_buttons: Dict[int, ttk.Button] = {}
        self.custom_raise_button: Optional[ttk.Button] = None
        self.raise_to_bb_var = tk.StringVar(value="")
        self.raise_hint_var = tk.StringVar(value="Raise-to sizing appears when it is your turn.")
        self.log_text: Optional[tk.Text] = None

        self._configure_styles()
        self._build_layout()
        self._reset_stats_labels()
        self._load_model(initial=True)

    @property
    def hero_seat(self) -> int:
        return int(self.current_hero_seat) % NUM_PLAYERS

    @property
    def big_blind(self) -> int:
        try:
            return max(2, int(self.big_blind_var.get()))
        except (TypeError, ValueError, tk.TclError):
            return DEFAULT_BIG_BLIND

    @property
    def small_blind(self) -> int:
        return max(1, self.big_blind // 2)

    @property
    def stack_bb(self) -> float:
        try:
            return max(20.0, float(self.stack_bb_var.get()))
        except (TypeError, ValueError, tk.TclError):
            return DEFAULT_STACK_BB

    def _configure_styles(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")

        self.option_add("*Font", "SegoeUI 10")

        style.configure("Root.TFrame", background=COLOR_BG)
        style.configure("Panel.TLabelframe", background=COLOR_PANEL, foreground=COLOR_ACCENT)
        style.configure("Panel.TLabelframe.Label", background=COLOR_PANEL, foreground=COLOR_ACCENT)
        style.configure("Panel.TFrame", background=COLOR_PANEL)
        style.configure("PanelAlt.TFrame", background=COLOR_PANEL_ALT)

        style.configure("TLabel", background=COLOR_PANEL, foreground=COLOR_TEXT)
        style.configure("Panel.TLabel", background=COLOR_PANEL, foreground=COLOR_TEXT)
        style.configure("Muted.TLabel", background=COLOR_PANEL, foreground=COLOR_MUTED)
        style.configure("Title.TLabel", background=COLOR_BG, foreground=COLOR_ACCENT, font=("Segoe UI", 11, "bold"))

        style.configure("TCheckbutton", background=COLOR_PANEL, foreground=COLOR_TEXT)
        style.configure("TCombobox", fieldbackground="#111827", background="#111827", foreground=COLOR_TEXT)
        style.configure("TSpinbox", fieldbackground="#111827", foreground=COLOR_TEXT)
        style.configure("TEntry", fieldbackground="#111827", foreground=COLOR_TEXT)

        style.configure(
            "Primary.TButton",
            background="#1f7ae0",
            foreground="#ffffff",
            bordercolor="#1f7ae0",
            padding=(8, 6),
        )
        style.map("Primary.TButton", background=[("active", "#2a8dff")])

        style.configure(
            "Warn.TButton",
            background="#ad3f37",
            foreground="#ffffff",
            bordercolor="#ad3f37",
            padding=(8, 6),
        )
        style.map("Warn.TButton", background=[("active", "#c74d44")])

    def _build_layout(self) -> None:
        root = ttk.Frame(self, style="Root.TFrame")
        root.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        top = ttk.LabelFrame(root, text="Session Setup", style="Panel.TLabelframe", padding=10)
        top.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(top, text="Model", style="Panel.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.model_path_var, width=70).grid(row=0, column=1, columnspan=6, sticky="ew", padx=(8, 8))
        ttk.Button(top, text="Browse", command=self._browse_model).grid(row=0, column=7, padx=(0, 6))
        ttk.Button(top, text="Load", style="Primary.TButton", command=self._load_model).grid(row=0, column=8)

        ttk.Label(top, text="Start Seat", style="Panel.TLabel").grid(row=1, column=0, sticky="w", pady=(8, 0))
        seat_box = ttk.Combobox(
            top,
            values=[_seat_label(seat) for seat in range(NUM_PLAYERS)],
            textvariable=self.hero_start_seat_var,
            state="readonly",
            width=12,
        )
        seat_box.grid(row=1, column=1, sticky="w", padx=(8, 10), pady=(8, 0))
        seat_box.bind("<<ComboboxSelected>>", self._on_start_seat_change)

        ttk.Label(top, text="Big Blind", style="Panel.TLabel").grid(row=1, column=2, sticky="w", pady=(8, 0))
        ttk.Spinbox(top, from_=2, to=1000, textvariable=self.big_blind_var, width=8).grid(
            row=1, column=3, sticky="w", padx=(8, 10), pady=(8, 0)
        )
        ttk.Label(top, text="Stack (BB)", style="Panel.TLabel").grid(row=1, column=4, sticky="w", pady=(8, 0))
        ttk.Spinbox(top, from_=20, to=1000, increment=5, textvariable=self.stack_bb_var, width=8).grid(
            row=1, column=5, sticky="w", padx=(8, 10), pady=(8, 0)
        )
        ttk.Label(top, text="Inter-Hand ms", style="Panel.TLabel").grid(row=1, column=6, sticky="w", pady=(8, 0))
        ttk.Spinbox(top, from_=100, to=5000, increment=50, textvariable=self.inter_hand_delay_var, width=8).grid(
            row=1, column=7, sticky="w", padx=(8, 6), pady=(8, 0)
        )

        ttk.Checkbutton(top, text="Show Model Cards", variable=self.show_model_cards_var, command=self._refresh_view).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )
        ttk.Checkbutton(top, text="Auto Continue", variable=self.auto_continue_var).grid(
            row=2, column=2, columnspan=2, sticky="w", pady=(8, 0)
        )
        ttk.Checkbutton(top, text="Rotate Hero Seat", variable=self.rotate_hero_var).grid(
            row=2, column=4, columnspan=2, sticky="w", pady=(8, 0)
        )

        ttk.Button(top, text="Start Session", style="Primary.TButton", command=self._start_session).grid(
            row=2, column=6, pady=(8, 0), padx=(4, 4)
        )
        ttk.Button(top, text="Stop Session", style="Warn.TButton", command=self._stop_session).grid(
            row=2, column=7, pady=(8, 0), padx=(4, 4)
        )
        ttk.Button(top, text="Play One Hand", command=self._play_one_hand).grid(row=2, column=8, pady=(8, 0), padx=(4, 0))

        ttk.Label(top, text="Model Rebuy", style="Panel.TLabel").grid(row=3, column=0, sticky="w", pady=(8, 0))
        rebuy_mode_box = ttk.Combobox(
            top,
            values=["Every hand", "Below Threshold", "Never"],
            textvariable=self.model_rebuy_mode_var,
            state="readonly",
            width=14,
        )
        rebuy_mode_box.grid(row=3, column=1, sticky="w", padx=(8, 10), pady=(8, 0))
        ttk.Label(top, text="Threshold (BB)", style="Panel.TLabel").grid(row=3, column=2, sticky="w", pady=(8, 0))
        ttk.Spinbox(top, from_=0, to=500, increment=1, textvariable=self.model_rebuy_threshold_bb_var, width=8).grid(
            row=3, column=3, sticky="w", padx=(8, 10), pady=(8, 0)
        )
        ttk.Label(top, text="Rebuy To (BB)", style="Panel.TLabel").grid(row=3, column=4, sticky="w", pady=(8, 0))
        ttk.Spinbox(top, from_=1, to=1000, increment=1, textvariable=self.model_rebuy_to_bb_var, width=8).grid(
            row=3, column=5, sticky="w", padx=(8, 10), pady=(8, 0)
        )
        ttk.Label(
            top,
            text="Applied to the 5 model clones at hand start.",
            style="Muted.TLabel",
        ).grid(row=3, column=6, columnspan=3, sticky="w", pady=(8, 0))

        ttk.Label(top, text="Human Rebuy", style="Panel.TLabel").grid(row=4, column=0, sticky="w", pady=(6, 0))
        human_rebuy_mode_box = ttk.Combobox(
            top,
            values=["Every hand", "Below Threshold", "Never"],
            textvariable=self.human_rebuy_mode_var,
            state="readonly",
            width=14,
        )
        human_rebuy_mode_box.grid(row=4, column=1, sticky="w", padx=(8, 10), pady=(6, 0))
        ttk.Label(top, text="Threshold (BB)", style="Panel.TLabel").grid(row=4, column=2, sticky="w", pady=(6, 0))
        ttk.Spinbox(top, from_=0, to=500, increment=1, textvariable=self.human_rebuy_threshold_bb_var, width=8).grid(
            row=4, column=3, sticky="w", padx=(8, 10), pady=(6, 0)
        )
        ttk.Label(top, text="Rebuy To (BB)", style="Panel.TLabel").grid(row=4, column=4, sticky="w", pady=(6, 0))
        ttk.Spinbox(top, from_=1, to=1000, increment=1, textvariable=self.human_rebuy_to_bb_var, width=8).grid(
            row=4, column=5, sticky="w", padx=(8, 10), pady=(6, 0)
        )
        ttk.Label(
            top,
            text="Applied to your bankroll at hand start.",
            style="Muted.TLabel",
        ).grid(row=4, column=6, columnspan=3, sticky="w", pady=(6, 0))

        top.columnconfigure(1, weight=1)

        status = ttk.Frame(root, style="Root.TFrame")
        status.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(status, textvariable=self.status_var, style="Title.TLabel").pack(side=tk.LEFT)

        body = ttk.Frame(root, style="Root.TFrame")
        body.pack(fill=tk.BOTH, expand=True)

        table_frame = ttk.LabelFrame(body, text="Table", style="Panel.TLabelframe", padding=8)
        table_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        self.table_canvas = tk.Canvas(
            table_frame,
            bg=COLOR_BG,
            highlightthickness=0,
            borderwidth=0,
            width=940,
            height=560,
        )
        self.table_canvas.pack(fill=tk.BOTH, expand=True)
        self.table_canvas.bind("<Configure>", lambda _e: self._draw_table_scene())

        side = ttk.Frame(body, style="Panel.TFrame")
        side.pack(side=tk.RIGHT, fill=tk.Y)

        action_frame = ttk.LabelFrame(side, text="Hero Actions", style="Panel.TLabelframe", padding=8)
        action_frame.pack(fill=tk.X, pady=(0, 8))
        for action_id, action_name in enumerate(ACTION_NAMES_V21):
            button = ttk.Button(
                action_frame,
                text=action_name,
                command=lambda a=action_id: self._on_hero_action(a),
                state=tk.DISABLED,
                width=22,
            )
            button.pack(fill=tk.X, pady=2)
            self.action_buttons[action_id] = button

        ttk.Separator(action_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(8, 6))
        ttk.Label(action_frame, text="Custom Raise To (BB)", style="Panel.TLabel").pack(anchor="w")
        ttk.Entry(action_frame, textvariable=self.raise_to_bb_var, width=12).pack(fill=tk.X, pady=(2, 4))
        self.custom_raise_button = ttk.Button(
            action_frame,
            text="Raise To Size",
            command=self._on_custom_raise,
            state=tk.DISABLED,
        )
        self.custom_raise_button.pack(fill=tk.X)
        ttk.Label(
            action_frame,
            textvariable=self.raise_hint_var,
            style="Muted.TLabel",
            justify=tk.LEFT,
            wraplength=240,
        ).pack(anchor="w", pady=(6, 0))

        stats_frame = ttk.LabelFrame(side, text="Statistics", style="Panel.TLabelframe", padding=8)
        stats_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(stats_frame, textvariable=self.session_var, justify=tk.LEFT, style="Panel.TLabel", font=("Consolas", 10)).pack(
            anchor="w", pady=(0, 4)
        )
        ttk.Label(stats_frame, textvariable=self.hero_style_var, justify=tk.LEFT, style="Panel.TLabel", font=("Consolas", 10)).pack(
            anchor="w", pady=(0, 4)
        )
        ttk.Label(stats_frame, textvariable=self.model_style_var, justify=tk.LEFT, style="Panel.TLabel", font=("Consolas", 10)).pack(
            anchor="w", pady=(0, 4)
        )
        ttk.Label(stats_frame, textvariable=self.action_mix_var, justify=tk.LEFT, style="Panel.TLabel", font=("Consolas", 10)).pack(
            anchor="w", pady=(0, 4)
        )
        ttk.Label(stats_frame, textvariable=self.illegal_var, justify=tk.LEFT, style="Panel.TLabel", font=("Consolas", 10)).pack(
            anchor="w"
        )
        ttk.Label(
            stats_frame,
            textvariable=self.exploit_read_var,
            justify=tk.LEFT,
            style="Panel.TLabel",
            font=("Consolas", 10),
            wraplength=300,
        ).pack(anchor="w", pady=(6, 0))

        log_frame = ttk.LabelFrame(root, text="Hand Log", style="Panel.TLabelframe", padding=8)
        log_frame.pack(fill=tk.BOTH, pady=(8, 0))
        self.log_text = tk.Text(
            log_frame,
            height=10,
            wrap=tk.WORD,
            state=tk.DISABLED,
            bg="#0c1220",
            fg=COLOR_TEXT,
            insertbackground=COLOR_TEXT,
            relief=tk.FLAT,
            font=("Consolas", 10),
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _append_log(self, text: str) -> None:
        if self.log_text is None:
            return
        stamp = time.strftime("%H:%M:%S")
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"[{stamp}] {text}\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _browse_model(self) -> None:
        path = filedialog.askopenfilename(
            title="Select v24 Checkpoint",
            filetypes=[("PyTorch Checkpoint", "*.pt *.pth"), ("All Files", "*.*")],
            initialdir=os.path.dirname(self.model_path_var.get() or DEFAULT_CHECKPOINT_PATH),
        )
        if path:
            self.model_path_var.set(path)

    def _load_model(self, initial: bool = False) -> None:
        path = self.model_path_var.get().strip()
        if not path:
            if not initial:
                messagebox.showerror("Model Error", "Model path is empty.")
            return
        if not os.path.isfile(path):
            msg = f"Model file not found:\n{path}"
            if initial:
                self.status_var.set(msg)
                return
            messagebox.showerror("Model Error", msg)
            return
        try:
            self.base_model, model_config = load_policy(path)
            self.model_runtime_config = _runtime_policy_config(model_config)
            self._rebuild_clone_lineup()
            self.status_var.set(f"Loaded model: {path}")
            self._append_log(f"Loaded v24 model from {path}")
            self._refresh_view()
        except Exception as exc:
            self.base_model = None
            self.model_runtime_config = _runtime_policy_config({})
            self.seat_models = {}
            self.clone_name_by_seat = {}
            msg = f"Failed to load model:\n{exc}"
            if initial:
                self.status_var.set(msg)
            else:
                messagebox.showerror("Model Error", msg)

    def _parse_selected_start_seat(self) -> int:
        label = self.hero_start_seat_var.get()
        if "(" in label and label.endswith(")"):
            try:
                return int(label.split("(")[-1].rstrip(")")) % NUM_PLAYERS
            except ValueError:
                return DEFAULT_HERO_SEAT
        return DEFAULT_HERO_SEAT

    def _on_start_seat_change(self, _event=None) -> None:
        self.hero_start_seat = self._parse_selected_start_seat()
        if self.hand_active:
            self.status_var.set("Start seat updated. It will apply after current hand.")
            return
        self.current_hero_seat = self.hero_start_seat
        self._rebuild_clone_lineup()
        self._refresh_view()

    def _rebuild_clone_lineup(self) -> None:
        self.seat_models = {}
        self.clone_name_by_seat = {}
        self.clone_id_by_seat = {}
        self.player_id_by_seat = {self.hero_seat: "human"}
        if self.base_model is None:
            return
        opponent_seats = [seat for seat in range(NUM_PLAYERS) if seat != self.hero_seat]
        for clone_idx, seat in enumerate(opponent_seats):
            self.seat_models[seat] = self.base_model.clone_cpu()
            self.clone_name_by_seat[seat] = f"Clone {clone_idx + 1}"
            self.clone_id_by_seat[seat] = clone_idx
            self.player_id_by_seat[seat] = f"clone_{clone_idx + 1}"

    def _reset_clone_bankrolls(self) -> None:
        try:
            rebuy_to_bb = max(1.0, float(self.model_rebuy_to_bb_var.get()))
        except (TypeError, ValueError, tk.TclError):
            rebuy_to_bb = max(1.0, float(self.stack_bb))
        rebuy_to_chips = int(round(rebuy_to_bb * float(self.big_blind)))
        self.clone_bankroll_chips = [rebuy_to_chips] * (NUM_PLAYERS - 1)

    def _reset_human_bankroll(self) -> None:
        try:
            rebuy_to_bb = max(1.0, float(self.human_rebuy_to_bb_var.get()))
        except (TypeError, ValueError, tk.TclError):
            rebuy_to_bb = max(1.0, float(self.stack_bb))
        rebuy_to_chips = int(round(rebuy_to_bb * float(self.big_blind)))
        self.hero_bankroll_chips = rebuy_to_chips

    def _reset_all_bankrolls(self) -> None:
        self._reset_human_bankroll()
        self._reset_clone_bankrolls()

    def _apply_rebuy_policy_to_human_bankroll(self) -> None:
        rebuy_mode = self.human_rebuy_mode_var.get().strip()
        try:
            rebuy_to_bb = max(1.0, float(self.human_rebuy_to_bb_var.get()))
        except (TypeError, ValueError, tk.TclError):
            rebuy_to_bb = max(1.0, float(self.stack_bb))
        rebuy_to_chips = int(round(rebuy_to_bb * float(self.big_blind)))

        if rebuy_mode == "Every hand":
            self.hero_bankroll_chips = rebuy_to_chips
            return

        if rebuy_mode == "Below Threshold":
            try:
                threshold_bb = max(0.0, float(self.human_rebuy_threshold_bb_var.get()))
            except (TypeError, ValueError, tk.TclError):
                threshold_bb = 0.0
            threshold_chips = int(round(threshold_bb * float(self.big_blind)))
            if int(self.hero_bankroll_chips) <= threshold_chips:
                self.hero_bankroll_chips = rebuy_to_chips
            return

        if rebuy_mode == "Never":
            return

        self.hero_bankroll_chips = rebuy_to_chips

    def _apply_rebuy_policy_to_clone_bankrolls(self) -> None:
        rebuy_mode = self.model_rebuy_mode_var.get().strip()
        try:
            rebuy_to_bb = max(1.0, float(self.model_rebuy_to_bb_var.get()))
        except (TypeError, ValueError, tk.TclError):
            rebuy_to_bb = max(1.0, float(self.stack_bb))
        rebuy_to_chips = int(round(rebuy_to_bb * float(self.big_blind)))

        if rebuy_mode == "Every hand":
            self.clone_bankroll_chips = [rebuy_to_chips] * len(self.clone_bankroll_chips)
            return

        if rebuy_mode == "Below Threshold":
            try:
                threshold_bb = max(0.0, float(self.model_rebuy_threshold_bb_var.get()))
            except (TypeError, ValueError, tk.TclError):
                threshold_bb = 0.0
            threshold_chips = int(round(threshold_bb * float(self.big_blind)))
            for idx, stack_chips in enumerate(self.clone_bankroll_chips):
                if stack_chips <= threshold_chips:
                    self.clone_bankroll_chips[idx] = rebuy_to_chips
            return

        if rebuy_mode == "Never":
            return

        self.clone_bankroll_chips = [rebuy_to_chips] * len(self.clone_bankroll_chips)

    def _get_raise_bounds(self) -> tuple[int, int]:
        if self.state is None:
            return 0, 0
        min_raise = getattr(self.state, "min_completion_betting_or_raising_to_amount", None)
        max_raise = getattr(self.state, "max_completion_betting_or_raising_to_amount", None)
        if min_raise is None:
            min_raise = getattr(self.state, "min_completion_betting_or_raising_to", 0)
        if max_raise is None:
            max_raise = getattr(self.state, "max_completion_betting_or_raising_to", 0)
        return int(min_raise or 0), int(max_raise or 0)

    def _start_session(self) -> None:
        if self.base_model is None:
            messagebox.showerror("Model Error", "Load a valid v24 model first.")
            return
        if self.hand_active:
            self.session_running = True
            self.status_var.set("Session is already running.")
            return

        self.hero_start_seat = self._parse_selected_start_seat()
        self.current_hero_seat = self.hero_start_seat
        self._reset_all_bankrolls()
        self.session_running = True
        self._append_log(f"Session started. Hero start seat: {_seat_label(self.current_hero_seat)}")
        self._start_new_hand(is_continuation=False)

    def _stop_session(self) -> None:
        if not self.session_running and not self.hand_active:
            self.status_var.set("Session stopped.")
            return
        self.session_running = False
        if self.hand_active:
            self.status_var.set("Session stop requested. Current hand will finish.")
        else:
            self.status_var.set("Session stopped.")
        self._append_log("Session stop requested.")

    def _play_one_hand(self) -> None:
        if self.base_model is None:
            messagebox.showerror("Model Error", "Load a valid v24 model first.")
            return
        if self.hand_active:
            messagebox.showwarning("Hand In Progress", "Finish the current hand before starting another.")
            return
        self.session_running = False
        self.hero_start_seat = self._parse_selected_start_seat()
        self.current_hero_seat = self.hero_start_seat
        self._reset_all_bankrolls()
        self._start_new_hand(is_continuation=False)

    def _create_state_and_context(self):
        self._apply_rebuy_policy_to_human_bankroll()
        self._apply_rebuy_policy_to_clone_bankrolls()
        hero_rebuy_mode = self.human_rebuy_mode_var.get().strip()
        rebuy_mode = self.model_rebuy_mode_var.get().strip()

        if hero_rebuy_mode == "Never" and int(self.hero_bankroll_chips) <= 0:
            raise ValueError("You are busted and human rebuy is set to Never.")

        if rebuy_mode == "Never":
            busted = [idx + 1 for idx, chips in enumerate(self.clone_bankroll_chips) if chips <= 0]
            if busted:
                raise ValueError(
                    f"Clone(s) {', '.join(str(value) for value in busted)} are busted and model rebuy is set to Never."
                )

        hero_stack_chips = max(1, int(self.hero_bankroll_chips))
        stacks = [hero_stack_chips] * NUM_PLAYERS
        for seat, clone_id in self.clone_id_by_seat.items():
            chips = int(self.clone_bankroll_chips[clone_id]) if clone_id < len(self.clone_bankroll_chips) else hero_stack_chips
            stacks[seat] = max(1, chips)
        state = NoLimitTexasHoldem.create_state(
            automations=(
                Automation.ANTE_POSTING,
                Automation.BET_COLLECTION,
                Automation.BLIND_OR_STRADDLE_POSTING,
                Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
                Automation.HAND_KILLING,
                Automation.CHIPS_PUSHING,
                Automation.CHIPS_PULLING,
                Automation.CARD_BURNING,
            ),
            ante_trimming_status=True,
            raw_antes={-1: 0},
            raw_blinds_or_straddles=(self.small_blind, self.big_blind),
            min_bet=self.big_blind,
            raw_starting_stacks=stacks,
            player_count=NUM_PLAYERS,
        )
        while state.can_deal_hole():
            state.deal_hole()

        contributions = [float(start - stack) for start, stack in zip(stacks, state.stacks)]
        hand_ctx = HandContext(
            starting_stacks=list(stacks),
            big_blind=self.big_blind,
            small_blind=self.small_blind,
            in_hand=[True] * NUM_PLAYERS,
            contributions=contributions,
        )
        return state, hand_ctx

    def _start_new_hand(self, is_continuation: bool) -> None:
        if self.base_model is None or self.hand_active:
            return

        if is_continuation and self.rotate_hero_var.get():
            self.current_hero_seat = (self.current_hero_seat + 1) % NUM_PLAYERS

        self._rebuild_clone_lineup()
        try:
            self.state, self.hand_ctx = self._create_state_and_context()
        except Exception as exc:
            self.hand_active = False
            self.awaiting_hero_action = False
            self.session_running = False
            self.status_var.set(f"Unable to start hand: {exc}")
            self._append_log(f"Session halted: {exc}")
            self._refresh_view()
            return
        self.hand_active = True
        self.awaiting_hero_action = False
        self.hand_index += 1
        self.last_action_by_seat = {seat: "-" for seat in range(NUM_PLAYERS)}
        self.hand_preflop_opp = [False] * NUM_PLAYERS
        self.hand_vpip = [False] * NUM_PLAYERS
        self.hand_pfr = [False] * NUM_PLAYERS
        self.hand_three_bet = [False] * NUM_PLAYERS

        hero_cards = _format_cards(self.state.hole_cards[self.hero_seat])
        self.status_var.set(f"Hand #{self.hand_index} started. Hero: {_seat_label(self.hero_seat)}")
        self._append_log(
            f"Hand #{self.hand_index} | Hero seat {_seat_label(self.hero_seat)} | Hero cards: {hero_cards}"
        )
        self._refresh_view()
        self._play_until_hero_or_hand_end()

    def _schedule_next_hand_if_needed(self) -> None:
        if not self.session_running:
            return
        if not self.auto_continue_var.get():
            self.session_running = False
            self.status_var.set("Auto-continue disabled. Session paused.")
            return
        delay_ms = max(100, int(self.inter_hand_delay_var.get()))
        self.after(delay_ms, self._try_start_next_hand)

    def _try_start_next_hand(self) -> None:
        if not self.session_running or self.hand_active:
            return
        self._start_new_hand(is_continuation=True)

    def _advance_chance_nodes(self) -> None:
        if self.state is None or self.hand_ctx is None:
            return
        dealt = False
        while self.state.status and self.state.can_deal_board():
            self.state.deal_board()
            board = flatten_cards_list(self.state.board_cards)
            self.hand_ctx.current_street = street_from_board_len(len(board))
            self.hand_ctx.street_raise_count = 0
            if self.hand_ctx.current_street == 1:
                self.hand_ctx.cbet_flop_initiator = None
                self.hand_ctx.cbet_turn_initiator = None
            elif self.hand_ctx.current_street == 2:
                self.hand_ctx.cbet_turn_initiator = None
            dealt = True
        if dealt:
            self._append_log(f"Board: {_format_cards(self.state.board_cards)}")

    def _policy_action_for_actor(self, actor: int) -> int:
        if self.state is None or self.hand_ctx is None:
            return ACTION_CHECK
        model = self.seat_models.get(actor)
        if model is None:
            return ACTION_CHECK
        try:
            opponent_profiles_by_seat = _profiles_by_seat_for_actor(
                self.opponent_profile_store,
                self.player_id_by_seat,
                self.model_runtime_config,
            )
            action, details = _policy_action_for_snapshot(
                model,
                self.state,
                actor,
                self.hand_ctx,
                self.rng,
                opponent_profiles_by_seat=opponent_profiles_by_seat,
                config=self.model_runtime_config,
                return_details=True,
            )
            self._refresh_exploit_readout(actor=actor, details=details)
            return int(action)
        except Exception:
            legal_mask = build_legal_action_mask(self.state, actor, self.hand_ctx)
            legal_actions = [idx for idx, value in enumerate(legal_mask) if value > 0.5]
            if legal_actions:
                return int(legal_actions[0])
            return ACTION_CHECK

    def _record_preflop_flag(self, actor: int, action_id: int) -> None:
        if self.hand_ctx is None or self.hand_ctx.current_street != 0:
            return

        self.hand_preflop_opp[actor] = True
        prior_raises = int(self.hand_ctx.preflop_raise_count)
        if action_id == ACTION_CALL or _is_raise_action(action_id):
            self.hand_vpip[actor] = True
        if _is_raise_action(action_id):
            self.hand_pfr[actor] = True
            if prior_raises >= 1:
                self.hand_three_bet[actor] = True

    def _apply_action(self, actor: int, action_id: int) -> None:
        if self.state is None or self.hand_ctx is None:
            return

        self._record_preflop_flag(actor, action_id)
        to_call = float(max(self.state.bets) - self.state.bets[actor])
        if self.hand_ctx.current_street == 0:
            _record_preflop_action_stats(
                self.hand_profile_stats,
                actor,
                self.hand_ctx,
                to_call,
                int(self.hand_ctx.preflop_raise_count),
                action_id,
            )
        else:
            _record_postflop_action_stats(self.hand_profile_stats, actor, self.hand_ctx, to_call, action_id)
        if actor == self.hero_seat:
            self.stats["hero_actions"][action_id] += 1
        else:
            self.stats["model_actions"][action_id] += 1

        valid = apply_abstract_action(self.state, actor, action_id, self.hand_ctx)
        actor_name = "You" if actor == self.hero_seat else self.clone_name_by_seat.get(actor, f"Clone {actor}")
        action_name = ACTION_NAMES_V21[action_id]
        suffix = "" if valid else " (fallback)"
        self.last_action_by_seat[actor] = action_name + suffix
        self._append_log(f"{actor_name} -> {action_name}{suffix}")

        if not valid:
            if actor == self.hero_seat:
                self.stats["hero_illegal_actions"] += 1
            else:
                self.stats["model_illegal_actions"] += 1

    def _apply_custom_raise(self, actor: int, target_to_chips: int) -> bool:
        if self.state is None or self.hand_ctx is None:
            return False

        legal_mask = build_legal_action_mask(self.state, actor, self.hand_ctx)
        can_raise = any(legal_mask[action_id] > 0.5 for action_id in ALL_RAISE_ACTIONS)
        if not can_raise:
            return False

        min_raise, max_raise = self._get_raise_bounds()
        if target_to_chips < min_raise or target_to_chips > max_raise:
            return False

        before_stack = float(self.state.stacks[actor])
        before_bet = float(self.state.bets[actor])
        to_call = float(max(self.state.bets) - self.state.bets[actor])
        action_id = ACTION_ALL_IN if int(target_to_chips) >= int(max_raise) else int(ALL_RAISE_ACTIONS[0])
        closest_gap = None
        for candidate in ALL_RAISE_ACTIONS:
            candidate_target = abstract_raise_target(self.state, candidate)
            if candidate_target is None:
                continue
            gap = abs(int(candidate_target) - int(target_to_chips))
            if closest_gap is None or gap < closest_gap:
                closest_gap = gap
                action_id = int(candidate)

        try:
            self.state.complete_bet_or_raise_to(int(target_to_chips))
        except Exception:
            return False

        self._record_preflop_flag(actor, action_id)
        if self.hand_ctx.current_street == 0:
            _record_preflop_action_stats(
                self.hand_profile_stats,
                actor,
                self.hand_ctx,
                to_call,
                int(self.hand_ctx.preflop_raise_count),
                action_id,
            )
        else:
            _record_postflop_action_stats(
                self.hand_profile_stats,
                actor,
                self.hand_ctx,
                to_call,
                action_id,
            )
        if actor == self.hero_seat:
            self.stats["hero_actions"][action_id] += 1
        else:
            self.stats["model_actions"][action_id] += 1

        invested = max(0.0, before_stack - float(self.state.stacks[actor]))
        self.hand_ctx.contributions[actor] += invested
        self.hand_ctx.last_aggressor = actor
        self.hand_ctx.street_raise_count += 1
        if self.hand_ctx.current_street == 0:
            self.hand_ctx.preflop_raise_count += 1
            self.hand_ctx.preflop_opened = True
            self.hand_ctx.preflop_last_raiser = actor
        raise_delta_bb = max(0.0, (float(self.state.bets[actor]) - before_bet) / float(self.big_blind))
        self.hand_ctx.last_aggressive_size_bb = raise_delta_bb

        action_name = f"Raise To {float(self.state.bets[actor]) / float(self.big_blind):.2f} BB"
        actor_name = "You" if actor == self.hero_seat else self.clone_name_by_seat.get(actor, f"Clone {actor}")
        self.last_action_by_seat[actor] = action_name
        self._append_log(f"{actor_name} -> {action_name}")
        return True

    def _play_until_hero_or_hand_end(self) -> None:
        if not self.hand_active or self.state is None or self.hand_ctx is None:
            return

        while self.state.status:
            self._advance_chance_nodes()
            if not self.state.status:
                break

            actor = self.state.actor_index
            if actor is None:
                break

            if actor == self.hero_seat:
                self.awaiting_hero_action = True
                self.status_var.set(f"Your turn ({_seat_label(self.hero_seat)})")
                self._refresh_view()
                self._set_action_buttons()
                return

            action = self._policy_action_for_actor(actor)
            self._apply_action(actor, action)
            self._refresh_view()

        self._finish_hand()

    def _on_hero_action(self, action_id: int) -> None:
        if not self.hand_active or not self.awaiting_hero_action:
            return
        if self.state is None or self.hand_ctx is None:
            return

        legal_mask = build_legal_action_mask(self.state, self.hero_seat, self.hand_ctx)
        if action_id < 0 or action_id >= len(legal_mask) or legal_mask[action_id] <= 0.5:
            self.status_var.set("That action is not legal right now.")
            self._set_action_buttons()
            return

        self.awaiting_hero_action = False
        self._apply_action(self.hero_seat, action_id)
        self._disable_action_buttons()
        self._refresh_view()
        self.after(5, self._play_until_hero_or_hand_end)

    def _on_custom_raise(self) -> None:
        if not self.hand_active or not self.awaiting_hero_action:
            return
        if self.state is None or self.hand_ctx is None:
            return

        legal_mask = build_legal_action_mask(self.state, self.hero_seat, self.hand_ctx)
        can_raise = any(legal_mask[action_id] > 0.5 for action_id in ALL_RAISE_ACTIONS)
        if not can_raise:
            self.status_var.set("Raise is not legal in this state.")
            self._set_action_buttons()
            return

        min_raise, max_raise = self._get_raise_bounds()
        min_bb = float(min_raise) / float(self.big_blind)
        max_bb = float(max_raise) / float(self.big_blind)
        raw_value = self.raise_to_bb_var.get().strip()
        try:
            target_bb = float(raw_value)
        except ValueError:
            self.status_var.set("Enter a numeric raise-to size in BB.")
            self._set_action_buttons()
            return

        target_to_chips = int(round(target_bb * float(self.big_blind)))
        if target_to_chips < min_raise or target_to_chips > max_raise:
            self.status_var.set(
                f"Raise-to must be within {min_bb:.2f} BB and {max_bb:.2f} BB."
            )
            self._set_action_buttons()
            return

        self.awaiting_hero_action = False
        applied = self._apply_custom_raise(self.hero_seat, target_to_chips)
        if not applied:
            self.awaiting_hero_action = True
            self.status_var.set("Unable to apply custom raise at that size.")
            self._set_action_buttons()
            return

        self.status_var.set(f"You raised to {target_to_chips / float(self.big_blind):.2f} BB.")
        self._disable_action_buttons()
        self._refresh_view()
        self.after(5, self._play_until_hero_or_hand_end)

    def _finish_hand(self) -> None:
        if self.state is None or self.hand_ctx is None:
            return

        self.hand_active = False
        self.awaiting_hero_action = False
        self._disable_action_buttons()

        self.hero_bankroll_chips = int(self.state.stacks[self.hero_seat])

        hero_start = float(self.hand_ctx.starting_stacks[self.hero_seat])
        hero_end = float(self.state.stacks[self.hero_seat])
        hero_profit_bb = (hero_end - hero_start) / float(self.big_blind)

        self.stats["hands"] += 1
        self.stats["hero_profit_bb"] += hero_profit_bb
        if hero_profit_bb > 0.0:
            self.stats["hero_wins"] += 1

        for seat in range(NUM_PLAYERS):
            if not self.hand_preflop_opp[seat]:
                continue
            if seat == self.hero_seat:
                self.stats["hero_preflop_opp"] += 1
                self.stats["hero_vpip"] += 1 if self.hand_vpip[seat] else 0
                self.stats["hero_pfr"] += 1 if self.hand_pfr[seat] else 0
                self.stats["hero_three_bet"] += 1 if self.hand_three_bet[seat] else 0
            else:
                self.stats["model_preflop_opp"] += 1
                self.stats["model_vpip"] += 1 if self.hand_vpip[seat] else 0
                self.stats["model_pfr"] += 1 if self.hand_pfr[seat] else 0
                self.stats["model_three_bet"] += 1 if self.hand_three_bet[seat] else 0

        for seat, clone_id in self.clone_id_by_seat.items():
            if clone_id < len(self.clone_bankroll_chips):
                self.clone_bankroll_chips[clone_id] = int(self.state.stacks[seat])

        _update_profile_store_from_hand(
            self.opponent_profile_store,
            self.player_id_by_seat,
            self.hand_profile_stats,
            self.model_runtime_config,
        )

        clone_cards = []
        for seat in range(NUM_PLAYERS):
            if seat == self.hero_seat:
                continue
            name = self.clone_name_by_seat.get(seat, f"Clone {seat}")
            clone_id = self.clone_id_by_seat.get(seat, -1)
            stack_chips = self.clone_bankroll_chips[clone_id] if 0 <= clone_id < len(self.clone_bankroll_chips) else int(
                self.state.stacks[seat]
            )
            clone_cards.append(
                f"{name}: {_format_cards(self.state.hole_cards[seat])} [{stack_chips / float(self.big_blind):.1f} BB]"
            )

        outcome = "won" if hero_profit_bb > 0 else "lost" if hero_profit_bb < 0 else "chopped"
        self._append_log(
            f"Hand #{self.hand_index} complete. You {outcome} {hero_profit_bb:+.2f} BB | " + " | ".join(clone_cards)
        )

        if self.session_running:
            self.status_var.set(f"Hand #{self.hand_index} complete. Preparing next hand...")
        else:
            self.status_var.set("Hand complete.")

        self._refresh_stats_labels()
        self._refresh_exploit_readout()
        self._refresh_view()
        self._schedule_next_hand_if_needed()

    def _seat_xy(self, seat: int, width: int, height: int) -> tuple[float, float]:
        center_x = width * 0.5
        center_y = height * 0.54
        rx = width * 0.34
        ry = height * 0.30
        angle_deg = -30 + (seat * 60)
        angle_rad = math.radians(angle_deg)
        x = center_x + (rx * math.cos(angle_rad))
        y = center_y + (ry * math.sin(angle_rad))
        return x, y

    def _draw_table_scene(self) -> None:
        if self.table_canvas is None:
            return
        canvas = self.table_canvas
        canvas.delete("all")

        canvas.update_idletasks()
        width = max(760, canvas.winfo_width())
        height = max(460, canvas.winfo_height())

        canvas.create_rectangle(0, 0, width, height, fill=COLOR_BG, outline="")
        canvas.create_oval(
            width * 0.10,
            height * 0.15,
            width * 0.90,
            height * 0.91,
            fill=COLOR_TABLE_OUTER,
            outline="#0a3e31",
            width=4,
        )
        canvas.create_oval(
            width * 0.14,
            height * 0.20,
            width * 0.86,
            height * 0.86,
            fill=COLOR_TABLE_INNER,
            outline="#0e4f3f",
            width=2,
        )

        if self.state is None:
            canvas.create_text(
                width * 0.5,
                height * 0.5,
                text="No active hand",
                fill=COLOR_TEXT,
                font=("Segoe UI", 16, "bold"),
            )
            return

        board_cards = flatten_cards_list(self.state.board_cards)
        board_text = _format_cards(board_cards)
        street_idx = street_from_board_len(len(board_cards))
        street_name = STREET_NAMES.get(street_idx, "-")
        total_pot = float(sum(pot.amount for pot in getattr(self.state, "pots", [])) + sum(self.state.bets))
        pot_bb = total_pot / float(self.big_blind)

        canvas.create_rectangle(
            width * 0.34,
            height * 0.41,
            width * 0.66,
            height * 0.61,
            fill=COLOR_BOARD_BG,
            outline="#2d4164",
            width=2,
        )
        canvas.create_text(
            width * 0.5,
            height * 0.445,
            text=f"{street_name}",
            fill=COLOR_ACCENT,
            font=("Segoe UI", 12, "bold"),
        )
        canvas.create_text(
            width * 0.5,
            height * 0.49,
            text=f"Board: {board_text}",
            fill=COLOR_TEXT,
            font=("Consolas", 12, "bold"),
        )
        canvas.create_text(
            width * 0.5,
            height * 0.54,
            text=f"Pot: {pot_bb:.2f} BB",
            fill=COLOR_TEXT,
            font=("Consolas", 12),
        )

        actor = self.state.actor_index if self.hand_active else None
        for seat in range(NUM_PLAYERS):
            x, y = self._seat_xy(seat, width, height)
            in_hand = bool(self.hand_ctx.in_hand[seat]) if self.hand_ctx is not None else True

            if seat == self.hero_seat:
                base_color = COLOR_HERO
                role = "YOU"
            else:
                base_color = COLOR_CLONE
                role = self.clone_name_by_seat.get(seat, "CLONE")

            if not in_hand:
                fill_color = COLOR_FOLDED
            else:
                fill_color = base_color

            outline_color = COLOR_ACTIVE if actor == seat else "#0d1a2e"
            outline_width = 3 if actor == seat else 1

            seat_w = 148
            seat_h = 98
            x0 = x - (seat_w / 2)
            y0 = y - (seat_h / 2)
            x1 = x + (seat_w / 2)
            y1 = y + (seat_h / 2)

            canvas.create_rectangle(
                x0,
                y0,
                x1,
                y1,
                fill=fill_color,
                outline=outline_color,
                width=outline_width,
            )

            stack_bb = float(self.state.stacks[seat]) / float(self.big_blind)
            bet_bb = float(self.state.bets[seat]) / float(self.big_blind)
            show_cards = seat == self.hero_seat or self.show_model_cards_var.get() or (not self.hand_active)
            cards = _format_cards(self.state.hole_cards[seat]) if show_cards else CARD_BACK
            last_action = self.last_action_by_seat.get(seat, "-")

            canvas.create_text(x, y0 + 14, text=f"{POSITION_NAMES_V21[seat]} | {role}", fill="#ffffff", font=("Segoe UI", 9, "bold"))
            canvas.create_text(x, y0 + 34, text=f"Stack {stack_bb:.1f} BB   Bet {bet_bb:.1f}", fill="#eef6ff", font=("Consolas", 9))
            canvas.create_text(x, y0 + 54, text=cards, fill="#ffffff", font=("Consolas", 10, "bold"))
            canvas.create_text(x, y0 + 74, text=f"{last_action}", fill="#e8f4ff", font=("Consolas", 8))

            if actor == seat:
                canvas.create_text(x, y1 + 12, text="TO ACT", fill=COLOR_ACTIVE, font=("Segoe UI", 9, "bold"))

        btn_x, btn_y = self._seat_xy(5, width, height)
        canvas.create_oval(btn_x + 64, btn_y - 52, btn_x + 90, btn_y - 26, fill="#f5f5f5", outline="#d6d6d6")
        canvas.create_text(btn_x + 77, btn_y - 39, text="D", fill="#111827", font=("Segoe UI", 10, "bold"))

    def _refresh_view(self) -> None:
        self._draw_table_scene()
        if self.hand_active and self.awaiting_hero_action:
            self._set_action_buttons()
        else:
            self._disable_action_buttons()

    def _set_action_buttons(self) -> None:
        if self.state is None or self.hand_ctx is None or not self.awaiting_hero_action:
            self._disable_action_buttons()
            return
        legal_mask = build_legal_action_mask(self.state, self.hero_seat, self.hand_ctx)
        for action_id, button in self.action_buttons.items():
            button.config(state=(tk.NORMAL if legal_mask[action_id] > 0.5 else tk.DISABLED))
        can_raise = any(legal_mask[action_id] > 0.5 for action_id in ALL_RAISE_ACTIONS)
        if self.custom_raise_button is not None:
            self.custom_raise_button.config(state=(tk.NORMAL if can_raise else tk.DISABLED))
        if can_raise:
            min_raise, max_raise = self._get_raise_bounds()
            min_bb = float(min_raise) / float(self.big_blind)
            max_bb = float(max_raise) / float(self.big_blind)
            self.raise_hint_var.set(f"Legal raise-to range: {min_bb:.2f} BB to {max_bb:.2f} BB")
            current = self.raise_to_bb_var.get().strip()
            if not current:
                self.raise_to_bb_var.set(f"{min_bb:.2f}")
        else:
            self.raise_hint_var.set("No raise available on this action.")

    def _disable_action_buttons(self) -> None:
        for button in self.action_buttons.values():
            button.config(state=tk.DISABLED)
        if self.custom_raise_button is not None:
            self.custom_raise_button.config(state=tk.DISABLED)
        if self.hand_active:
            self.raise_hint_var.set("Waiting for your turn.")
        else:
            self.raise_hint_var.set("Raise-to sizing appears when it is your turn.")

    def _rate(self, hits: int, opportunities: int) -> float:
        if opportunities <= 0:
            return 0.0
        return float(hits) / float(opportunities)

    def _format_action_mix(self, counts: np.ndarray) -> str:
        total = int(np.sum(counts))
        if total <= 0:
            return "No actions yet."
        parts = []
        for action_id in range(ACTION_COUNT_V21):
            pct = (float(counts[action_id]) / float(total)) * 100.0
            parts.append(f"{ACTION_SHORT_NAMES[action_id]} {pct:4.1f}%")
        return " | ".join(parts)

    def _refresh_exploit_readout(self, actor: Optional[int] = None, details: Optional[Dict[str, object]] = None) -> None:
        if actor is not None and details is not None:
            self.last_exploit_actor = int(actor)
            self.last_exploit_details = details

        hero_profile = _profile_tuple_for_player(self.opponent_profile_store, "human", self.model_runtime_config)
        confidence = float(hero_profile[-1]) * 100.0 if hero_profile else 0.0
        vpip = float(hero_profile[0]) * 100.0 if len(hero_profile) > 0 else 0.0
        pfr = float(hero_profile[1]) * 100.0 if len(hero_profile) > 1 else 0.0
        three_bet = float(hero_profile[2]) * 100.0 if len(hero_profile) > 2 else 0.0
        fold_open = float(hero_profile[3]) * 100.0 if len(hero_profile) > 3 else 0.0
        fold_cbet = float(hero_profile[7]) * 100.0 if len(hero_profile) > 7 else 0.0

        guidance = {}
        if isinstance(self.last_exploit_details, dict):
            guidance = self.last_exploit_details.get("guidance", {}) or {}
        blend_lambda = float(guidance.get("blend_lambda", 0.0) or 0.0) * 100.0
        reasons = guidance.get("reasons", []) or []
        actor_text = "-"
        if self.last_exploit_actor is not None:
            actor_text = self.clone_name_by_seat.get(int(self.last_exploit_actor), f"Clone {self.last_exploit_actor}")
        why_text = "No clear exploit adjustment yet."
        if reasons:
            why_text = " | ".join(str(reason) for reason in reasons[:2])

        self.exploit_read_var.set(
            "Model Read On Hero: "
            f"conf {confidence:.0f}% | lambda {blend_lambda:.0f}% | "
            f"VPIP/PFR/3B {vpip:.0f}/{pfr:.0f}/{three_bet:.0f}% | "
            f"FvOpen {fold_open:.0f}% | FvCBet {fold_cbet:.0f}%\n"
            f"Latest Exploit Actor: {actor_text} | Why: {why_text}"
        )

    def _refresh_stats_labels(self) -> None:
        hands = int(self.stats["hands"])
        hero_profit_bb = float(self.stats["hero_profit_bb"])
        hero_bb100 = (hero_profit_bb / float(hands) * 100.0) if hands > 0 else 0.0
        hero_win_rate = self._rate(int(self.stats["hero_wins"]), hands) * 100.0

        hero_opp = int(self.stats["hero_preflop_opp"])
        model_opp = int(self.stats["model_preflop_opp"])
        hero_vpip = self._rate(int(self.stats["hero_vpip"]), hero_opp) * 100.0
        hero_pfr = self._rate(int(self.stats["hero_pfr"]), hero_opp) * 100.0
        hero_three = self._rate(int(self.stats["hero_three_bet"]), hero_opp) * 100.0
        model_vpip = self._rate(int(self.stats["model_vpip"]), model_opp) * 100.0
        model_pfr = self._rate(int(self.stats["model_pfr"]), model_opp) * 100.0
        model_three = self._rate(int(self.stats["model_three_bet"]), model_opp) * 100.0
        human_rebuy_mode = self.human_rebuy_mode_var.get().strip() or "Below Threshold"
        try:
            human_rebuy_threshold = float(self.human_rebuy_threshold_bb_var.get())
        except (TypeError, ValueError, tk.TclError):
            human_rebuy_threshold = 0.0
        try:
            human_rebuy_to = float(self.human_rebuy_to_bb_var.get())
        except (TypeError, ValueError, tk.TclError):
            human_rebuy_to = float(self.stack_bb)
        rebuy_mode = self.model_rebuy_mode_var.get().strip() or "Every hand"
        try:
            rebuy_threshold = float(self.model_rebuy_threshold_bb_var.get())
        except (TypeError, ValueError, tk.TclError):
            rebuy_threshold = 0.0
        try:
            rebuy_to = float(self.model_rebuy_to_bb_var.get())
        except (TypeError, ValueError, tk.TclError):
            rebuy_to = float(self.stack_bb)
        if rebuy_mode == "Below Threshold":
            rebuy_text = f"{rebuy_mode} ({rebuy_threshold:.1f} -> {rebuy_to:.1f} BB)"
        elif rebuy_mode == "Every hand":
            rebuy_text = f"{rebuy_mode} ({rebuy_to:.1f} BB)"
        else:
            rebuy_text = rebuy_mode
        if human_rebuy_mode == "Below Threshold":
            human_rebuy_text = f"{human_rebuy_mode} ({human_rebuy_threshold:.1f} -> {human_rebuy_to:.1f} BB)"
        elif human_rebuy_mode == "Every hand":
            human_rebuy_text = f"{human_rebuy_mode} ({human_rebuy_to:.1f} BB)"
        else:
            human_rebuy_text = human_rebuy_mode
        hero_roll_bb = float(self.hero_bankroll_chips) / float(self.big_blind)

        self.session_var.set(
            f"Hands: {hands}\n"
            f"Current Hero Seat: {_seat_label(self.hero_seat)}\n"
            f"Human Rebuy: {human_rebuy_text}\n"
            f"Model Rebuy: {rebuy_text}\n"
            f"Hero Roll: {hero_roll_bb:.1f} BB\n"
            f"Hero P/L: {hero_profit_bb:+.2f} BB\n"
            f"Hero BB/100: {hero_bb100:+.2f}\n"
            f"Hero Win Rate: {hero_win_rate:.1f}%"
        )
        self.hero_style_var.set(
            f"Hero VPIP/PFR/3B: {hero_vpip:.1f}% / {hero_pfr:.1f}% / {hero_three:.1f}%\n"
            f"Hero Opportunities: {hero_opp}"
        )
        self.model_style_var.set(
            f"Models VPIP/PFR/3B: {model_vpip:.1f}% / {model_pfr:.1f}% / {model_three:.1f}%\n"
            f"Model Opportunities: {model_opp}"
        )
        self.action_mix_var.set(
            "Hero Mix:  " + self._format_action_mix(self.stats["hero_actions"]) + "\n"
            "Model Mix: " + self._format_action_mix(self.stats["model_actions"])
        )
        self.illegal_var.set(
            f"Fallbacks -> Hero: {self.stats['hero_illegal_actions']} | Models: {self.stats['model_illegal_actions']}"
        )

    def _reset_stats_labels(self) -> None:
        self.session_var.set(
            "Hands: 0\nCurrent Hero Seat: -\nHuman Rebuy: -\nModel Rebuy: -\nHero Roll: -\nHero P/L: +0.00 BB\nHero BB/100: +0.00\nHero Win Rate: 0.0%"
        )
        self.hero_style_var.set("Hero VPIP/PFR/3B: 0.0% / 0.0% / 0.0%\nHero Opportunities: 0")
        self.model_style_var.set("Models VPIP/PFR/3B: 0.0% / 0.0% / 0.0%\nModel Opportunities: 0")
        self.action_mix_var.set("Hero Mix:  No actions yet.\nModel Mix: No actions yet.")
        self.illegal_var.set("Fallbacks -> Hero: 0 | Models: 0")
        self.exploit_read_var.set("Model Read On Hero: conf 0% | lambda 0% | No exploit read yet.")

    def _reset_stats(self) -> None:
        self.stats = {
            "hands": 0,
            "hero_profit_bb": 0.0,
            "hero_wins": 0,
            "hero_actions": np.zeros(ACTION_COUNT_V21, dtype=np.int64),
            "model_actions": np.zeros(ACTION_COUNT_V21, dtype=np.int64),
            "hero_illegal_actions": 0,
            "model_illegal_actions": 0,
            "hero_preflop_opp": 0,
            "hero_vpip": 0,
            "hero_pfr": 0,
            "hero_three_bet": 0,
            "model_preflop_opp": 0,
            "model_vpip": 0,
            "model_pfr": 0,
            "model_three_bet": 0,
        }
        self._reset_all_bankrolls()
        self.opponent_profile_store = {}
        self.hand_profile_stats = _new_preflop_stats(NUM_PLAYERS)
        self.last_exploit_actor = None
        self.last_exploit_details = None
        self._reset_stats_labels()
        self.status_var.set("Statistics reset.")
        self._append_log("Statistics reset.")
        self._refresh_view()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Human eval GUI for Poker v24 model vs 5 clones.")
    parser.add_argument("--model-path", type=str, default=DEFAULT_CHECKPOINT_PATH, help="Path to v24 checkpoint (.pt/.pth).")
    parser.add_argument("--hero-seat", type=int, default=DEFAULT_HERO_SEAT, help="Hero start seat index [0..5].")
    parser.add_argument("--big-blind", type=int, default=DEFAULT_BIG_BLIND, help="Big blind size in chips.")
    parser.add_argument("--stack-bb", type=float, default=DEFAULT_STACK_BB, help="Starting stack in BB for each seat.")
    parser.add_argument("--inter-hand-ms", type=int, default=DEFAULT_INTER_HAND_DELAY_MS, help="Delay between auto hands in milliseconds.")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed for sampling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = HumanEvalGUI(
        model_path=args.model_path,
        hero_seat=max(0, min(NUM_PLAYERS - 1, int(args.hero_seat))),
        big_blind=max(2, int(args.big_blind)),
        stack_bb=max(20.0, float(args.stack_bb)),
        seed=args.seed,
    )
    app.inter_hand_delay_var.set(max(100, int(args.inter_hand_ms)))
    app.mainloop()


if __name__ == "__main__":
    main()


