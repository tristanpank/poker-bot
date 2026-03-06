import os
import sys
import threading
import time
from collections import OrderedDict
from multiprocessing.connection import Client, Listener, wait
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.dirname(CURRENT_DIR)
MODELS_DIR = os.path.join(SRC_ROOT, "models")
if MODELS_DIR not in sys.path:
    sys.path.insert(0, MODELS_DIR)

from poker_model_v24 import PokerDeepCFRNet, load_compatible_state_dict


def _align_state_vector(state_vec: np.ndarray, expected_dim: int) -> np.ndarray:
    expected = int(expected_dim)
    arr = np.asarray(state_vec, dtype=np.float32).reshape(-1)
    if expected <= 0 or arr.shape[0] == expected:
        return arr
    if arr.shape[0] > expected:
        return arr[:expected]
    aligned = np.zeros(expected, dtype=np.float32)
    aligned[: arr.shape[0]] = arr
    return aligned


def _normalize_heads(heads) -> Tuple[str, ...]:
    if heads is None:
        return ("regret", "strategy", "exploit")
    if isinstance(heads, str):
        heads = (heads,)
    normalized: List[str] = []
    for head in heads:
        name = str(head).strip().lower()
        if name in {"regret", "strategy", "exploit"} and name not in normalized:
            normalized.append(name)
    return tuple(normalized) or ("strategy",)


def _forward_selected_heads(model: PokerDeepCFRNet, tensor: torch.Tensor, heads: Tuple[str, ...]) -> Dict[str, torch.Tensor]:
    selected = _normalize_heads(heads)
    if selected == ("regret",):
        return {"regret": model.forward_regret(tensor)}
    if selected == ("strategy",):
        return {"strategy": model.forward_strategy(tensor)}
    if selected == ("exploit",):
        return {"exploit": model.forward_exploit(tensor)}
    trunk = model._forward_trunk(tensor)
    outputs: Dict[str, torch.Tensor] = {}
    if "regret" in selected:
        outputs["regret"] = model.regret_head(trunk)
    if "strategy" in selected:
        outputs["strategy"] = model.strategy_head(trunk)
    if "exploit" in selected:
        outputs["exploit"] = model.exploit_head(trunk)
    return outputs


class GPURolloutInferenceService:
    def __init__(
        self,
        device: str,
        max_batch_size: int = 512,
        max_wait_ms: float = 1.5,
        model_cache_size: int = 6,
    ):
        self.device = torch.device(device)
        self.max_batch_size = max(1, int(max_batch_size))
        self.max_wait_seconds = max(0.0, float(max_wait_ms) / 1000.0)
        self.model_cache_size = max(1, int(model_cache_size))

        self._listener: Optional[Listener] = None
        self._authkey = os.urandom(32)
        self._accept_thread: Optional[threading.Thread] = None
        self._serve_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._connections: List[object] = []
        self._connections_lock = threading.Lock()

        self._registry_lock = threading.Lock()
        self._cpu_registry: Dict[str, Dict[str, torch.Tensor]] = {}
        self._model_specs: Dict[str, Tuple[int, int, int]] = {}
        self._gpu_cache: "OrderedDict[str, PokerDeepCFRNet]" = OrderedDict()

    def start(self) -> None:
        if self._listener is not None:
            return
        self._listener = Listener(("127.0.0.1", 0), authkey=self._authkey)
        self._accept_thread = threading.Thread(target=self._accept_loop, name="v24-gpu-infer-accept", daemon=True)
        self._serve_thread = threading.Thread(target=self._serve_loop, name="v24-gpu-infer-serve", daemon=True)
        self._accept_thread.start()
        self._serve_thread.start()

    @property
    def address(self):
        return None if self._listener is None else self._listener.address

    @property
    def authkey(self) -> bytes:
        return bytes(self._authkey)

    def register_model(
        self,
        model_key: str,
        state_dict: Dict[str, torch.Tensor],
        state_dim: int,
        hidden_dim: int,
        action_dim: int,
    ) -> None:
        key = str(model_key)
        cpu_state = {name: tensor.detach().cpu() for name, tensor in state_dict.items()}
        with self._registry_lock:
            self._cpu_registry[key] = cpu_state
            self._model_specs[key] = (int(state_dim), int(hidden_dim), int(action_dim))
            stale = self._gpu_cache.pop(key, None)
        if stale is not None:
            del stale

    def register_snapshot(self, model_key: str, snapshot: PokerDeepCFRNet) -> None:
        self.register_model(
            model_key,
            snapshot.state_dict(),
            int(getattr(snapshot, "state_dim", 0) or 0),
            int(getattr(snapshot, "hidden_dim", 0) or 0),
            int(getattr(snapshot, "action_dim", 0) or 0),
        )

    def retain_model_keys(self, active_keys: Iterable[str]) -> None:
        keep = {str(key) for key in active_keys if str(key)}
        with self._registry_lock:
            for key in list(self._cpu_registry.keys()):
                if key not in keep:
                    self._cpu_registry.pop(key, None)
                    self._model_specs.pop(key, None)
                    stale = self._gpu_cache.pop(key, None)
                    if stale is not None:
                        del stale

    def stop(self) -> None:
        self._stop_event.set()
        listener = self._listener
        if listener is not None:
            address = listener.address
            try:
                Client(address, authkey=self._authkey).close()
            except Exception:
                pass
            try:
                listener.close()
            except Exception:
                pass
            self._listener = None
        with self._connections_lock:
            connections = list(self._connections)
            self._connections.clear()
        for conn in connections:
            try:
                conn.close()
            except Exception:
                pass
        with self._registry_lock:
            self._cpu_registry.clear()
            self._model_specs.clear()
            self._gpu_cache.clear()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _accept_loop(self) -> None:
        listener = self._listener
        if listener is None:
            return
        while not self._stop_event.is_set():
            try:
                conn = listener.accept()
            except Exception:
                if self._stop_event.is_set():
                    break
                time.sleep(0.01)
                continue
            with self._connections_lock:
                self._connections.append(conn)

    def _drop_connection(self, conn) -> None:
        with self._connections_lock:
            self._connections = [existing for existing in self._connections if existing is not conn]
        try:
            conn.close()
        except Exception:
            pass

    def _load_model_to_device(self, model_key: str) -> PokerDeepCFRNet:
        with self._registry_lock:
            cached = self._gpu_cache.pop(model_key, None)
            if cached is not None:
                self._gpu_cache[model_key] = cached
                return cached
            state_dict = self._cpu_registry.get(model_key)
            dims = self._model_specs.get(model_key)
        if state_dict is None or dims is None:
            raise KeyError(f"GPU rollout model is not registered: {model_key}")

        state_dim, hidden_dim, action_dim = dims
        model = PokerDeepCFRNet(
            state_dim=int(state_dim),
            hidden_dim=int(hidden_dim),
            action_dim=int(action_dim),
            init_weights=False,
        ).to(self.device)
        load_compatible_state_dict(model, state_dict)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)

        evicted: Optional[PokerDeepCFRNet] = None
        with self._registry_lock:
            self._gpu_cache[model_key] = model
            while len(self._gpu_cache) > self.model_cache_size:
                _, evicted = self._gpu_cache.popitem(last=False)
        if evicted is not None:
            del evicted
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        return model

    def _serve_loop(self) -> None:
        while not self._stop_event.is_set():
            with self._connections_lock:
                connections = list(self._connections)
            if not connections:
                time.sleep(0.001)
                continue
            try:
                ready = list(wait(connections, timeout=self.max_wait_seconds))
            except Exception:
                ready = []
            if not ready:
                continue

            batch: List[Tuple[object, dict]] = []
            deadline = time.perf_counter() + self.max_wait_seconds
            pending_ready = ready
            while pending_ready and len(batch) < self.max_batch_size:
                next_ready: List[object] = []
                for conn in pending_ready:
                    if len(batch) >= self.max_batch_size:
                        break
                    try:
                        request = conn.recv()
                    except Exception:
                        self._drop_connection(conn)
                        continue
                    if not isinstance(request, dict):
                        self._send_error(conn, "Invalid GPU rollout inference request.")
                        continue
                    if request.get("type") == "close":
                        self._drop_connection(conn)
                        continue
                    batch.append((conn, request))
                if len(batch) >= self.max_batch_size:
                    break
                remaining = deadline - time.perf_counter()
                if remaining <= 1e-6:
                    break
                with self._connections_lock:
                    connections = list(self._connections)
                try:
                    next_ready = list(wait(connections, timeout=remaining))
                except Exception:
                    next_ready = []
                pending_ready = next_ready
            if not batch:
                continue
            self._process_batch(batch)

    def _process_batch(self, batch: List[Tuple[object, dict]]) -> None:
        grouped: Dict[Tuple[str, Tuple[str, ...]], List[Tuple[object, np.ndarray]]] = {}
        for conn, request in batch:
            model_key = str(request.get("model_key", "") or "")
            state_vec = request.get("state_vec")
            heads = _normalize_heads(request.get("heads"))
            if not model_key:
                self._send_error(conn, "Missing model_key in GPU rollout inference request.")
                continue
            if not isinstance(state_vec, np.ndarray):
                try:
                    state_vec = np.asarray(state_vec, dtype=np.float32)
                except Exception:
                    self._send_error(conn, "state_vec is not array-like.")
                    continue
            if state_vec.dtype != np.float32:
                state_vec = state_vec.astype(np.float32, copy=False)
            grouped.setdefault((model_key, heads), []).append((conn, state_vec))

        for (model_key, heads), items in grouped.items():
            try:
                model = self._load_model_to_device(model_key)
                expected_dim = int(getattr(model, "state_dim", 0) or 0)
                states = np.stack([_align_state_vector(state, expected_dim) for _, state in items], axis=0)
                with torch.inference_mode():
                    tensor = torch.from_numpy(states).to(self.device, dtype=torch.float32, non_blocking=False)
                    outputs = _forward_selected_heads(model, tensor, heads)
                    cpu_outputs = {
                        name: value.detach().cpu().numpy().astype(np.float32, copy=False) for name, value in outputs.items()
                    }
                for idx, (conn, _) in enumerate(items):
                    try:
                        conn.send({"ok": True, "outputs": {name: value[idx] for name, value in cpu_outputs.items()}})
                    except Exception:
                        self._drop_connection(conn)
            except Exception as exc:
                message = f"GPU rollout inference failed for '{model_key}': {exc}"
                for conn, _ in items:
                    self._send_error(conn, message)

    @staticmethod
    def _send_error(conn, message: str) -> None:
        try:
            conn.send({"ok": False, "error": str(message)})
        except Exception:
            try:
                conn.close()
            except Exception:
                pass


