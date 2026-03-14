"""
Model service for loading and running inference on trained poker models.
"""

import sys
import importlib
from pathlib import Path
from typing import Optional
import threading

import torch
import numpy as np

# Add the training models directory to path for imports
_project_root = Path(__file__).parent.parent.parent
_models_path = _project_root / "training" / "src" / "models"
if str(_models_path) not in sys.path:
    sys.path.insert(0, str(_models_path))
_features_path = _project_root / "training" / "src" / "features"
if str(_features_path) not in sys.path:
    sys.path.insert(0, str(_features_path))
_workers_path = _project_root / "training" / "src" / "workers"
if str(_workers_path) not in sys.path:
    sys.path.insert(0, str(_workers_path))

from poker_worker_v24 import infoset_key_from_vector
from tabular_policy_v24 import (
    TabularPolicySnapshot,
    deserialize_node_store,
    freeze_policy_snapshot,
    normalize_masked_policy,
    uniform_legal_policy,
)

from backend.config import get_settings
from backend.poker_versions import get_action_names, get_version_spec, version_to_int


class ModelService:
    """
    Service for loading and managing poker bot models.
    
    Supports multiple model versions and provides thread-safe inference.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._models: dict[str, object] = {}
        self._lock = threading.Lock()

    def _extract_state_dict(self, checkpoint: object) -> dict[str, torch.Tensor]:
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint and isinstance(checkpoint["model_state_dict"], dict):
                return checkpoint["model_state_dict"]
            if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
                return checkpoint["state_dict"]
            if any(str(key).endswith("weight") for key in checkpoint.keys()):
                return checkpoint  # assume the payload itself is the state dict
        if isinstance(checkpoint, dict):
            raise ValueError("Checkpoint dictionary does not include a recognized model state dict.")
        raise ValueError("Checkpoint payload is not a dictionary.")

    def _extract_tabular_snapshot(self, checkpoint: object) -> Optional[TabularPolicySnapshot]:
        if isinstance(checkpoint, TabularPolicySnapshot):
            return checkpoint
        if not isinstance(checkpoint, dict):
            return None
        format_version = str(checkpoint.get("format_version", "")).strip().lower()
        if format_version != "tabular_mccfr_v24" and "actor_snapshot" not in checkpoint and "node_store" not in checkpoint:
            return None

        actor_snapshot = checkpoint.get("actor_snapshot")
        if isinstance(actor_snapshot, TabularPolicySnapshot):
            return actor_snapshot
        if isinstance(actor_snapshot, dict):
            return TabularPolicySnapshot.from_payload(actor_snapshot)

        node_store = checkpoint.get("node_store")
        if isinstance(node_store, dict):
            metadata = {}
            if isinstance(checkpoint.get("config"), dict):
                metadata["config"] = dict(checkpoint["config"])
            return freeze_policy_snapshot(deserialize_node_store(node_store), metadata=metadata)
        return None

    def _tabular_policy_for_observation(
        self,
        snapshot: TabularPolicySnapshot,
        observation: np.ndarray,
        legal_actions: list[int],
        version: str,
    ) -> np.ndarray:
        spec = get_version_spec(version)
        vec = np.asarray(observation, dtype=np.float32).reshape(-1)
        if vec.shape[0] < spec.state_dim:
            padded = np.zeros(spec.state_dim, dtype=np.float32)
            padded[: vec.shape[0]] = vec
            vec = padded
        elif vec.shape[0] > spec.state_dim:
            vec = vec[: spec.state_dim]

        infoset_key = infoset_key_from_vector(vec)
        legal_mask = np.zeros(spec.action_dim, dtype=np.float32)
        for action in legal_actions:
            if 0 <= int(action) < spec.action_dim:
                legal_mask[int(action)] = 1.0

        entry = snapshot.policy_table.get(infoset_key)
        if entry is None:
            return uniform_legal_policy(legal_mask)
        if float(np.asarray(entry.average_policy, dtype=np.float32).sum()) > 1e-8:
            return normalize_masked_policy(entry.average_policy, legal_mask)
        if float(np.asarray(entry.current_policy, dtype=np.float32).sum()) > 1e-8:
            return normalize_masked_policy(entry.current_policy, legal_mask)
        return uniform_legal_policy(legal_mask)

    def _infer_deepcfr_dims(
        self,
        state_dict: dict[str, torch.Tensor],
        checkpoint: object,
        default_state_dim: int,
        default_action_dim: int,
    ) -> tuple[int, int, int]:
        state_dim = int(default_state_dim)
        hidden_dim = 256
        action_dim = int(default_action_dim)

        if isinstance(checkpoint, dict) and isinstance(checkpoint.get("config"), dict):
            config = checkpoint["config"]
            state_dim = int(config.get("state_dim", state_dim))
            hidden_dim = int(config.get("hidden_dim", hidden_dim))
            action_dim = int(config.get("action_count", action_dim))

        input_weight = state_dict.get("input_layer.weight")
        strategy_weight = state_dict.get("strategy_head.2.weight")
        if input_weight is not None and len(input_weight.shape) == 2:
            hidden_dim = int(input_weight.shape[0])
            state_dim = int(input_weight.shape[1])
        if strategy_weight is not None and len(strategy_weight.shape) == 2:
            action_dim = int(strategy_weight.shape[0])

        return state_dim, hidden_dim, action_dim

    def _deepcfr_runtime(self, version: str):
        version_num = version_to_int(version)
        if version_num >= 24:
            module_name = "poker_model_v24"
        elif version_num >= 23:
            module_name = "poker_model_v23"
        elif version_num >= 22:
            module_name = "poker_model_v22"
        else:
            module_name = "poker_model_v21"

        module = importlib.import_module(module_name)
        return (
            module.PokerDeepCFRNet,
            getattr(module, "load_compatible_state_dict", None),
            module.masked_policy,
        )

    def _action_label(self, action_id: int, version: str) -> str:
        return get_action_names(version).get(action_id, f"ACTION_{action_id}")
        
    def load_model(self, version: str = None) -> object:
        """
        Load a model by version. Models are cached after first load.
        
        Args:
            version: Model version (e.g., 'v18'). Uses default if not specified.
            
        Returns:
            Loaded PyTorch model in eval mode.
        """
        version = (version or self.settings.model_version).lower()
        version_num = version_to_int(version)
        
        # Return cached model if available
        if version in self._models:
            return self._models[version]
        
        with self._lock:
            # Double-check after acquiring lock
            if version in self._models:
                return self._models[version]
            
            model_path = self.settings.get_model_path(version)
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
            
            # Load the checkpoint
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

            tabular_snapshot = self._extract_tabular_snapshot(checkpoint)
            if tabular_snapshot is not None:
                self._models[version] = tabular_snapshot
                return tabular_snapshot

            if version_num >= 21:
                state_dict = self._extract_state_dict(checkpoint)
                spec = get_version_spec(version)
                model_cls, compatible_loader, _ = self._deepcfr_runtime(version)
                state_dim, hidden_dim, action_dim = self._infer_deepcfr_dims(
                    state_dict,
                    checkpoint,
                    default_state_dim=spec.state_dim,
                    default_action_dim=spec.action_dim,
                )
                init_kwargs = {
                    "state_dim": state_dim,
                    "hidden_dim": hidden_dim,
                    "action_dim": action_dim,
                }
                if "init_weights" in model_cls.__init__.__code__.co_varnames:
                    init_kwargs["init_weights"] = False
                model = model_cls(**init_kwargs)
                if compatible_loader is not None:
                    compatible_loader(model, state_dict)
                else:
                    model.load_state_dict(state_dict, strict=True)
            else:
                # Import the appropriate model class based on version
                model = self._create_model_for_version(version)
                state_dict = self._extract_state_dict(checkpoint)
                model.load_state_dict(state_dict)
            
            model.to(self.device)
            model.eval()
            
            self._models[version] = model
            return model
    
    def _create_model_for_version(self, version: str) -> torch.nn.Module:
        """Create a model instance for the given version."""
        version_num = version_to_int(version)
        
        # Try to import the version-specific model
        try:
            if version_num >= 19:
                # V19+ uses 520-dim state with weighted equity
                from poker_model_v19 import DuelingPokerNet
                return DuelingPokerNet(state_dim=520)
            if version_num >= 15:
                # V15-18 uses 520-dim state for 6-max
                from poker_model_v18 import DuelingPokerNet
                return DuelingPokerNet(state_dim=520)
            if version_num >= 13:
                # V13-14 uses 385-dim state
                from poker_model_v14 import DuelingPokerNet
                return DuelingPokerNet(state_dim=385)
            # Fallback to latest legacy architecture
            from poker_model_v19 import DuelingPokerNet
            return DuelingPokerNet(state_dim=520)
        except ImportError:
            # Fallback: try V19 model as the latest legacy architecture.
            from poker_model_v19 import DuelingPokerNet
            return DuelingPokerNet(state_dim=520)
    
    def get_action(
        self, 
        observation: np.ndarray, 
        legal_actions: list[int],
        version: str = None
    ) -> tuple[int, dict[str, float]]:
        """
        Get the best action for a given observation.
        
        Args:
            observation: State observation vector (520-dim for V15+, 385-dim for older)
            legal_actions: List of legal action IDs
            version: Model version to use
            
        Returns:
            Tuple of (action_id, q_values_dict)
        """
        version = (version or self.settings.model_version).lower()
        version_num = version_to_int(version)
        model = self.load_model(version)

        if isinstance(model, TabularPolicySnapshot):
            scores = self._tabular_policy_for_observation(model, observation, legal_actions, version)
            best_action = int(np.argmax(scores))
            score_dict = {
                self._action_label(i, version): float(scores[i])
                for i in range(len(scores))
            }
            return best_action, score_dict
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            if version_num >= 21:
                _, _, masked_policy = self._deepcfr_runtime(version)

                logits = model.forward_strategy(state_tensor).squeeze(0).cpu().numpy()
                legal_mask = np.zeros_like(logits, dtype=np.float32)
                for action in legal_actions:
                    if 0 <= action < len(legal_mask):
                        legal_mask[action] = 1.0
                probs = masked_policy(logits, legal_mask)
                best_action = int(np.argmax(probs))
                scores = probs
            else:
                scores = model(state_tensor).squeeze(0).cpu().numpy()
                masked = np.full_like(scores, float("-inf"))
                for action in legal_actions:
                    if 0 <= action < len(masked):
                        masked[action] = scores[action]
                best_action = int(np.argmax(masked))

        score_dict = {
            self._action_label(i, version): float(scores[i])
            for i in range(len(scores))
        }

        return best_action, score_dict

    def is_loaded(self, version: str = None) -> bool:
        """Check if a model version is loaded."""
        version = (version or self.settings.model_version).lower()
        return version in self._models
    
    def get_available_models(self) -> list[str]:
        """Get list of available model versions."""
        return self.settings.get_available_models()


# Singleton instance
_model_service: Optional[ModelService] = None


def get_model_service() -> ModelService:
    """Get the singleton model service instance."""
    global _model_service
    if _model_service is None:
        _model_service = ModelService()
    return _model_service
