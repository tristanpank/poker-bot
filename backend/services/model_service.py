"""
Model service for loading and running inference on trained poker models.
"""

import sys
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

from backend.config import get_settings
from backend.models.schemas import ACTION_NAMES


class ModelService:
    """
    Service for loading and managing poker bot models.
    
    Supports multiple model versions and provides thread-safe inference.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._models: dict[str, torch.nn.Module] = {}
        self._lock = threading.Lock()
        
    def load_model(self, version: str = None) -> torch.nn.Module:
        """
        Load a model by version. Models are cached after first load.
        
        Args:
            version: Model version (e.g., 'v18'). Uses default if not specified.
            
        Returns:
            Loaded PyTorch model in eval mode.
        """
        version = (version or self.settings.model_version).lower()
        
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
            
            # Import the appropriate model class based on version
            model = self._create_model_for_version(version)
            
            # Load the checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"])
                elif "state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["state_dict"])
                else:
                    # Assume the dict IS the state dict
                    model.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint)
            
            model.to(self.device)
            model.eval()
            
            self._models[version] = model
            return model
    
    def _create_model_for_version(self, version: str) -> torch.nn.Module:
        """Create a model instance for the given version."""
        version_num = version.replace("v", "")
        
        # Try to import the version-specific model
        try:
            if version_num >= "15":
                # V15+ uses 520-dim state for 6-max
                from poker_model_v18 import DuelingPokerNet
                return DuelingPokerNet(state_dim=520)
            elif version_num >= "13":
                # V13-14 uses 385-dim state
                from poker_model_v14 import DuelingPokerNet
                return DuelingPokerNet(state_dim=385)
            else:
                # Fallback to latest architecture
                from poker_model_v18 import DuelingPokerNet
                return DuelingPokerNet(state_dim=520)
        except ImportError:
            # Fallback: try V18 model as it's the latest
            from poker_model_v18 import DuelingPokerNet
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
        model = self.load_model(version)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            q_values = model(state_tensor).squeeze(0).cpu().numpy()
        
        # Mask illegal actions
        masked_q = np.full_like(q_values, float('-inf'))
        for action in legal_actions:
            masked_q[action] = q_values[action]
        
        best_action = int(np.argmax(masked_q))
        
        # Create q_values dict for response
        q_values_dict = {
            ACTION_NAMES[i]: float(q_values[i]) 
            for i in range(len(q_values))
        }
        
        return best_action, q_values_dict
    
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
