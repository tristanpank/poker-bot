"""
Configuration settings for the poker bot backend API.

Uses pydantic-settings for environment variable support.
"""

import os
from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )
    
    # API Settings
    app_name: str = "Poker Bot API"
    debug: bool = False
    
    # Model Settings
    model_version: str = "v19"
    model_checkpoint_dir: Path = Path(__file__).parent.parent / "training" / "checkpoints"
    default_model_path: str = ""  # Set dynamically below
    
    # Game Settings
    default_big_blind: int = 10
    default_starting_stack: int = 1000
    equity_iterations: int = 30  # Monte Carlo iterations for equity calculation
    
    # CORS Settings (for frontend integration)
    cors_origins: list[str] = ["*"]
    
    def get_model_path(self, version: str = None) -> Path:
        """Get the path to a model checkpoint."""
        version = version or self.model_version
        version_lower = version.lower().replace("v", "")
        
        # Try different naming conventions
        possible_names = [
            f"poker_agent_v{version_lower}.pt",
            f"poker_agent_v{version_lower}.pth",
            f"poker_agent_{version.lower()}.pt",
        ]
        
        for name in possible_names:
            path = self.model_checkpoint_dir / name
            if path.exists():
                return path
        
        # Return default path even if doesn't exist (will error on load)
        return self.model_checkpoint_dir / possible_names[0]
    
    def get_available_models(self) -> list[str]:
        """List available model versions based on checkpoint files."""
        if not self.model_checkpoint_dir.exists():
            return []
        
        models = []
        for f in self.model_checkpoint_dir.glob("poker_agent_v*.pt*"):
            # Extract version from filename like poker_agent_v18.pt
            name = f.stem  # poker_agent_v18
            version = name.replace("poker_agent_v", "")
            models.append(f"v{version}")
        
        return sorted(models)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
