"""
Configuration settings for the poker bot backend API.

Uses pydantic-settings for environment variable support.
"""

import re
from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

from backend.poker_versions import version_to_int


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
    model_version: str = "v24"
    model_checkpoint_dir: Path = Path(__file__).parent.parent / "training" / "models"
    legacy_model_checkpoint_dir: Path = Path(__file__).parent.parent / "training" / "checkpoints"
    default_model_path: str = ""  # Set dynamically below
    
    # Game Settings
    default_big_blind: int = 10
    default_starting_stack: int = 1000
    equity_iterations: int = 30  # Monte Carlo iterations for equity calculation
    
    # CORS Settings (for frontend integration)
    cors_origins: list[str] = ["*"]

    def _checkpoint_dirs(self) -> list[Path]:
        dirs = [self.model_checkpoint_dir, self.legacy_model_checkpoint_dir]
        unique: list[Path] = []
        for path in dirs:
            if path not in unique:
                unique.append(path)
        return unique

    def get_model_path(self, version: str = None) -> Path:
        """Get the path to a model checkpoint."""
        version = version or self.model_version
        version_lower = version.lower().replace("v", "")

        # Try different naming conventions
        possible_names = [
            f"poker_agent_v{version_lower}_deepcfr.pt",
            f"poker_agent_v{version_lower}_deepcfr.pth",
            f"poker_agent_v{version_lower}.pt",
            f"poker_agent_v{version_lower}.pth",
            f"poker_agent_{version.lower()}.pt",
            f"poker_agent_{version.lower()}.pth",
        ]

        for directory in self._checkpoint_dirs():
            for name in possible_names:
                path = directory / name
                if path.exists():
                    return path

        # Return default path even if doesn't exist (will error on load)
        return self.model_checkpoint_dir / possible_names[0]

    def get_available_models(self) -> list[str]:
        """List available model versions based on checkpoint files."""
        version_pattern = re.compile(r"poker_agent_v(\d+)(?:_deepcfr)?\.(?:pt|pth)$", re.IGNORECASE)
        versions: set[str] = set()

        for directory in self._checkpoint_dirs():
            if not directory.exists():
                continue
            for file_path in directory.iterdir():
                if not file_path.is_file():
                    continue
                match = version_pattern.match(file_path.name)
                if match:
                    versions.add(f"v{match.group(1)}")

        return sorted(versions, key=version_to_int)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
