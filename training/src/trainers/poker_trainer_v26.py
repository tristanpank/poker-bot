from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


CURRENT_DIR = Path(__file__).resolve().parent
SRC_ROOT = CURRENT_DIR.parent
PROJECT_ROOT = SRC_ROOT.parent.parent
DEFAULT_DEEPCFR_REPO_ROOT = PROJECT_ROOT / "external" / "deepcfr-texas-no-limit-holdem-6-players"
DEFAULT_V26_RESULTS_ROOT = PROJECT_ROOT / "training" / "results" / "v26"
DEFAULT_V26_FLAGSHIP_DIR = DEFAULT_DEEPCFR_REPO_ROOT / "flagship_models" / "first"
DEFAULT_V26_FLAGSHIP_CHECKPOINT = DEFAULT_V26_FLAGSHIP_DIR / "mixed_checkpoint_iter_11200.pt"


@dataclass
class DeepCFRConfigV26:
    repo_root: str = str(DEFAULT_DEEPCFR_REPO_ROOT)
    results_root: str = str(DEFAULT_V26_RESULTS_ROOT)
    nickname: str = "v26"


class DeepCFRTrainerV26:
    """Thin adapter around the external 6-player NLHE Deep CFR repository."""

    def __init__(self, config: Optional[DeepCFRConfigV26] = None):
        self.config = config or DeepCFRConfigV26()

    @property
    def repo_root(self) -> Path:
        return Path(self.config.repo_root).resolve()

    @property
    def results_root(self) -> Path:
        return Path(self.config.results_root).resolve()

    @property
    def flagship_models_dir(self) -> Path:
        return (self.repo_root / "flagship_models" / "first").resolve()

    @property
    def flagship_checkpoint(self) -> Path:
        preferred = self.flagship_models_dir / "mixed_checkpoint_iter_11200.pt"
        if preferred.exists():
            return preferred
        fallback = self.flagship_models_dir / "1-model.pt"
        return fallback.resolve()

    def _ensure_repo_available(self) -> None:
        readme_path = self.repo_root / "readme.md"
        if self.repo_root.exists() and readme_path.exists():
            return
        raise FileNotFoundError(
            f"External v26 repo not found at '{self.repo_root}'. "
            "Clone https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players "
            "into external/deepcfr-texas-no-limit-holdem-6-players first."
        )

    @staticmethod
    def _timestamp() -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def create_run_dir(self, label: str) -> Path:
        run_dir = self.results_root / f"{label}_{self._timestamp()}"
        (run_dir / "models").mkdir(parents=True, exist_ok=True)
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)
        return run_dir

    def latest_run_dir(self) -> Optional[Path]:
        if not self.results_root.exists():
            return None
        runs = sorted(
            (path for path in self.results_root.iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
        )
        return runs[-1] if runs else None

    def latest_checkpoint(self) -> Optional[Path]:
        if not self.results_root.exists():
            return None
        checkpoints = sorted(
            self.results_root.rglob("*.pt"),
            key=lambda path: path.stat().st_mtime,
        )
        return checkpoints[-1] if checkpoints else None

    def latest_models_dir(self) -> Optional[Path]:
        checkpoint = self.latest_checkpoint()
        return checkpoint.parent if checkpoint is not None else None

    def preferred_checkpoint(self) -> Optional[Path]:
        latest = self.latest_checkpoint()
        if latest is not None:
            return latest
        if self.flagship_checkpoint.exists():
            return self.flagship_checkpoint
        return None

    def preferred_models_dir(self) -> Optional[Path]:
        latest = self.latest_models_dir()
        if latest is not None:
            return latest
        if self.flagship_models_dir.exists():
            return self.flagship_models_dir
        return None

    def _run_external(self, module_args: list[str], *, cwd: Optional[Path] = None) -> subprocess.CompletedProcess[str]:
        self._ensure_repo_available()
        env = os.environ.copy()
        pythonpath_parts = [str(self.repo_root)]
        if env.get("PYTHONPATH"):
            pythonpath_parts.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
        return subprocess.run(
            [sys.executable, *module_args],
            cwd=str(cwd or self.repo_root),
            env=env,
            check=True,
            text=True,
        )

    def install_requirements(self, *, upgrade_pip: bool = False) -> None:
        self._ensure_repo_available()
        if upgrade_pip:
            self._run_external(["-m", "pip", "install", "--upgrade", "pip"])
        self._run_external(["-m", "pip", "install", "-r", "requirements.txt"])

    def start_training(
        self,
        *,
        iterations: int = 1000,
        traversals: int = 200,
        verbose: bool = False,
        strict: bool = False,
    ) -> Path:
        run_dir = self.create_run_dir(f"{self.config.nickname}_train")
        cmd = [
            "-m",
            "src.training.train",
            "--iterations",
            str(int(iterations)),
            "--traversals",
            str(int(traversals)),
            "--save-dir",
            str((run_dir / "models").resolve()),
            "--log-dir",
            str((run_dir / "logs").resolve()),
        ]
        if verbose:
            cmd.append("--verbose")
        if strict:
            cmd.append("--strict")
        self._run_external(cmd)
        return run_dir

    def resume_training(
        self,
        checkpoint_path: Optional[str] = None,
        *,
        iterations: int = 1000,
        traversals: int = 200,
        verbose: bool = False,
        strict: bool = False,
    ) -> Path:
        checkpoint = Path(checkpoint_path).resolve() if checkpoint_path else self.preferred_checkpoint()
        if checkpoint is None or not checkpoint.exists():
            raise FileNotFoundError(
                "No v26 checkpoint found. Train first, use the shipped flagship model, or pass --checkpoint."
            )
        run_dir = self.create_run_dir(f"{self.config.nickname}_resume")
        cmd = [
            "-m",
            "src.training.train",
            "--checkpoint",
            str(checkpoint),
            "--iterations",
            str(int(iterations)),
            "--traversals",
            str(int(traversals)),
            "--save-dir",
            str((run_dir / "models").resolve()),
            "--log-dir",
            str((run_dir / "logs").resolve()),
        ]
        if verbose:
            cmd.append("--verbose")
        if strict:
            cmd.append("--strict")
        self._run_external(cmd)
        return run_dir

    def self_play_training(
        self,
        checkpoint_path: Optional[str] = None,
        *,
        iterations: int = 2000,
        traversals: int = 400,
        verbose: bool = False,
        strict: bool = False,
    ) -> Path:
        checkpoint = Path(checkpoint_path).resolve() if checkpoint_path else self.preferred_checkpoint()
        if checkpoint is None or not checkpoint.exists():
            raise FileNotFoundError(
                "No v26 checkpoint found for self-play. Train first, use the shipped flagship model, or pass --checkpoint."
            )
        run_dir = self.create_run_dir(f"{self.config.nickname}_selfplay")
        cmd = [
            "-m",
            "src.training.train",
            "--checkpoint",
            str(checkpoint),
            "--self-play",
            "--iterations",
            str(int(iterations)),
            "--traversals",
            str(int(traversals)),
            "--save-dir",
            str((run_dir / "models").resolve()),
            "--log-dir",
            str((run_dir / "logs").resolve()),
        ]
        if verbose:
            cmd.append("--verbose")
        if strict:
            cmd.append("--strict")
        self._run_external(cmd)
        return run_dir

    def mixed_training(
        self,
        checkpoint_dir: Optional[str] = None,
        *,
        iterations: int = 10000,
        traversals: int = 400,
        model_prefix: str = "t_",
        refresh_interval: int = 1000,
        num_opponents: int = 5,
        verbose: bool = False,
        strict: bool = False,
    ) -> Path:
        resolved_dir = Path(checkpoint_dir).resolve() if checkpoint_dir else self.preferred_models_dir()
        if resolved_dir is None or not resolved_dir.exists():
            raise FileNotFoundError(
                "No v26 checkpoint directory found for mixed training. Train first or pass --checkpoint-dir."
            )
        resolved_prefix = model_prefix
        if checkpoint_dir is None and resolved_dir == self.flagship_models_dir and model_prefix == "t_":
            resolved_prefix = ""
        run_dir = self.create_run_dir(f"{self.config.nickname}_mixed")
        cmd = [
            "-m",
            "src.training.train",
            "--mixed",
            "--checkpoint-dir",
            str(resolved_dir),
            "--model-prefix",
            str(resolved_prefix),
            "--refresh-interval",
            str(int(refresh_interval)),
            "--num-opponents",
            str(int(num_opponents)),
            "--iterations",
            str(int(iterations)),
            "--traversals",
            str(int(traversals)),
            "--save-dir",
            str((run_dir / "models").resolve()),
            "--log-dir",
            str((run_dir / "logs").resolve()),
        ]
        if verbose:
            cmd.append("--verbose")
        if strict:
            cmd.append("--strict")
        self._run_external(cmd)
        return run_dir

    def play_cli(
        self,
        models_dir: Optional[str] = None,
        *,
        model_pattern: str = "*.pt",
        num_models: int = 5,
        position: int = 0,
        stake: float = 200.0,
        sb: float = 1.0,
        bb: float = 2.0,
        verbose: bool = False,
        strict: bool = False,
        no_shuffle: bool = False,
    ) -> Path:
        resolved_dir = Path(models_dir).resolve() if models_dir else self.preferred_models_dir()
        if resolved_dir is None or not resolved_dir.exists():
            raise FileNotFoundError(
                "No v26 model directory found. Train v26 first or use the shipped flagship models."
            )
        cmd = [
            "-m",
            "scripts.play",
            "--models-dir",
            str(resolved_dir),
            "--model-pattern",
            str(model_pattern),
            "--num-models",
            str(int(num_models)),
            "--position",
            str(int(position)),
            "--stake",
            str(float(stake)),
            "--sb",
            str(float(sb)),
            "--bb",
            str(float(bb)),
        ]
        if verbose:
            cmd.append("--verbose")
        if strict:
            cmd.append("--strict")
        if no_shuffle:
            cmd.append("--no-shuffle")
        self._run_external(cmd)
        return resolved_dir

    def play_gui(
        self,
        models_dir: Optional[str] = None,
        *,
        position: int = 0,
        stake: float = 200.0,
        sb: float = 1.0,
        bb: float = 2.0,
    ) -> Path:
        resolved_dir = Path(models_dir).resolve() if models_dir else self.preferred_models_dir()
        if resolved_dir is None or not resolved_dir.exists():
            raise FileNotFoundError(
                "No v26 model directory found. Train v26 first or use the shipped flagship models."
            )
        cmd = [
            "-m",
            "scripts.poker_gui",
            "--models_folder",
            str(resolved_dir),
            "--position",
            str(int(position)),
            "--stake",
            str(float(stake)),
            "--sb",
            str(float(sb)),
            "--bb",
            str(float(bb)),
        ]
        self._run_external(cmd)
        return resolved_dir


# Backward-compatible aliases for earlier in-repo references.
PluribusConfigV26 = DeepCFRConfigV26
PluribusTrainerV26 = DeepCFRTrainerV26


__all__ = [
    "DEFAULT_DEEPCFR_REPO_ROOT",
    "DEFAULT_V26_FLAGSHIP_CHECKPOINT",
    "DEFAULT_V26_FLAGSHIP_DIR",
    "DEFAULT_V26_RESULTS_ROOT",
    "DeepCFRConfigV26",
    "DeepCFRTrainerV26",
    "PluribusConfigV26",
    "PluribusTrainerV26",
]
