from __future__ import annotations

import argparse
import sys
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
SRC_ROOT = CURRENT_DIR.parent
FEATURES_DIR = SRC_ROOT / "features"
TRAINERS_DIR = SRC_ROOT / "trainers"
for _path in (FEATURES_DIR, TRAINERS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

try:
    from poker_gui_trainer_v26 import DeepCFRTrainerV26GUI, SYNTHETIC_OPPONENT_STYLES
    from poker_state_v26 import ACTION_NAMES_V26
    from poker_trainer_v26 import DeepCFRConfigV26, DeepCFRTrainerV26
except ImportError:  # pragma: no cover - package-style imports
    from ..trainers.poker_gui_trainer_v26 import DeepCFRTrainerV26GUI, SYNTHETIC_OPPONENT_STYLES
    from ..features.poker_state_v26 import ACTION_NAMES_V26
    from ..trainers.poker_trainer_v26 import DeepCFRConfigV26, DeepCFRTrainerV26


def configure_legacy_gui_module():
    try:
        import poker_agent_v25 as legacy_gui
    except ImportError:  # pragma: no cover - package-style imports
        from . import poker_agent_v25 as legacy_gui

    model_version = "v26"
    model_label = model_version.upper()
    default_checkpoint_filename = f"poker_agent_{model_version}_resume_checkpoint.pt"
    default_checkpoint_path = str(Path(legacy_gui.PROJECT_ROOT) / "models" / default_checkpoint_filename)

    legacy_gui.DeepCFRTrainerV25 = DeepCFRTrainerV26GUI
    legacy_gui.SYNTHETIC_OPPONENT_STYLES = SYNTHETIC_OPPONENT_STYLES
    legacy_gui.ACTION_NAMES = list(ACTION_NAMES_V26)
    legacy_gui.MODEL_VERSION = model_version
    legacy_gui.MODEL_LABEL = model_label
    legacy_gui.DEFAULT_CHECKPOINT_FILENAME = default_checkpoint_filename
    legacy_gui.DEFAULT_CHECKPOINT_PATH = default_checkpoint_path
    legacy_gui.EVAL_MODE_VALUES = ["self_play", "heuristics", "checkpoints", "eval_suite", *list(SYNTHETIC_OPPONENT_STYLES)]

    def _v26_planned_total_batches(gui) -> int:
        trainer = getattr(gui, "trainer", None)
        if trainer is None:
            return 0
        if getattr(trainer.config, "training_monitor_mode", "") == "phased" and hasattr(trainer, "planned_total_traversals"):
            return int(trainer.planned_total_traversals())
        return 0

    if not getattr(legacy_gui.TrainingGUI, "_v26_ingest_patched", False):
        original_ingest_snapshot = legacy_gui.TrainingGUI._ingest_snapshot

        def _v26_ingest_snapshot(self, snapshot, chunk_size, chunk_elapsed_seconds, append_history=True):
            original_ingest_snapshot(self, snapshot, chunk_size, chunk_elapsed_seconds, append_history=append_history)
            planned_total = _v26_planned_total_batches(self)
            if planned_total > 0:
                self.gui_state["total_batches"] = max(int(snapshot.traversals_completed), planned_total)
            rolling_speed = float(getattr(snapshot, "hands_per_second", 0.0))
            if rolling_speed > 0.0:
                self.gui_state["speed"] = rolling_speed
                remaining = (
                    (self.gui_state["total_batches"] - snapshot.traversals_completed) / rolling_speed / 60.0
                    if rolling_speed > 0.0
                    else 0.0
                )
                self.gui_state["eta"] = float(max(0.0, remaining))

        legacy_gui.TrainingGUI._ingest_snapshot = _v26_ingest_snapshot
        legacy_gui.TrainingGUI._v26_ingest_patched = True

    if not getattr(legacy_gui.TrainingGUI, "_v26_progress_target_patched", False):
        original_start_training = legacy_gui.TrainingGUI._start_training
        original_extend_training = legacy_gui.TrainingGUI._extend_training

        def _v26_start_training(self) -> None:
            planned_total = _v26_planned_total_batches(self)
            if planned_total > 0:
                self.gui_state["total_batches"] = max(int(self.gui_state.get("batch", 0)), planned_total)
            original_start_training(self)
            if planned_total > 0:
                self.gui_state["total_batches"] = max(int(self.gui_state.get("batch", 0)), planned_total)

        def _v26_extend_training(self) -> None:
            planned_total = _v26_planned_total_batches(self)
            if planned_total > 0:
                self.gui_state["total_batches"] = max(int(self.gui_state.get("batch", 0)), planned_total)
                return
            original_extend_training(self)

        legacy_gui.TrainingGUI._start_training = _v26_start_training
        legacy_gui.TrainingGUI._extend_training = _v26_extend_training
        legacy_gui.TrainingGUI._v26_progress_target_patched = True

    if not getattr(legacy_gui.TrainingGUI, "_v26_save_load_text_patched", False):
        def _v26_save_model(self) -> None:
            self.trainer.save_checkpoint(default_checkpoint_path)
            legacy_gui.os.makedirs(legacy_gui.DEFAULT_RESULTS_DIR, exist_ok=True)
            image_filename = f"{Path(default_checkpoint_filename).stem}_training_tab.png"
            image_path = legacy_gui.os.path.join(legacy_gui.DEFAULT_RESULTS_DIR, image_filename)
            image_saved, detail = self._save_training_tab_image(image_path)
            if image_saved:
                legacy_gui.messagebox.showinfo(
                    "Save",
                    f"Resumable v26 checkpoint saved to:\n{default_checkpoint_path}\n\nTraining tab image saved to:\n{detail}",
                )
            else:
                legacy_gui.messagebox.showwarning(
                    "Save",
                    f"Resumable v26 checkpoint saved to:\n{default_checkpoint_path}\n\nTraining tab image export failed:\n{detail}",
                )

        def _v26_load_model(self) -> None:
            if self.training_running:
                legacy_gui.messagebox.showwarning("Load", "Stop training before loading a checkpoint.")
                return

            initial_dir = legacy_gui.os.path.dirname(default_checkpoint_path)
            initial_file = legacy_gui.os.path.basename(default_checkpoint_path)
            path = legacy_gui.filedialog.askopenfilename(
                title="Load Checkpoint",
                initialdir=initial_dir if legacy_gui.os.path.isdir(initial_dir) else legacy_gui.PROJECT_ROOT,
                initialfile=initial_file,
                filetypes=[("PyTorch Checkpoints", "*.pt *.pth"), ("All Files", "*.*")],
            )
            if not path:
                return

            try:
                self.trainer.load_checkpoint(path)
            except Exception as exc:
                legacy_gui.messagebox.showerror("Load Error", str(exc))
                return

            self._reset_histories()
            self.training_start_time = None
            self._last_eval_report = None
            self.protocol("WM_DELETE_WINDOW", self._on_close)
            self.gui_state["speed"] = 0.0
            self.gui_state["eta"] = 0.0
            self.gui_state["status"] = "Loaded"
            snapshot = self.trainer.get_snapshot()
            planned_total = _v26_planned_total_batches(self)
            current_target = planned_total if planned_total > 0 else self._read_int(self.spin_batches, 524288)
            self.gui_state["total_batches"] = max(int(snapshot.traversals_completed), int(current_target))
            self._ingest_snapshot(snapshot, 0, 0.0, append_history=False)
            self.update_gui()
            legacy_gui.messagebox.showinfo(
                "Load",
                f"Checkpoint loaded:\n{path}\n\nIf this is a resumable v26 checkpoint, training can continue from the saved state.",
            )

        legacy_gui.TrainingGUI._save_model = _v26_save_model
        legacy_gui.TrainingGUI._load_model = _v26_load_model
        legacy_gui.TrainingGUI._v26_save_load_text_patched = True
    return legacy_gui


def launch_gui() -> int:
    legacy_gui = configure_legacy_gui_module()
    app = legacy_gui.TrainingGUI()
    if getattr(app.trainer.config, "training_monitor_mode", "") == "phased" and hasattr(app.trainer, "planned_total_traversals"):
        planned_total = int(app.trainer.planned_total_traversals())
        app.spin_batches.delete(0, "end")
        app.spin_batches.insert(0, str(planned_total))
        app.gui_state["total_batches"] = max(int(app.gui_state.get("batch", 0)), planned_total)
    try:
        app.mainloop()
    finally:
        try:
            app.trainer.shutdown()
        except Exception:
            pass
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="v26 wrapper for the external 6-player no-limit hold'em DeepCFR repo."
    )
    parser.add_argument("--repo-root", type=str, default=None, help="Path to the external DeepCFR checkout.")
    parser.add_argument("--results-root", type=str, default=None, help="Directory for v26 training runs.")
    parser.add_argument("--nickname", type=str, default=None, help="Nickname prefix for new v26 runs.")

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status", help="Show current v26 artifact paths and defaults.")

    install = subparsers.add_parser("install", help="Install the external v26 repo requirements into the active Python env.")
    install.add_argument("--upgrade-pip", action="store_true")

    train = subparsers.add_parser("train", help="Start a new v26 training run against random opponents.")
    train.add_argument("--iterations", type=int, default=1000)
    train.add_argument("--traversals", type=int, default=200)
    train.add_argument("--verbose", action="store_true")
    train.add_argument("--strict", action="store_true")

    resume = subparsers.add_parser("resume", help="Continue v26 training from a checkpoint.")
    resume.add_argument("--checkpoint", type=str, default=None)
    resume.add_argument("--iterations", type=int, default=1000)
    resume.add_argument("--traversals", type=int, default=200)
    resume.add_argument("--verbose", action="store_true")
    resume.add_argument("--strict", action="store_true")

    self_play = subparsers.add_parser("self-play", help="Run v26 self-play training from a checkpoint.")
    self_play.add_argument("--checkpoint", type=str, default=None)
    self_play.add_argument("--iterations", type=int, default=2000)
    self_play.add_argument("--traversals", type=int, default=400)
    self_play.add_argument("--verbose", action="store_true")
    self_play.add_argument("--strict", action="store_true")

    mixed = subparsers.add_parser("mixed", help="Run v26 mixed training against a checkpoint pool.")
    mixed.add_argument("--checkpoint-dir", type=str, default=None)
    mixed.add_argument("--iterations", type=int, default=10000)
    mixed.add_argument("--traversals", type=int, default=400)
    mixed.add_argument("--model-prefix", type=str, default="t_")
    mixed.add_argument("--refresh-interval", type=int, default=1000)
    mixed.add_argument("--num-opponents", type=int, default=5)
    mixed.add_argument("--verbose", action="store_true")
    mixed.add_argument("--strict", action="store_true")

    play = subparsers.add_parser("play", help="Play the external v26 bot in the CLI. Defaults to the shipped flagship models.")
    play.add_argument("--models-dir", type=str, default=None)
    play.add_argument("--model-pattern", type=str, default="*.pt")
    play.add_argument("--num-models", type=int, default=5)
    play.add_argument("--position", type=int, default=0)
    play.add_argument("--stake", type=float, default=200.0)
    play.add_argument("--sb", type=float, default=1.0)
    play.add_argument("--bb", type=float, default=2.0)
    play.add_argument("--verbose", action="store_true")
    play.add_argument("--strict", action="store_true")
    play.add_argument("--no-shuffle", action="store_true")

    gui = subparsers.add_parser("gui", help="Launch the old Tk v26 dashboard.")
    gui.add_argument("--legacy-tk", action="store_true", help="Unused compatibility flag; the Tk dashboard is the default.")

    return parser


def build_config(args: argparse.Namespace) -> DeepCFRConfigV26:
    config = DeepCFRConfigV26()
    if args.repo_root is not None:
        config.repo_root = args.repo_root
    if args.results_root is not None:
        config.results_root = args.results_root
    if args.nickname is not None:
        config.nickname = args.nickname
    return config


def print_status(trainer: DeepCFRTrainerV26) -> None:
    latest_run = trainer.latest_run_dir()
    latest_checkpoint = trainer.latest_checkpoint()
    preferred_models_dir = trainer.preferred_models_dir()
    print(f"Repo Root:          {trainer.repo_root}")
    print(f"Results Root:       {trainer.results_root}")
    print(f"Flagship Models:    {trainer.flagship_models_dir}")
    print(f"Flagship Checkpoint:{trainer.flagship_checkpoint if trainer.flagship_checkpoint.exists() else '(missing)'}")
    print(f"Latest Run:         {latest_run or '(none)'}")
    print(f"Latest Checkpoint:  {latest_checkpoint or '(none)'}")
    print(f"Default Play Models:{preferred_models_dir or '(none)'}")


def run_cli(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "gui":
        return launch_gui()

    trainer = DeepCFRTrainerV26(build_config(args))
    if args.command == "status":
        print_status(trainer)
        return 0
    if args.command == "install":
        trainer.install_requirements(upgrade_pip=bool(args.upgrade_pip))
        print("v26 requirements installed")
        return 0
    if args.command == "train":
        run_dir = trainer.start_training(
            iterations=args.iterations,
            traversals=args.traversals,
            verbose=bool(args.verbose),
            strict=bool(args.strict),
        )
        print(f"v26 training artifacts written to {run_dir}")
        return 0
    if args.command == "resume":
        run_dir = trainer.resume_training(
            checkpoint_path=args.checkpoint,
            iterations=args.iterations,
            traversals=args.traversals,
            verbose=bool(args.verbose),
            strict=bool(args.strict),
        )
        print(f"v26 resumed training artifacts written to {run_dir}")
        return 0
    if args.command == "self-play":
        run_dir = trainer.self_play_training(
            checkpoint_path=args.checkpoint,
            iterations=args.iterations,
            traversals=args.traversals,
            verbose=bool(args.verbose),
            strict=bool(args.strict),
        )
        print(f"v26 self-play artifacts written to {run_dir}")
        return 0
    if args.command == "mixed":
        run_dir = trainer.mixed_training(
            checkpoint_dir=args.checkpoint_dir,
            iterations=args.iterations,
            traversals=args.traversals,
            model_prefix=args.model_prefix,
            refresh_interval=args.refresh_interval,
            num_opponents=args.num_opponents,
            verbose=bool(args.verbose),
            strict=bool(args.strict),
        )
        print(f"v26 mixed-training artifacts written to {run_dir}")
        return 0
    if args.command == "play":
        models_dir = trainer.play_cli(
            models_dir=args.models_dir,
            model_pattern=args.model_pattern,
            num_models=args.num_models,
            position=args.position,
            stake=args.stake,
            sb=args.sb,
            bb=args.bb,
            verbose=bool(args.verbose),
            strict=bool(args.strict),
            no_shuffle=bool(args.no_shuffle),
        )
        print(f"v26 CLI play used models from {models_dir}")
        return 0
    parser.error(f"Unknown command '{args.command}'")
    return 2


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        return launch_gui()
    return run_cli(args)


if __name__ == "__main__":
    raise SystemExit(main())
