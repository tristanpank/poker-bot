"""Pytest configuration for backend tests."""

import pytest
from pathlib import Path


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", 
        "requires_models: mark test as requiring model checkpoint files"
    )


@pytest.fixture(scope="session")
def models_available():
    """Check if model checkpoint files are available."""
    project_root = Path(__file__).parent.parent.parent
    candidate_dirs = [
        project_root / "training" / "models",
        project_root / "training" / "checkpoints",
    ]

    for checkpoint_dir in candidate_dirs:
        if not checkpoint_dir.exists():
            continue
        model_files = list(checkpoint_dir.glob("poker_agent_v*.pt*"))
        if model_files:
            return True
    return False


def pytest_collection_modifyitems(config, items):
    """Skip tests that require models if models aren't available."""
    project_root = Path(__file__).parent.parent.parent
    candidate_dirs = [
        project_root / "training" / "models",
        project_root / "training" / "checkpoints",
    ]

    models_available = False
    for checkpoint_dir in candidate_dirs:
        if not checkpoint_dir.exists():
            continue
        if list(checkpoint_dir.glob("poker_agent_v*.pt*")):
            models_available = True
            break

    if not models_available:
        skip_models = pytest.mark.skip(reason="Model checkpoint files not available")
        for item in items:
            if "requires_models" in item.keywords:
                item.add_marker(skip_models)
