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
    checkpoint_dir = Path(__file__).parent.parent.parent / "training" / "checkpoints"
    
    if not checkpoint_dir.exists():
        return False
    
    # Check if any model files exist
    model_files = list(checkpoint_dir.glob("poker_agent_v*.pt*"))
    return len(model_files) > 0


def pytest_collection_modifyitems(config, items):
    """Skip tests that require models if models aren't available."""
    checkpoint_dir = Path(__file__).parent.parent.parent / "training" / "checkpoints"
    
    models_available = False
    if checkpoint_dir.exists():
        model_files = list(checkpoint_dir.glob("poker_agent_v*.pt*"))
        models_available = len(model_files) > 0
    
    if not models_available:
        skip_models = pytest.mark.skip(reason="Model checkpoint files not available")
        for item in items:
            if "requires_models" in item.keywords:
                item.add_marker(skip_models)
