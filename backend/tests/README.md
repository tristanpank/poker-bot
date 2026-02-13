# Backend API Testing - Model Weights Handling

## Problem
GitHub Actions fails because model checkpoint files (~100MB+) aren't committed to git.

## Solution
Tests are split into two categories:

### ✅ Tests that run in CI (no models needed)
- `/poker/health` endpoint
- `/poker/models` endpoint  
- `/` root endpoint
- Schema validation
- Error handling

### ⏭️ Tests skipped in CI (require models)
Marked with `@pytest.mark.requires_models`:
- Actual inference tests
- Q-value validation
- Model response structure

## How It Works

`conftest.py` automatically detects if model files exist:
- **Local dev** (models present): All tests run ✅
- **GitHub CI** (no models): Model tests skipped ⏭️

## Running Tests

```bash
# Locally (with models) - all tests run
pytest backend/tests/test_api.py -v

# Simulate CI (without models)
mv training/checkpoints training/checkpoints.bak
pytest backend/tests/test_api.py -v
mv training/checkpoints.bak training/checkpoints
```

## CI Output Example
```
test_health_returns_200 ✅ PASSED
test_action_returns_200 ⏭️ SKIPPED (Model checkpoint files not available)
```
