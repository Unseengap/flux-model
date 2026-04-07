.PHONY: test install lint clean

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

test-quick:
	pytest tests/ -q

clean:
	rm -rf build/ dist/ *.egg-info flx.egg-info/ .pytest_cache/ __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
