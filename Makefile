.PHONY: install lint typecheck test test-cov run audit clean

install:
	pip install -e .[dev]

lint:
	ruff check agent/ tests/

lint-fix:
	ruff check --fix agent/ tests/

typecheck:
	mypy agent/

test:
	pytest -q --no-cov

test-cov:
	pytest --cov=agent --cov-report=term-missing --cov-fail-under=60

run:
	python -m agent.main

audit:
	pip-audit

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
