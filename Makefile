.PHONY: install lint typecheck test test-cov run server docker-build docker-run docker-server clean audit

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

server:
	uvicorn agent.server:app --reload --port 3000

docker-build:
	docker build -t mimoe-agent .

docker-run:
	docker run --rm \
		-e MIMOE_BASE_URL=http://host.docker.internal:8083/mimik-ai/openai/v1 \
		-e MIMOE_API_KEY=1234 \
		-e MIMOE_MODEL=smollm-360m \
		mimoe-agent python -m agent.main

docker-server:
	docker run --rm -p 3000:3000 \
		-e MIMOE_BASE_URL=http://host.docker.internal:8083/mimik-ai/openai/v1 \
		-e MIMOE_API_KEY=1234 \
		-e MIMOE_MODEL=smollm-360m \
		mimoe-agent

audit:
	pip-audit

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
