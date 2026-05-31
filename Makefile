.PHONY: install lint typecheck test test-cov run server docker-build docker-run docker-server mim-package mim-image-deploy mim-deploy mim-undeploy clean audit

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

# ── mimik mim deployment ──────────────────────────────────────────────────────
# Requires: npm install -g @mimik/mimik-edge-cli
# Requires: EDGE_TOKEN env var  (get from developer.mimik.com developer console)
#
# Full deploy sequence:
#   export EDGE_TOKEN=<your_token>
#   make mim-package          # build image + save as .tar
#   make mim-image-deploy     # load the .tar into edgeEngine
#   make mim-deploy           # start the container instance
#
# After deploy the agent is reachable at:
#   http://localhost:8083/mimoe-local-agent/v1/health
#   http://localhost:8083/mimoe-local-agent/v1/query  (POST)

mim-package:
	docker build -t mimoe-local-agent-v1 .
	docker save mimoe-local-agent-v1 -o build/mimoe-local-agent-v1.tar
	@echo "Image saved to build/mimoe-local-agent-v1.tar"

mim-image-deploy:
	@test -n "$(EDGE_TOKEN)" || (echo "ERROR: set EDGE_TOKEN=<your_edge_access_token>"; exit 1)
	mimik-edge-cli image deploy --image=build/mimoe-local-agent-v1.tar --token=$(EDGE_TOKEN)

mim-deploy:
	@test -n "$(EDGE_TOKEN)" || (echo "ERROR: set EDGE_TOKEN=<your_edge_access_token>"; exit 1)
	mimik-edge-cli container deploy --payload=start.json --token=$(EDGE_TOKEN)

mim-undeploy:
	@test -n "$(EDGE_TOKEN)" || (echo "ERROR: set EDGE_TOKEN=<your_edge_access_token>"; exit 1)
	mimik-edge-cli container delete --name=mimoe-local-agent --token=$(EDGE_TOKEN)

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
