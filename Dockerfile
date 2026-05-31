FROM python:3.12-slim

# Non-root user for security
RUN useradd -m -u 1000 agentuser

WORKDIR /app

# Install dependencies (layer-cached separately from source code)
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e "." && pip install --no-cache-dir httpx

# Copy source code
COPY agent/ ./agent/

USER agentuser

# Non-sensitive defaults. Pass MIMOE_API_KEY at runtime — never bake secrets
# into an image. On Linux add: --add-host=host.docker.internal:host-gateway
ENV MIMOE_BASE_URL=http://host.docker.internal:8083/mimik-ai/openai/v1
ENV MIMOE_MODEL=smollm-360m
ENV LOG_LEVEL=INFO
ENV SERVER_HOST=0.0.0.0
ENV SERVER_PORT=3000

EXPOSE 3000

# Default: HTTP server (for mim / container deployment)
# Override for CLI:  docker run ... python -m agent.main -q "..."
ENTRYPOINT ["python", "-m", "agent.server"]
