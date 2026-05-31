FROM python:3.12-slim

# Non-root user for security
RUN useradd -m -u 1000 agentuser

WORKDIR /app

# Install dependencies first (layer-cached separately from source code)
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e "." && pip install --no-cache-dir httpx

# Copy source code
COPY agent/ ./agent/

# Switch to non-root user
USER agentuser

# Non-sensitive runtime defaults. Sensitive values (MIMOE_API_KEY) must be
# passed at runtime via -e or docker-compose env_file — never baked into the image.
#
# On Linux: also pass --add-host=host.docker.internal:host-gateway
ENV MIMOE_BASE_URL=http://host.docker.internal:8083/mimik-ai/openai/v1
ENV MIMOE_MODEL=smollm-360m
ENV LOG_LEVEL=INFO

ENTRYPOINT ["python", "-m", "agent.main"]
