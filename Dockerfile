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

# mimOE runs natively on the host device — NOT inside this container.
# Point MIMOE_BASE_URL at host.docker.internal to reach it from Docker:
#   docker run -e MIMOE_BASE_URL=http://host.docker.internal:8083/mimik-ai/openai/v1 ...
# On Linux you must also pass: --add-host=host.docker.internal:host-gateway

ENV MIMOE_BASE_URL=http://host.docker.internal:8083/mimik-ai/openai/v1
ENV MIMOE_API_KEY=1234
ENV MIMOE_MODEL=smollm-360m
ENV LOG_LEVEL=INFO

ENTRYPOINT ["python", "-m", "agent.main"]
