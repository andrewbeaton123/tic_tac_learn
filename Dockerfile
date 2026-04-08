# Build stage
FROM python:3.10-slim AS builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy poetry files first
COPY pyproject.toml poetry.lock* ./

# Copy package files
COPY tic_tac_learn ./tic_tac_learn/

# Install poetry and dependencies
RUN pip install poetry \
    && poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# Runtime stage
FROM python:3.10-slim

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.10/site-packages/ /usr/local/lib/python3.10/site-packages/
COPY --from=builder /app/tic_tac_learn ./tic_tac_learn

ENV PYTHONPATH=/app \
    TICLEARN_ENV=development

CMD ["python", "-m", "tic_tac_learn"]
