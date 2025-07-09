FROM python:3.10-alpine AS builder
WORKDIR /usr/src/app
# Install git and build dependencies
RUN apk add --no-cache \
    gcc \
    g++ \
    musl-dev \
    linux-headers \
    python3-dev \
    git



COPY requirements.txt ./
RUN pip install --no-cache-dir --user -r requirements.txt|| true \
    && mkdir -p /root/.local

FROM python:3.10-alpine

WORKDIR /usr/src/app

# Install ONLY runtime dependencies (just git)
RUN apk add --no-cache git
# Copy only the installed Python packages
COPY --from=builder /root/.local /root/.local

COPY . .
ENV PATH=/root/.local/bin:$PATH

CMD [ "python", "main.py"]
