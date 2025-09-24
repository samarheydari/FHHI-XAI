# Optimized Dockerfile for Python application
# Uses multi-stage build and improved layer caching

# Build stage for installing dependencies
FROM python:3.8.12-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    # Therse two are for h5py
    pkg-config \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy and install requirements first (better layer caching)
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /build/wheels -r requirements.txt

# Final stage
FROM python:3.8.12-slim

# Set environment variables
ENV PORT=8080 \
    BASE_PATH='/tfa02' \
    BROKER_URL=https://orion.tema.digital-enabler.eng.it \
    DEBUG=False \
    PROCESSING_UNIT=gpu \
    REDIS_HOST=localhost \
    REDIS_PORT=6379

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    redis-server \
    supervisor \
    ffmpeg \
    libsm6 \
    libxext6 \
    libhdf5-103 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /var/log/supervisor /var/lib/redis \
    && chown -R redis:redis /var/lib/redis \
    && rm -f /var/lib/redis/dump.rdb

# Copy wheels from builder stage and install
COPY --from=builder /build/wheels /wheels
RUN pip install --no-cache-dir /wheels/*

# Configure supervisor
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Copy application code (done last to maximize caching)
COPY . .

# Expose port
EXPOSE ${PORT}

# Start supervisord
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]