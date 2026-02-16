FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

# Set work directory
WORKDIR /app

# Install system dependencies
ARG BUILD_DNS_SERVERS="10.0.2.3 10.88.0.1 1.1.1.1 8.8.8.8"
ARG INSTALL_PG_DEPS="true"
ARG INSTALL_BUILD_DEPS="true"
ARG REQUIREMENTS_FILE="requirements.txt"
RUN set -eux; \
    # Work around rootless userns + systemd-resolved DNS during image builds
    if ! grep -q "^nameserver" /etc/resolv.conf || grep -q "127.0.0.53" /etc/resolv.conf; then \
        : > /etc/resolv.conf; \
        for ns in $BUILD_DNS_SERVERS; do echo "nameserver $ns" >> /etc/resolv.conf; done; \
        echo "options ndots:0" >> /etc/resolv.conf; \
    fi; \
    echo 'APT::Sandbox::User "root";' > /etc/apt/apt.conf.d/99rootless; \
    apt_lists=/tmp/apt-lists; \
    apt_cache=/tmp/apt-cache; \
    mkdir -p "$apt_lists/partial" "$apt_cache/partial"; \
    DEBIAN_FRONTEND=noninteractive apt-get \
        -o Acquire::ForceIPv4=true \
        -o Dir::State::Lists="$apt_lists" \
        -o Dir::Cache::Archives="$apt_cache" \
        update; \
        DEBIAN_FRONTEND=noninteractive apt-get \
        -o Acquire::ForceIPv4=true \
        -o Dir::State::Lists="$apt_lists" \
        -o Dir::Cache::Archives="$apt_cache" \
        install -y --no-install-recommends \
            $([ "$INSTALL_BUILD_DEPS" = "true" ] && echo "build-essential") \
            curl \
            poppler-utils \
            $([ "$INSTALL_PG_DEPS" = "true" ] && echo "libpq-dev postgresql-client") \
    ; \
    rm -rf "$apt_lists" "$apt_cache"

# Install CPU-only PyTorch and compatible numpy first (faster download)
ARG INSTALL_TORCH="true"
ARG TORCH_VERSION="2.6.0"
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$INSTALL_TORCH" = "true" ]; then \
    pip install --no-cache-dir \
        torch==${TORCH_VERSION}+cpu \
        --index-url https://download.pytorch.org/whl/cpu \
        -f https://download.pytorch.org/whl/torch_stable.html ; \
else \
    echo "Skipping PyTorch installation (INSTALL_TORCH=$INSTALL_TORCH)"; \
fi

# Copy requirements and install Python dependencies
COPY ${REQUIREMENTS_FILE} ./requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Optional: Download spaCy English model for NLP processing (commented out for faster builds)
# Uncomment if you install requirements-nlp.txt and need entity extraction
# RUN python -m spacy download en_core_web_sm

# Copy application code
COPY . .

# Copy and make startup script executable
COPY scripts/start.sh /usr/local/bin/start.sh
RUN chmod +x /usr/local/bin/start.sh

# Copy and make migration script executable
COPY scripts/migrate.sh /usr/local/bin/migrate.sh
RUN chmod +x /usr/local/bin/migrate.sh

# Create logs directory
RUN mkdir -p logs

# Default uvicorn options for this image
ENV UVICORN_ARGS="--host 0.0.0.0 --port 8000 --reload"

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["/usr/local/bin/start.sh"]
