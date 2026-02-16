# Mini-Enclava: Single Container with HTMX/Jinja2 Frontend
# Multi-stage build for production

# =============================================================================
# Stage 1: Build Tailwind CSS
# =============================================================================
FROM node:18-alpine AS css-builder

WORKDIR /build

# Copy package files
COPY backend/package.json backend/package-lock.json* ./

# Install dependencies
RUN --mount=type=cache,target=/root/.npm \
    npm ci --no-audit --no-fund || npm install --no-audit --no-fund

# Copy Tailwind config and source files
COPY backend/tailwind.config.js ./
COPY backend/app/static/css/input.css ./app/static/css/
COPY backend/app/templates ./app/templates

# Build CSS
RUN npm run build:css

# =============================================================================
# Stage 2: Python Application
# =============================================================================
FROM python:3.11-slim

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

WORKDIR /app

# Install system dependencies
ARG INSTALL_PG_DEPS="true"
ARG INSTALL_BUILD_DEPS="true"
ARG INSTALL_PDF_DEPS="true"
ARG REQUIREMENTS_FILE="backend/requirements.txt"
RUN apt-get update && \
    if [ "$INSTALL_PG_DEPS" = "true" ]; then \
        apt-get install -y --no-install-recommends \
            $([ "$INSTALL_BUILD_DEPS" = "true" ] && echo "build-essential") \
            libpq-dev \
            postgresql-client \
            $([ "$INSTALL_PDF_DEPS" = "true" ] && echo "poppler-utils") ; \
    else \
        apt-get install -y --no-install-recommends \
            $([ "$INSTALL_BUILD_DEPS" = "true" ] && echo "build-essential") \
            $([ "$INSTALL_PDF_DEPS" = "true" ] && echo "poppler-utils") ; \
    fi && \
    rm -rf /var/lib/apt/lists/*

# Optional CPU-only PyTorch install for ML-dependent features (disabled by default in sqlite/extract-only flows)
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

# Copy application code
COPY backend/app ./app
COPY backend/alembic ./alembic
COPY backend/alembic.ini .
COPY backend/scripts/migrate.sh /usr/local/bin/migrate.sh
RUN chmod +x /usr/local/bin/migrate.sh

# Copy compiled CSS from builder
COPY --from=css-builder /build/app/static/css/tailwind.css ./app/static/css/

# Copy static assets (JS files)
COPY backend/app/static/js ./app/static/js

# Create logs directory
RUN mkdir -p logs

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD ["python", "-c", "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=3).status == 200 else 1)"]

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
