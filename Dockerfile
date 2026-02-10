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
RUN npm ci || npm install

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
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        postgresql-client \
        curl \
        ffmpeg \
        poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch (for vision models)
RUN pip install --no-cache-dir torch==2.5.1+cpu torchaudio==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu -f https://download.pytorch.org/whl/torch_stable.html

# Copy requirements and install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

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
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
