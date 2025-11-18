# ==========================================
# Research Assistant GraphRAG - Backend Dockerfile
# ==========================================

FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Create app user for security
RUN useradd --create-home --shell /bin/bash app \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libpq-dev \
        curl \
        procps \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/uploads data/sample_papers evaluation/results logs /tmp/uploads && \
    chown -R app:app /app

# Switch to non-root user
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "main.py"]
