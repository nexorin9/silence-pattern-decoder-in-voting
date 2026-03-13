# Silence Pattern Decoder - Docker Image

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY silence_decoder/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY silence_decoder/ ./silence_decoder/

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Create directories for data and logs
RUN mkdir -p /app/data /app/logs /app/cache /app/results

# Expose ports (for web dashboard if needed)
EXPOSE 8501

# Set entrypoint
ENTRYPOINT ["python", "-m", "silence_decoder.src.main"]

# Default command
CMD ["--help"]
