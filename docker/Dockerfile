FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip

# Install watchdog for file monitoring in development
RUN pip install watchdog[watchmedo]

# Install your requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (this will be overridden by volume mount in dev)
COPY backend/ backend/

# Create directories
RUN mkdir -p /app/backend/rag/temp_audioFiles \
    /app/backend/rag/cache \
    /app/backend/rag/vector_stores

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Default command (can be overridden in docker-compose)
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]