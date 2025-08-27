# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgdal-dev \
    gdal-bin \
    libspatialindex-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src/ ./src/
COPY notebooks/ ./notebooks/
COPY tests/ ./tests/
COPY data/ ./data/
COPY outputs/ ./outputs/
COPY tutor.md ./

# Create necessary directories
RUN mkdir -p /app/data/raw /app/data/clean /app/data/processed \
    /app/outputs/figures /app/outputs/reports

# Set permissions
RUN chmod -R 755 /app

# Expose port for Jupyter
EXPOSE 8888

# Default command - start Jupyter notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
