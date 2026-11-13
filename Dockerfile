# Base image for Python 3.13 slim version
FROM python:3.13-slim@sha256:ffb752e139c0a19692a43af8d8523b274222dd68eebad5d583b45c2201c6e30a

# Set environment variables to optimize Python behavior
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Set the working directory
WORKDIR /app

# Create a non-root user
RUN useradd --create-home --shell /bin/bash appuser

# Copy dependency definitions first to leverage Docker layer caching
COPY pyproject.toml README.md /app/

# Install dependencies
RUN pip install --upgrade pip && \
    pip install .

# Copy the rest of the application code
COPY --chown=appuser:appuser src/ ./src
COPY --chown=appuser:appuser app/ ./app

# Create MLflow storage directory owned by appuser
RUN mkdir -p /app/mlflow-store && chown -R appuser:appuser /app/mlflow-store

# Switch to the non-root user
USER appuser

# Export the port the app runs
EXPOSE 8000

# Native container health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]