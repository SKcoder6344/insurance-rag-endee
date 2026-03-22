FROM python:3.11-slim

WORKDIR /app

# Install system deps for sentence-transformers and endee-model
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create empty __init__.py so scripts/ is importable
RUN touch scripts/__init__.py app/__init__.py

EXPOSE 8000 8501
