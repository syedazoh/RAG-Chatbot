# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies needed for ChromaDB and other tools
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Hugging Face Spaces runs on port 7860 by default
EXPOSE 7860

# Command to run your app (adjust 'src.main:app' to your actual file/app name)
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "7860"]