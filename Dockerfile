FROM python:3.11-slim

WORKDIR /app

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# EXPOSE the port for Hugging Face
EXPOSE 7860

# THE FIX: 
# 1. Start the FastAPI server in the background (&)
# 2. Wait 10 seconds to ensure the server is fully "Live"
# 3. Run the inference script
CMD uvicorn credit_card_env.server.app:app --host 0.0.0.0 --port 7860 & sleep 10 && python inference.py