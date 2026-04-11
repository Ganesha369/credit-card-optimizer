# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements/lock files first to leverage Docker cache
COPY pyproject.toml .
# If you are using uv or pip, ensure you copy the relevant lock file
# COPY uv.lock . 

# Install dependencies
RUN pip install --no-cache-dir .

# Copy the rest of your application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=7860

# EXPOSE the port FastAPI runs on
EXPOSE 7860

# CRITICAL: Point to the new location of app.py
# Format: uvicorn <folder>.<folder>.app:app
CMD ["uvicorn", "credit_card_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]