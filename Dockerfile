FROM python:3.10-slim

WORKDIR /app

# Install basic dependencies directly to avoid 'pyproject.toml' missing errors
RUN pip install --no-cache-dir fastapi uvicorn pydantic

# Copy the entire project
COPY . .

# Set the path so Python can find your 'credit_card_env' folder
ENV PYTHONPATH=/app

# EXPOSE the HF port
EXPOSE 7860

# Point exactly to where your app.py is now
CMD ["uvicorn", "credit_card_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]