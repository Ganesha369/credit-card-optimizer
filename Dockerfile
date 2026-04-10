FROM python:3.11-slim

WORKDIR /app

# Prevent Python from writing .pyc files (keeps the container clean)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# We expose 7860 because the server runs there, 
# but the main command should be your inference script.
CMD ["python", "inference.py"]