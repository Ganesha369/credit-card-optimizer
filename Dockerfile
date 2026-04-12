FROM python:3.10-slim
# Force logs to show up in Hugging Face immediately
ENV PYTHONUNBUFFERED=1
WORKDIR /app
RUN pip install --no-cache-dir fastapi uvicorn pydantic openai
COPY . .
ENV PYTHONPATH=/app
EXPOSE 7860
# Direct path to the real app
CMD ["uvicorn", "credit_card_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]