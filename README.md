---
title: Credit Card Optimizer
emoji: 💳
sdk: docker
app_port: 7860
---

# 💳 Credit Card Reward Optimizer

An autonomous AI agent that maximizes credit card rewards using LLMs. This project simulates real-world transactions and uses an AI "brain" to select the optimal card for maximum cashback.

## 🛠️ Tech Stack
- Language: Python 3.11
- Framework: FastAPI
- AI Model: Meta Llama-3-8B (via Hugging Face)
- Deployment: Docker & Hugging Face Spaces

## 📂 Project Structure
- inference.py: The AI Agent logic (connects to Llama-3).
- credit_card_env/: The environment simulation and server code.
- Dockerfile: Container configuration for deployment.
- requirements.txt: Python dependencies.

## 🚀 How to Run Locally
1. Clone the repository:
   git clone https://github.com/Ganesha369/credit-card-optimizer.git
2. Set up Environment Variables:
   Create a .env file and add your HF_TOKEN.
3. Run the Application:
   uvicorn credit_card_env.server.app:app --host 0.0.0.0 --port 7860 & sleep 5 && python inference.py

## 📊 Deployment
The project is hosted live on Hugging Face Spaces. The agent automatically processes tasks (Easy, Medium, Hard) and logs rewards in real-time.