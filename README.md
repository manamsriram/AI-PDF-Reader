# AI-PDF-Reader 🗂️🤖

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20App-000?logo=flask)](https://flask.palletsprojects.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT4-10A37F?logo=openai&logoColor=white)](https://openai.com/)
[![Redis](https://img.shields.io/badge/Redis-Cache-DC382D?logo=redis&logoColor=white)](https://redis.io/)
[![RabbitMQ](https://img.shields.io/badge/RabbitMQ-Queue-FF6600?logo=rabbitmq)](https://www.rabbitmq.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

> **Intelligent chatbot that reads, indexes, and answers questions from PDFs using LLMs, caching, and robust messaging architecture.**

---

## 🔥 Overview

AI-PDF-Reader lets you "chat with your documents" — ask questions, find facts, and summarize content directly from your PDF files, powered by OpenAI's GPT-4 Turbo.

- **Upload PDFs** 👉 indexed and chunked for efficient search
- **Ask questions** 👉 answers generated from relevant excerpts
- **Multi-layered caching** with Redis and SQLite for speed
- **Distributed & Fault-Tolerant** using RabbitMQ for queueing and reliability

---

## 🛠️ Tech Stack

| ![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python) | ![Flask](https://img.shields.io/badge/-Flask-000000?logo=flask) | ![OpenAI](https://img.shields.io/badge/-OpenAI-10A37F?logo=openai) | ![Redis](https://img.shields.io/badge/-Redis-DC382D?logo=redis) | ![RabbitMQ](https://img.shields.io/badge/-RabbitMQ-FF6600?logo=rabbitmq) | ![scikit-learn](https://img.shields.io/badge/-scikit--learn-F7931E?logo=scikit-learn) | ![NumPy](https://img.shields.io/badge/-NumPy-013243?logo=numpy) | ![PyMuPDF](https://img.shields.io/badge/-PyMuPDF-4E8B93?logo=pypi) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|

## Deployment

The simplest production split for this repo is:

1. Deploy the Flask app on Render as the backend.
2. Deploy only the `templates/` folder on Vercel as a static frontend.
3. Proxy the frontend requests for `/ask` and `/upload` to the Render backend.

Set the Vercel project root directory to `templates/`. That keeps Vercel from seeing the backend `requirements.txt` at the repo root, so it only deploys the static frontend.

This repository includes a [`templates/vercel.json`](templates/vercel.json) that does the routing. Replace `https://ai-pdf-reader-ezm2.onrender.com` with your actual Render service URL.

On Render, keep the existing backend environment variables set for the Flask app, especially `GROQ_API_KEY`, `QDRANT_URL`, and `QDRANT_API_KEY`.

On Vercel, no Python runtime is needed for the frontend. There is no separate frontend requirements file here because the HTML page is static and only uses browser-side JavaScript.

---

## 🎯 Features

- **PDF-to-Text Extraction:** Uses PyMuPDF to extract text by page, chunked for semantic search
- **Smart Retrieval:** TF-IDF vectorizer and cosine similarity to find most relevant chunks
- **LLM-Powered Q&A:** OpenAI GPT-4 Turbo generates human-like, context-aware answers
- **Backend Architecture:** Flask web app with message queuing (RabbitMQ) and caching (Redis/SQLite)
- **Persistence:** Query/response history saved for instant lookup and fast response
- **Fault Tolerance:** Robust RabbitMQ setup, retries, and threads for reliability
- **Templates-based Web UI:** Simple homepage via Flask templates (`index.html`)

---

## 📦 Installation

git clone https://github.com/manamsriram/AI-PDF-Reader.git
cd AI-PDF-Reader

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt

Set up `.env` file with your keys and config:

OPENAI_API_KEY=sk-xxxxxxx
REDIS_HOST=localhost
REDIS_PORT=6379
RABBITMQ_HOST=127.0.0.1
RABBITMQ_PORT=5672
RABBITMQ_USER=guest
RABBITMQ_PASS=guest

---

## 🖥️ Usage

1. **Start the Flask app**
    ```
    python app.py
    ```
2. **Upload PDFs** to the `pdfFolder` directory.
3. **Visit the web page** and ask document questions!

---

## 🗂️ Typical Workflow

1. **Document Processing:** PDFs in `pdfFolder` become chunked/extracted for search
2. **Query Submission:** Web UI posts question to Flask endpoint `/ask`, which queues message to RabbitMQ
3. **Answer Pipeline:** 
    - Checks Redis cache
    - Checks SQLite history
    - If not cached, finds top relevant document chunks and passes them to OpenAI GPT-4 Turbo for answer generation
4. **Response Display:** Answers streamed back to UI, cached if new

---

## 🚦 Example Queries

- "List major Food insecurity reasons in 2024"  
- "Explain malnutrition in war zones"  
- "How do increased prices impact food security?"  

Answers will reference actual pages from your PDFs!

---

## 💡 Prompts & Customization

See [`prompts.txt`](prompts.txt) for example questions and responses.

---

## 🏗️ Roadmap

- Drag-and-drop PDF upload UI
- Support multiple file types (.docx, .txt)
- User authentication
- Visualization of answer position in document
- Improved semantic search

---

## 👤 Author

Sri Ram Mannam  
[GitHub](https://github.com/manamsriram) | [LinkedIn](https://www.linkedin.com/in/sri-ram-mannam-8b61aa228/)

---

## 📜 License

MIT License. See [`LICENSE`](LICENSE).

---
