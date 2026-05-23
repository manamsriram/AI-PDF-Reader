FROM python:3.11-slim

WORKDIR /app

# System deps required by pymupdf
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p pdfFolder

EXPOSE 10000

# Single worker — BM25 and ML models live in process memory
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "--workers", "1", "--timeout", "120", "app:app"]
