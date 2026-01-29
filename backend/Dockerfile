FROM python:3.11-slim

WORKDIR /app

# The "Fail-Proof" update: 
# 1. Clear existing lists 
# 2. Update with a fix for 'missing' repositories
# 3. Install dependencies
RUN apt-get clean && apt-get update --fix-missing && apt-get install -y \
    tesseract-ocr \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install without cache to keep image small
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT}