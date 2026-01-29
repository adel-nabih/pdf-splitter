# Use the slim version to keep the image small
FROM python:3.11-slim

WORKDIR /app

# Install only the essential system libraries
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to take advantage of Docker caching
COPY requirements.txt .

# --no-cache-dir is the magic flag that prevents build timeouts
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything else (including your 5.4MB best.pt)
COPY . .

# Railway uses the $PORT variable automatically
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT}