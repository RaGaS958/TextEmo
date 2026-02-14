# ----------------------------
# Base Image
# ----------------------------
FROM python:3.11-slim

# ----------------------------
# Environment settings
# ----------------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ----------------------------
# Set working directory
# ----------------------------
WORKDIR /app

# ----------------------------
# Install system dependencies (needed for TensorFlow, etc.)
# ----------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------
# Copy requirements first (for caching)
# ----------------------------
COPY requirements.txt .

# ----------------------------
# Install Python dependencies
# ----------------------------
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# ----------------------------
# Copy project files
# ----------------------------
COPY . .

# ----------------------------
# Expose Streamlit port
# ----------------------------
EXPOSE 8501

# ----------------------------
# Run Streamlit app
# ----------------------------
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
