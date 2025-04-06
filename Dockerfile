FROM python:3.8-slim-buster

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        awscli \
        ffmpeg \
        libsm6 \
        libxext6 \
        unzip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy code into the container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run the app
CMD ["python3", "app.py"]

