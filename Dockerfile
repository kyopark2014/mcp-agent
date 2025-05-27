FROM python:3.13-slim

RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    unzip \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && curl "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install \
    && rm -rf aws awscliv2.zip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create AWS credentials directory
RUN mkdir -p /root/.aws

# Set ARG for AWS credentials
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG AWS_DEFAULT_REGION

# Create AWS credentials file
RUN echo "[default]" > /root/.aws/credentials && \
    echo "aws_access_key_id = ${AWS_ACCESS_KEY_ID}" >> /root/.aws/credentials && \
    echo "aws_secret_access_key = ${AWS_SECRET_ACCESS_KEY}" >> /root/.aws/credentials && \
    echo "region = ${AWS_DEFAULT_REGION}" >> /root/.aws/credentials

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir streamlit

RUN mkdir -p /root/.streamlit
COPY config.toml /root/.streamlit/config.toml

COPY . .

RUN npm install -g playwright
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
RUN npx playwright install --with-deps chromium

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["python", "-m", "streamlit", "run", "application/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
