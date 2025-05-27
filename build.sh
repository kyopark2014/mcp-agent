#!/bin/bash

# Create AWS credentials file
mkdir -p .aws
cat > .aws/credentials << EOL
[default]
aws_access_key_id = $(aws configure get aws_access_key_id)
aws_secret_access_key = $(aws configure get aws_secret_access_key)
region = $(aws configure get region)
EOL

# Pass credentials to Docker build
docker build \
  --build-arg AWS_ACCESS_KEY_ID="$(aws configure get aws_access_key_id)" \
  --build-arg AWS_SECRET_ACCESS_KEY="$(aws configure get aws_secret_access_key)" \
  --build-arg AWS_DEFAULT_REGION="$(aws configure get region)" \
  -t mcp-agent .

# Clean up temporary credentials file
rm -rf .aws 