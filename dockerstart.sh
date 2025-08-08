#!/bin/bash
# docker/deploy.sh - Deployment script

set -e

echo "🚀 Deploying Apollo Voicebot with SSL..."

# Check if environment file exists
if [ ! -f .env ]; then
    echo "❌ .env file not found. Please create one based on sample.env"
    exit 1
fi

# Load environment variables
source .env
# Build and start services
echo "📦 Building Docker images..."
docker-compose -f docker/docker-compose.yml down

# Build and start services
echo "📦 Building Docker images..."
docker-compose -f docker/docker-compose.yml build

echo "🔧 Starting services..."
docker-compose -f docker/docker-compose.yml up -d

echo "⏳ Waiting for services to be ready..."
sleep 10

# Check service health
echo "🏥 Checking service health..."
docker-compose -f docker/docker-compose.yml ps

echo "✅ Deployment complete!"
echo "🌐 Application is running on https://localhost:443"
echo "📊 View logs with: docker-compose -f docker/docker-compose.yml logs -f web"