#!/bin/bash
# Generate SSL certificates for custom domain

DOMAIN=${1:-localhost}

echo "🔐 Generating SSL certificates for domain: $DOMAIN"

# Create ssl directory if it doesn't exist
mkdir -p docker/ssl

# Generate SSL certificates
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout docker/ssl/server.key \
    -out docker/ssl/server.crt \
    -subj "/CN=$DOMAIN"

echo "✅ SSL certificates generated:"
echo "   Certificate: docker/ssl/server.crt"
echo "   Private Key: docker/ssl/server.key"
echo ""
echo "💡 To use these certificates in production:"
echo "   1. Uncomment the SSL volume mounts in docker-compose.yml"
echo "   2. Update SSL_CERT_PATH and SSL_KEY_PATH in .env if needed"