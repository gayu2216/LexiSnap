#!/bin/bash
# Fix SSL certificate errors for pip/requests on macOS (OSStatus -26276)
# Usage: source fix_ssl.sh   OR   . fix_ssl.sh

get_cert_path() {
    # Try project venv first
    local venv="$1/hack_ai_env"
    if [ -f "$venv/lib/python3.12/site-packages/certifi/cacert.pem" ]; then
        echo "$venv/lib/python3.12/site-packages/certifi/cacert.pem"
        return
    fi
    # Try current venv
    if [ -n "$VIRTUAL_ENV" ] && [ -f "$VIRTUAL_ENV/lib/python3.12/site-packages/certifi/cacert.pem" ]; then
        echo "$VIRTUAL_ENV/lib/python3.12/site-packages/certifi/cacert.pem"
        return
    fi
    # Fallback: use Python to find certifi
    python3 -c "import certifi; print(certifi.where())" 2>/dev/null
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CERT_PATH=$(get_cert_path "$SCRIPT_DIR")

if [ -n "$CERT_PATH" ] && [ -f "$CERT_PATH" ]; then
    export SSL_CERT_FILE="$CERT_PATH"
    export REQUESTS_CA_BUNDLE="$CERT_PATH"
    echo "SSL certificates set: $CERT_PATH"
else
    echo "Certifi not found. Run: pip install certifi"
fi
