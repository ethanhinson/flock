#!/usr/bin/env bash
# Write a self-signed certificate for the coordinator into .certs/.
#
# Why this exists at all: Chrome and Edge expose WebGPU only in a secure
# context, so a phone opening http://<lan-ip>:8000/flock sees no navigator.gpu
# and looks like a device with no GPU. Safari does not gate it this way, which is
# why the flock worked on iPhones before anyone noticed. Serving https with a
# certificate each device trusts once makes every browser usable.
#
# The certificate names this machine's LAN address, so it has to be regenerated
# if that address changes.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p .certs
IP=$(ipconfig getifaddr en0 2>/dev/null || hostname -I 2>/dev/null | awk '{print $1}' || echo 127.0.0.1)
openssl req -x509 -newkey rsa:2048 -nodes -days 825 \
  -keyout .certs/key.pem -out .certs/cert.pem -subj /CN=flock \
  -addext "subjectAltName=IP:$IP,IP:127.0.0.1,DNS:localhost" 2>/dev/null
echo "certificate for $IP written to .certs/ -- restart the coordinator, then open"
echo "https://$IP:8000 on each device once and accept the warning"
