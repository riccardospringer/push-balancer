#!/bin/bash
# Railway Setup für Push Balancer
# Einmal ausführen: bash railway-setup.sh

set -e
RAILWAY=~/.local/node_modules/.bin/railway
cd ~/push-balancer

echo "=== 1/4: Railway Login ==="
$RAILWAY login

echo ""
echo "=== 2/4: Projekt erstellen ==="
$RAILWAY init

echo ""
echo "=== 3/4: Environment Variables setzen ==="
read -p "PUSH_API_BASE [http://push-frontend.bildcms.de]: " PUSH_API
PUSH_API=${PUSH_API:-http://push-frontend.bildcms.de}

read -p "OPENAI_API_KEY: " OAI_KEY
if [ -z "$OAI_KEY" ]; then
  # Versuche aus .env zu lesen
  OAI_KEY=$(grep -oP '(?<=AI_API_KEY=).*' .env 2>/dev/null || true)
  if [ -n "$OAI_KEY" ]; then
    echo "  (aus .env gelesen)"
  else
    echo "  WARNUNG: Kein Key angegeben, GPT-Features deaktiviert"
  fi
fi

$RAILWAY vars set "PUSH_API_BASE=$PUSH_API"
[ -n "$OAI_KEY" ] && $RAILWAY vars set "OPENAI_API_KEY=$OAI_KEY"

echo ""
echo "=== 4/4: Deploy ==="
$RAILWAY up --detach

echo ""
echo "=== Fertig! ==="
$RAILWAY domain
echo ""
echo "Oeffne die URL oben im Browser"
