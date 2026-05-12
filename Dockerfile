FROM python:3.13-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ ./app/
COPY dist-frontend/ ./dist-frontend/
COPY push-balancer.html ./push-balancer.html
COPY push_title_agent.py .
# Non-root user — /data (Render persistent disk) wird via Entrypoint chowned
RUN adduser --disabled-password --gecos '' appuser && chown -R appuser /app
EXPOSE 8050
# Render mounted /data als root — chown + chmod als root, damit appuser sicher schreiben kann.
# - mkdir -p falls /data zum Startzeitpunkt noch nicht existiert (Mount-Race)
# - chown versucht Ownership, Fehler werden geloggt, brechen aber nicht ab
# - chmod o+rw als Belt+Suspenders: selbst wenn chown failed, ist /data schreibbar
CMD ["sh", "-c", "mkdir -p /data; chown -R appuser /data 2>&1 | head -3 || true; chmod -R u+rw,g+rw,o+rw /data 2>&1 | head -3 || true; ls -la /data | head -3; exec su -s /bin/sh appuser -c 'uvicorn app.main:app --host 0.0.0.0 --port 8050'"]
