FROM python:3.13-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ ./app/
COPY push-balancer.html .
COPY push_title_agent.py .
COPY push-balancer-server.py .
COPY push_balancer_server_compat.py .
COPY push-snapshot.json .
# Non-root user — /data (Render persistent disk) wird via Entrypoint chowned
RUN adduser --disabled-password --gecos '' appuser && chown -R appuser /app
EXPOSE 8050
# Render mounted /data als root — chown vor Start damit appuser schreiben kann
CMD ["sh", "-c", "chown -R appuser /data 2>/dev/null || true && exec su -s /bin/sh appuser -c 'uvicorn app.main:app --host 0.0.0.0 --port 8050'"]
