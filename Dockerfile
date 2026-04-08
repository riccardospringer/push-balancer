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
COPY dist-frontend/ ./dist-frontend/
EXPOSE 8050
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8050}
