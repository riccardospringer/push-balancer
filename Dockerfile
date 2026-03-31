FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY push-balancer-server.py .
COPY push-balancer.html .
COPY push_title_agent.py .
EXPOSE 8050
CMD ["python", "push-balancer-server.py"]
