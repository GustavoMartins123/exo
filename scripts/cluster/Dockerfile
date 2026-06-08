FROM python:3.13-slim

WORKDIR /app
COPY node_agent.py controller.py /app/

EXPOSE 8765 8766

CMD ["python", "/app/node_agent.py"]
