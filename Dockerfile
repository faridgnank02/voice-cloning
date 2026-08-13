FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

RUN useradd --create-home --uid 10001 voiceagent
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY voice_agent ./voice_agent
COPY src ./src
COPY scripts ./scripts
COPY app.py README.md ./
RUN mkdir -p /app/data && chown -R voiceagent:voiceagent /app

USER voiceagent
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')"
CMD ["uvicorn", "voice_agent.server:app", "--host", "0.0.0.0", "--port", "8000"]
