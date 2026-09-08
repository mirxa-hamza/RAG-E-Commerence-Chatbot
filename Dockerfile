FROM python:3.13-slim AS builder
WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt requirements-agent.txt ./
RUN pip install --no-cache-dir -r requirements-agent.txt

FROM python:3.13-slim AS runtime
ENV PATH="/opt/venv/bin:$PATH" PYTHONUNBUFFERED=1 HF_HOME=/app/model-cache
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
RUN useradd --uid 10001 --create-home app && mkdir -p /app/storage /app/model-cache && chown -R app:app /app
COPY --chown=app:app src ./src
COPY --chown=app:app scripts ./scripts
USER app
EXPOSE 8000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
