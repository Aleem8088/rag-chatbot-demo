FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY data ./data

# Run as a non-root user
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

# ANTHROPIC_API_KEY is passed at runtime (never baked into the image).
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "120", "app.main:app"]
