FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements-prod.txt /app/requirements-prod.txt
RUN pip install --prefer-binary \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.3.1 torchvision==0.18.1 \
    && pip install --prefer-binary -r /app/requirements-prod.txt

COPY app /app/app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
