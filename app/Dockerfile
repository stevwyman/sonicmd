FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY stenopod ./stenopod
COPY static ./static

ENV DATA_DIR=/data
EXPOSE 8080

CMD ["uvicorn", "stenopod.main:app", "--host", "0.0.0.0", "--port", "8080"]
