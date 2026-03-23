FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install langchain-classic==1.0.1

COPY . .

EXPOSE 8004

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8004"]
