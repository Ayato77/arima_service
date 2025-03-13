FROM python:3.9

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY arima_api.py .

CMD ["uvicorn", "arima_api:app", "--host", "0.0.0.0", "--port", "8090"]
