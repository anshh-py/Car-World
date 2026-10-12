FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY backend /app/backend
COPY Data /app/Data
COPY app.py /app/app.py
COPY car_database.py /app/car_database.py
WORKDIR /app/backend

EXPOSE 8000

CMD ["sh", "-c", "alembic upgrade head && python scripts/import_cars.py && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
