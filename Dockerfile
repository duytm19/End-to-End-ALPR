FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app


COPY requirements.txt .

RUN pip install --no-cache-dir --timeout=900 -r requirements.txt

COPY ./src ./src
COPY ./app.py .
COPY ./models ./models

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.enableCORS=false"]