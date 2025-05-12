FROM python:3.10-slim

WORKDIR /ml

RUN apt-get update && apt-get install -y gcc libpq-dev && rm -rf /var/lib/apt/lists/*

COPY requirements.ml.txt .
RUN pip install --no-cache-dir -r requirements.ml.txt

RUN python -c "import urduhack; urduhack.download()"

COPY ml_model/train_model.py .

CMD ["python", "train_model.py"]
