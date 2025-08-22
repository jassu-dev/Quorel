FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt \
 && python -m nltk.downloader vader_lexicon

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
