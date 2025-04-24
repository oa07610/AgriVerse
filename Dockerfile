 FROM python:3.11-slim

 WORKDIR /app

 # 1) install netcat and dos2unix first
 RUN apt-get update \
  && apt-get install -y netcat-openbsd dos2unix \
  && rm -rf /var/lib/apt/lists/*

 # 2) install Python deps
 COPY requirements.txt .
 RUN pip install --no-cache-dir -r requirements.txt

 # 3) copy your entire app (including wait-for-db.sh)
 COPY . .

 # 4) normalize line endings and make it executable
 RUN dos2unix wait-for-db.sh \
  && chmod +x wait-for-db.sh

 EXPOSE 5000
 ENV FLASK_APP=app.py


# 5) entrypoint runs wait-for-db.sh (it’ll use $1/$2 *or* fall back to DB_HOST/DB_PORT)
ENTRYPOINT ["./wait-for-db.sh"]
CMD ["gunicorn","--bind","0.0.0.0:5000","app:app"]
