FROM python:3.9-slim

WORKDIR /app

COPY . /app/

RUN pip install six
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "app.py"]