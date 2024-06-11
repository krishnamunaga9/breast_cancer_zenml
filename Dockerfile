FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Set the PYTHONPATH environment variable
ENV PYTHONPATH="/app"

CMD ["python", "main.py"]
