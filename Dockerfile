FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 7860

ENV PORT=7860

CMD ["python", "-m", "streamlit", "run", "src/app.py", \
     "--server.port", "7860", \
     "--server.address", "0.0.0.0", \
     "--server.headless", "true", \
     "--server.enableCORS", "false", \
     "--server.enableXsrfProtection", "false"]