FROM python:3.10-slim

# RUN useradd -m -u 1000 user

# USER user
# ENV PATH="/home/user/.local/bin:$PATH"
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY data/ data/
COPY .env .

EXPOSE 7860
ENV PYTHONPATH=/app/src
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "7860"]

