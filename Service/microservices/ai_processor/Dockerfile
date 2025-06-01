FROM python:3.12-slim
WORKDIR /app

COPY requirements.txt .

RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /root/.cache/pip

COPY . .
CMD ["python", "main.py"]
