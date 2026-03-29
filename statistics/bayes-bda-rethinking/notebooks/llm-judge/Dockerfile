FROM python:3.11-slim

# HF Spaces requires the app to run on port 7860
ENV PORT=7860
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# System deps needed by pytensor (C compiler for PyMC)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer-cached)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source
COPY app.py .
COPY llm_judge/ ./llm_judge/
COPY pages/ ./pages/
COPY data/ ./data/

# HF Spaces runs as a non-root user
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 7860

CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
