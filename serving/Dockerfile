FROM python:3.12-slim
WORKDIR /app
COPY serving/requirements.txt serving/requirements.txt
RUN pip install --no-cache-dir -r serving/requirements.txt \
    && useradd --create-home --uid 10001 poc \
    && mkdir -p /app/data/audit && chown -R poc:poc /app/data
COPY serving/ serving/
COPY training/format_sft.py training/
USER poc
EXPOSE 8080
CMD ["uvicorn", "serving.api:app", "--host", "0.0.0.0", "--port", "8080", "--no-access-log"]
