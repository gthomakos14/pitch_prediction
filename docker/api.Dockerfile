FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir fastapi uvicorn[standard] pydantic pyyaml torch scikit-learn numpy polars mlflow

COPY src/ ./src/
COPY api/ ./api/
COPY models/artifacts/ ./models/artifacts/

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
