FROM python:3.9-slim

WORKDIR /app

RUN pip install pandas scikit-learn mlflow dagshub joblib matplotlib seaborn

COPY . .

CMD ["python", "modelling.py"]