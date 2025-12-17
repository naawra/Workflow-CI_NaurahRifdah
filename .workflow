name: CI Pipeline MLOps

on:
  push:
    branches: [ "main" ]
  workflow_dispatch:

env:
  MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
  MLFLOW_TRACKING_USERNAME: ${{ secrets.MLFLOW_TRACKING_USERNAME }}
  MLFLOW_TRACKING_PASSWORD: ${{ secrets.MLFLOW_TRACKING_PASSWORD }}
  DAGSHUB_USER_TOKEN: ${{ secrets.DAGSHUB_TOKEN }}

jobs:
  build-and-train:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout Code
      uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'

    - name: Install Dependencies
      run: |
        pip install mlflow dagshub pandas scikit-learn joblib matplotlib seaborn

    - name: Login to Docker Hub
      uses: docker/login-action@v2
      with:
        username: ${{ secrets.DOCKER_USERNAME }}
        password: ${{ secrets.DOCKER_PASSWORD }}

    - name: Run MLflow Project
      run: |
        cd MLProject
        mlflow run . --no-conda

    - name: Build & Push Docker Image
      uses: docker/build-push-action@v4
      with:
        context: ./MLProject
        push: true
        tags: ${{ secrets.DOCKER_USERNAME }}/telco-churn-mlops:latest