FROM python:3.11-slim
WORKDIR /regressione_lineare
COPY . .
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["python", "main.py"]