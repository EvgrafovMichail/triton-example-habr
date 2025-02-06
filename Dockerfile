FROM nvcr.io/nvidia/tritonserver:23.12-py3

WORKDIR /app

EXPOSE 8000
EXPOSE 8002

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt --no-cache-dir

CMD [ "tritonserver", "--model-repository=/models" ]
