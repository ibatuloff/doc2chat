FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

COPY ./app.py /app/app.py

COPY req.txt /app/req.txt

RUN apt-get update 
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r req.txt

CMD ["python", "/app/app.py"]