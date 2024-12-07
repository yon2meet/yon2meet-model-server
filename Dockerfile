FROM python:3.12

RUN apt-get update
RUN apt-get install -y libgl1

COPY . /app
WORKDIR /app
RUN mkdir uploads
RUN python3 -m pip install -r requirements.txt
CMD flask --app=application run --host=0.0.0.0 --port=8080

EXPOSE 8080