FROM python:3.12-slim

WORKDIR /app

COPY . /app

RUN pip install -r requirement.txt

ENV FLASK_APP=flaskr
ENV FLASK_ENV=development
ENV FLASK_RUN_HOST='0.0.0.0'
ENV FLASK_RUN_PORT=80

EXPOSE 80

CMD ["flask", "run"]